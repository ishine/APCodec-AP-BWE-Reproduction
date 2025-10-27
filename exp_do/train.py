import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import json
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import glob
import math

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

class TokenDataset(Dataset):
    def __init__(self, file_path, max_len=1155, pad_id=3072):
        self.pad_id = pad_id
        self.samples = []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            tokens = [int(x) for x in item["token"].split()]
            if len(tokens) < max_len:
                tokens += [self.pad_id] * (max_len - len(tokens))
            self.samples.append(tokens)
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.long)
        return x[:-1], x[1:]

class DecoderOnlyBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        ff_out = self.ffn(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        return x

class AudioTransformer(nn.Module):
    def __init__(self, vocab_size=3073, d_model=256, nhead=4, num_layers=4, dim_feedforward=1024, dropout=0.2, max_len=1155, pad_id=4096):
        super().__init__()
        self.max_len = max_len
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderOnlyBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pad_mask = (x == self.pad_id)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask, key_padding_mask=pad_mask)
        x = self.norm(x)
        logits = self.fc_out(x)
        return logits #[B, T, vocab_size]

def get_lr(step, h, total_steps):
    warmup_steps = h.warmup_steps
    peak_lr = h.learning_rate
    min_lr = h.min_lr
    decay_steps = total_steps - warmup_steps
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    elif step > total_steps:
        return min_lr
    else:
        progress = (step - warmup_steps) / decay_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (peak_lr - min_lr) * cosine_decay

def train(h):

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(0))

    print("Initializing transformer model...")
    model = AudioTransformer(vocab_size=3073, d_model=256, nhead=4, num_layers=4, dim_feedforward=1024, dropout=0.1, max_len=1155, pad_id=3072).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2], weight_decay=1e-4)

    os.makedirs(h.transformer_checkpoint_path, exist_ok=True)
    print(f"Checkpoints directory: {h.transformer_checkpoint_path}")
    cp_model = scan_checkpoint(h.transformer_checkpoint_path, 'model_')

    steps = 0
    last_epoch = -1

    if cp_model is None:
        last_epoch = -1
    else:
        state_dict_model = load_checkpoint(cp_model, device)
        model.load_state_dict((state_dict_model['model']))
        optimizer.load_state_dict(state_dict_model['optimizer'])
        steps = state_dict_model['steps'] + 1
        last_epoch = state_dict_model['epoch']

    train_dataset = TokenDataset(h.training_data_json, max_len=1155, pad_id=3072)
    train_loader = DataLoader(train_dataset, batch_size=h.batch_size, shuffle=True, num_workers=h.num_workers, pin_memory=True, drop_last=True)

    valid_dataset = TokenDataset(h.valid_data_json, max_len=1155, pad_id=3072)
    valid_loader = DataLoader(valid_dataset, batch_size=h.batch_size, shuffle=False, num_workers=h.num_workers, pin_memory=True, drop_last=False)

    sw = SummaryWriter(os.path.join(h.transformer_checkpoint_path, 'logs'))
    criterion = nn.CrossEntropyLoss(ignore_index=3072)
    total_steps = len(train_loader) * h.training_epochs

    for epoch in range(max(0,last_epoch+1), h.training_epochs):

        #training
        model.train()
        for i, (context, target) in enumerate(train_loader):
            lr = get_lr(steps, h, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            sw.add_scalar("learning_rate", lr, steps)
            
            context, target = context.to(device), target.to(device)
            logits = model(context)
            loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
            
            #stdout logging
            if steps % h.stdout_interval == 0:
                with torch.no_grad():
                    train_loss = loss.item()
                    preds = logits.argmax(dim=-1)
                    mask = (target != 3072)
                    correct = (preds == target) & mask
                    num_correct = correct.sum().item()
                    num_tokens = mask.sum().item()
                    acc = num_correct / num_tokens if num_tokens > 0 else 0.0
                print(f"Step: {steps}, Epoch: {epoch+1}, loss: {train_loss:.4f}, accuracy: {acc:.4f}")
                sw.add_scalar("loss/train", train_loss, steps)
                sw.add_scalar("accuracy/train", acc, steps)

            #save checkpoint
            if steps % h.checkpoint_interval == 0 and steps > 0:
                checkpoint_name = f"model_{steps:08d}"
                checkpoint_path = os.path.join(h.transformer_checkpoint_path, checkpoint_name)
                save_checkpoint(checkpoint_path, {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'steps': steps,
                    'epoch': epoch
                })

            #validation
            if steps % h.validation_interval == 0:
                model.eval()
                val_loss_total = 0.0
                val_num_correct = 0
                val_num_tokens = 0
                with torch.no_grad():
                    for j, (val_context, val_target) in enumerate(valid_loader):
                        val_context, val_target = val_context.to(device), val_target.to(device)
                        val_logits = model(val_context)
                        val_loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
                        val_loss_total += val_loss.item()

                        val_preds = val_logits.argmax(dim=-1)
                        val_mask = (val_target != 3072)
                        val_correct = (val_preds == val_target) & val_mask
                        val_num_correct += val_correct.sum().item()
                        val_num_tokens += val_mask.sum().item()

                    val_acc = val_num_correct / val_num_tokens if val_num_tokens > 0 else 0.0
                    avg_val_loss = val_loss_total / len(valid_loader)
                sw.add_scalar("loss/validation", avg_val_loss, steps)
                sw.add_scalar("accuracy/validation", val_acc, steps)

                model.train()

def main():
    print('Initializing Training Process...')

    config_file = '/mnt/nvme_share/srt30/APCodec-Reproduction/exp_do/config.json' 
    with open(config_file) as f:
        data = f.read()
    
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
    else:
        pass

    train(h)

if __name__ == '__main__':
    main()
