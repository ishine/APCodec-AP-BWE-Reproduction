import numpy as np
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(a)
offset = np.array([[0],[1024],[2048]])
a = a + offset
print(a)
a = a.T
a = a.flatten()
print(a)
i_offset = np.array([[0],[1024],[2048]])
a = a.reshape(-1,3)
print(a)
a = a.T
print(a)
a = a - i_offset
print(a)