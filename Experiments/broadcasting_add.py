import numpy as np

b = np.array([1, 2, 3])
b_broadcast = np.broadcast_to(b, (4, 3))

print(b_broadcast.strides)

