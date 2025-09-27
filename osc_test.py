import numpy as np

x = np.linspace(0, np.pi, 1024)
A = []
for i in range(6):
    A.append(np.zeros(1024, 1024, 1024), dtype = np.float64)
for i in range(10):
    for j in range(6):
        A[j] += np.sin(i * x[:, None, None]) * np.sin(i * x[None, :, None]) * np.sin(i * x[None, None, :])