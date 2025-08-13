import numpy as np
import torch
from torch import roll
import os

eps = 8.854e-12
mu = np.pi * 4e-7
c = 1/np.sqrt(mu * eps)

#Boundary   PEC                         Periodic
#1          Analytic trig sum           Analytic trig rank-1
#2          Nonanalytic initial trig    Analytic trig rank-max
#3          Point trig source
#4          Analytic trig sum source
boundary = "PEC" #Periodic or PEC
iters = 4
n = 20
grid_size = 128
a = [-0.488, 0.145]
solution = 3

aanalytic = True
source = False
ending = f'{boundary}-{solution}-{iters}'
if boundary == "PEC":
    if solution == 1:
        ending += f'-{n}'
    elif solution == 2:
        analytic = False
        ending += f'-{grid_size}'
    elif solution == 3:
        analytic = False
        source = True
        ending += f'-{grid_size}-{a}'
    elif solution == 4:
        source = True
        ending += f'-{n}'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE:{device}", flush = True)
print(f'Solution: {ending}, Total iters: {iters * grid_size}', flush = True)

if boundary == "Periodic":
    cell_size = 1 / grid_size
elif boundary == "PEC":
    cell_size = 1 / (grid_size - 1)
t = 1 / grid_size / c / np.sqrt(3) #At or Below Courant limit
#Reused constants
eps1 = t / eps / cell_size
mu1 = -t / mu / cell_size

#Assume appropirate yee cell for both H and E
if boundary == "Periodic":
    x, y, z, w = torch.linspace(1 / grid_size / 2, 1 - 1 / grid_size / 2, grid_size, dtype = torch.float64, device = device), torch.linspace(1 / grid_size / 2, 1 - 1 / grid_size / 2, grid_size, dtype = torch.float64, device = device), torch.linspace(1 / grid_size / 2, 1 - 1 / grid_size / 2, grid_size, dtype = torch.float64, device = device), torch.tensor([0, 1, 2], dtype = torch.float64, device = device)
    X, Y, Z, W = torch.meshgrid(x, y, z, w, indexing = 'ij')
    E = (torch.sin(np.pi * X) * torch.sin(np.pi * Z)) #Defined at each integer timestep
    H = torch.zeros((grid_size, grid_size, grid_size, 3), dtype = torch.float64, device = device) #Defined at each half integer timestep
elif boundary == "PEC":
    if source:
        Ex = torch.zeros((grid_size - 1, grid_size, grid_size), device = device, dtype = torch.float64)
        Hy = torch.zeros((grid_size - 1, grid_size, grid_size - 1), device = device, dtype = torch.float64)
    else:
        x, y, z = torch.linspace(0, 1, grid_size * 2 - 1, dtype = torch.float64, device = device), torch.linspace(0, 1, grid_size * 2 - 1, dtype = torch.float64, device = device), torch.linspace(0, 1, grid_size * 2 - 1, dtype = torch.float64, device = device)
        X, Y, Z = torch.meshgrid(x[1::2], y[::2], z[1::2], indexing = 'ij')
        Ex = torch.zeros((grid_size - 1, grid_size, grid_size), device = device, dtype = torch.float64)
        Hy = np.sqrt(eps/mu) * 2 * torch.sin(np.pi * (X + c * t / 2)) * torch.sin(np.pi * (Y + c * t / 2)) * torch.sin(np.pi * (Z + c * t / 2))
        del X, Y, Z, x, y, z
        torch.cuda.empty_cache()
    Ey = torch.zeros((grid_size, grid_size - 1, grid_size), device = device, dtype = torch.float64)
    Ez = torch.zeros((grid_size, grid_size, grid_size - 1), device = device, dtype = torch.float64)
    Hx = torch.zeros((grid_size, grid_size - 1, grid_size - 1), device = device, dtype = torch.float64)
    Hz = torch.zeros((grid_size - 1, grid_size - 1, grid_size), device = device, dtype = torch.float64)
    Ex[:, :, 0] = 0
    Ex[:, :, -1] = 0
    Ex[:, 0, :] = 0
    Ex[:, -1, :] = 0
    Ey[:, :, 0] = 0
    Ey[:, :, -1] = 0
    Ey[0, :, :] = 0
    Ey[-1, :, :] = 0
    Ez[0, :, :] = 0
    Ez[-1, :, :] = 0
    Ez[:, 0, :] = 0
    Ez[:, -1, :] = 0

for i in range(iters * grid_size):
    if i % iters == 0:
            pass
            print(f'{round(i / iters)}/{grid_size}', flush = True)
    if boundary == "PEC":
        Hx += mu1 * (Ez[:, 1:, :] - Ez[:, :-1, :] - Ey[:, :, 1:] + Ey[:, :, :-1])
        Hy += mu1 * (Ex[:, :, 1:] - Ex[:, :, :-1] - Ez[1:, :, :] + Ez[:-1, :, :])
        Hz += mu1 * (Ey[1:, :, :] - Ey[:-1, :, :] - Ex[:, 1:, :] + Ex[:, :-1, :])
        Ex[:, 1:-1, 1:-1] += eps1 * (Hz[:, 1:, 1:-1] - Hz[:, :-1, 1:-1] - Hy[:, 1:-1, 1:] + Hy[:, 1:-1, :-1])
        Ey[1:-1, :, 1:-1] += eps1 * (Hx[1:-1, :, 1:] - Hx[1:-1, :, :-1] - Hz[1:, :, 1:-1] + Hz[:-1, :, 1:-1])
        Ez[1:-1, 1:-1, :] += eps1 * (Hy[1:, 1:-1, :] - Hy[:-1, 1:-1, :] - Hx[1:-1, 1:, :] + Hx[1:-1, :-1, :])
        if source and i < grid_size:
            T = (i + 1) * 2 * np.pi / grid_size
            Ex[grid_size // 2, grid_size // 2, grid_size // 2] += (a[0] * np.sin(T) + a[1] * 2 * np.sin(2 * T) - (a[0] + 4 * a[1]) / 3 * np.sin(3 * T)) * grid_size * grid_size

os.makedirs(ending, exist_ok = True)
for i in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']:
    np.save(os.path.join(ending, i), globals()[i].cpu().numpy())