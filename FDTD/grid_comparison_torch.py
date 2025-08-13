import numpy as np
import time
import matplotlib.pyplot as plt
import json
import os
import sys

eps = 8.854e-12
mu = np.pi * 4e-7
c = 1/np.sqrt(mu * eps).item()

#Boundary   PEC                         Periodic
#1          yz Analytic trig sum        Analytic trig rank-1
#2          Nonanalytic initial trig    Analytic trig rank-max
#3          Point trig source           yz Analytic trig sum
#4          Analytic trig sum source    xy Analytic trig sum
#5          xy Analytic trig sum        xyz Analytic trig sum
boundary = "PEC" #Periodic or PEC
solution = 5
error = 1e-4
n = 20 #For trig sum
a = [-0.488, 0.145] #For point trig source
iters = 4
grid_size = 4 #Starting size
end_size = 128
solver_size = 256
caps = 'd'  #Set to d for default
simulation_type = 2 #Basic Only, TT Only, Both
end_only = True #For Basic Only
npy = False
device = 'cuda'
if npy:
    from Tensor_Train.TT import TT
    from numpy import roll
    import numpy as torch
    device = 'cpu'
else:
    from Tensor_Train.TT import torchTT as TT
    import torch
    from torch import roll
    torch.backends.cuda.matmul.allow_tf32 = True
    pre2 = np.float64
    device = device if torch.cuda.is_available() else "cpu"
    print(f"DEVICE:{device}", flush = True)
precision = torch.float64
TT.roundInPlus = False
TT.oldRound = False

if caps == 'd':
    caps = [None for i in range(6)]
    if boundary == "PEC":
        if solution in [1, 4]:
            caps = [[1, n], [1, 1], [1, 1], [1, 1], [1, n], [1, n]]
        elif solution == 2:
            caps[1] = [1, 1]
        elif solution == 3:
            caps[3] = None
        elif solution == 5:
            caps = [[1, 1], [1, 1], [n, 1], [n, 1], [n, 1], [1, 1]]
    elif boundary == "Periodic":
        if solution in [1, 2]:
            caps[3] = [1, 1]
        elif solution == 3:
            caps = [[1, n] for i in range(6)]
        elif solution == 4:
            caps = [[n, 1] for i in range(6)]
        elif solution == 5:
            caps = [[n, n] for i in range(6)]
analytic = True
source = False
ending = f"{boundary}-{solution}-{iters}"
if boundary == "PEC":
    if solution in [1, 5]:
        ending += f'-{n}'
    elif solution == 2:
        analytic = False
        ending += f'-{solver_size}'
    elif solution == 3:
        analytic = False
        source = True
        ending += f'-{solver_size}-{a}'
    elif solution == 4:
        source = True
        ending += f'-{n}'
elif boundary == "Periodic":
    if solution in [3, 4, 5]:
        ending += f'-{n}'

if simulation_type == 0:
    print(f'Basic Simulation: {ending} with {"numpy" if npy else "torch"}', flush = True)
else:
    print(f'Simulation: {ending} with error {error} and caps {caps} and {"numpy" if npy else "torch"}', flush = True)

#TT.preallocate(64 * 64 * 16, precision, device)


if not analytic:
    if simulation_type > 0:
        solved_Ex = np.load(os.path.join(ending, 'Ex.npy'))
        solved_Ey = np.load(os.path.join(ending, 'Ey.npy'))
        solved_Ez = np.load(os.path.join(ending, 'Ez.npy'))
        solved_Hx = np.load(os.path.join(ending, 'Hx.npy'))
        solved_Hy = np.load(os.path.join(ending, 'Hy.npy'))
        solved_Hz = np.load(os.path.join(ending, 'Hz.npy'))
        if grid_size != solver_size or end_size != solver_size:
            from scipy.interpolate import RegularGridInterpolator
            su = np.linspace(0, 1, 2 * solver_size - 1, dtype = precision if npy else pre2)
            solved_Ex = RegularGridInterpolator((su[1::2], su[::2], su[::2]), solved_Ex)
            solved_Ey = RegularGridInterpolator((su[::2], su[1::2], su[::2]), solved_Ey)
            solved_Ez = RegularGridInterpolator((su[::2], su[::2], su[1::2]), solved_Ez)
            solved_Hx = RegularGridInterpolator((su[::2], su[1::2], su[1::2]), solved_Hx)
            solved_Hy = RegularGridInterpolator((su[1::2], su[::2], su[1::2]), solved_Hy)
            solved_Hz = RegularGridInterpolator((su[1::2], su[1::2], su[::2]), solved_Hz)
elif boundary == "PEC":
    if solution == 1:
        def solved_Ex(x, y, z, t):
            s = x * 0
            for i in range(1, n + 1):
                s += np.sin(np.pi * i * y) * np.sin(np.pi * i * z) * np.cos(np.pi * np.sqrt(2) * c * t * i)
            return s

        def solved_Ey(x, y, z, t):
            return 0 * x

        def solved_Ez(x, y, z, t):
            return 0 * x

        def solved_Hx(x, y, z, t):
            return 0 * x

        def solved_Hy(x, y, z, t):
            s = 0 * x
            for i in range(1, n + 1):
                s -= np.sin(np.pi * i * y) * np.cos(np.pi * i * z) * np.sin(np.pi * np.sqrt(2) * c * t * i)
            return np.sqrt(eps / (2 * mu)).item() * s

        def solved_Hz(x, y, z, t):
            s = 0 * x
            for i in range(1, n + 1):
                s += np.cos(np.pi * i * y) * np.sin(np.pi * i * z) * np.sin(np.pi * np.sqrt(2) * c * t * i)
            return np.sqrt(eps / (2 * mu)).item() * s
    elif solution == 4:
        def solved_Ex(x, y, z, t):
            s = x * 0
            for i in range(1, n + 1):
                s += np.sin(np.pi * i * y) * np.sin(np.pi * i * z) * np.cos(np.pi * 2 * c * t * i)
            return s
    
        def solved_Ey(x, y, z, t):
            return 0 * x
    
        def solved_Ez(x, y, z, t):
            return 0 * x
    
        def solved_Hx(x, y, z, t):
            return 0 * x
    
        def solved_Hy(x, y, z, t):
            s = 0 * x
            for i in range(1, n + 1):
                s -= np.sin(np.pi * i * y) * np.cos(np.pi * i * z) * np.sin(np.pi * 2 * c * t * i)
            return np.sqrt(eps / mu).item() * s / 2
    
        def solved_Hz(x, y, z, t):
            s = 0 * x
            for i in range(1, n + 1):
                s += np.cos(np.pi * i * y) * np.sin(np.pi * i * z) * np.sin(np.pi * 2 * c * t * i)
            return np.sqrt(eps / mu).item() * s / 2
    elif solution == 5:
        def solved_Ex(x, y, z, t):
            return 0 * z
    
        def solved_Ey(x, y, z, t):
            return 0 * z
    
        def solved_Ez(x, y, z, t):
            s = z * 0
            for i in range(1, n + 1):
                s += np.sin(np.pi * i * y) * np.sin(np.pi * i * x) * np.cos(np.pi * np.sqrt(2) * c * t * i)
            return s

        def solved_Hx(x, y, z, t):
            s = 0 * z
            for i in range(1, n + 1):
                s -= np.cos(np.pi * i * y) * np.sin(np.pi * i * x) * np.sin(np.pi * np.sqrt(2) * c * t * i)
            return np.sqrt(eps / (2 * mu)).item() * s

        def solved_Hy(x, y, z, t):
            s = 0 * z
            for i in range(1, n + 1):
                s += np.sin(np.pi * i * y) * np.cos(np.pi * i * x) * np.sin(np.pi * np.sqrt(2) * c * t * i)
            return np.sqrt(eps / (2 * mu)).item() * s
    
        def solved_Hz(x, y, z, t):
            return 0 * z
elif boundary == "Periodic":
    if solution == 1:
        def solved_Ex(x, y, z, t):
            return np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t))

        def solved_Ey(x, y, z, t):
            return -0.5 * np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t))

        def solved_Ez(x, y, z, t):
            return 0.5 * np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t))

        def solved_Hx(x, y, z, t):
            return 0 * x

        def solved_Hy(x, y, z, t):
            return -np.sqrt(3 * eps / mu) / 2  * np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t))

        def solved_Hz(x, y, z, t):
            return -np.sqrt(3 * eps / mu) / 2  * np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t))
    elif solution == 2:
        def solved_Ex(x, y, z, t):
            return np.cos(2 * np.pi * np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t)))
        
        def solved_Ey(x, y, z, t):
            return -0.5 * np.cos(2 * np.pi * np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t)))
        
        def solved_Ez(x, y, z, t):
            return 0.5 * np.cos(2 * np.pi * np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t)))
        
        def solved_Hx(x, y, z, t):
            return 0 * x
        
        def solved_Hy(x, y, z, t):
            return -np.sqrt(3 * eps / mu).item() / 2  * np.cos(2 * np.pi * np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t)))
        
        def solved_Hz(x, y, z, t):
            return -np.sqrt(3 * eps / mu).item() / 2  * np.cos(2 * np.pi * np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t)))
    elif solution == 3:
        def solved_Ex(x, y, z, t):
            s = x * 0
            for i in range(1, n + 1):
                s += np.sin(2 * np.pi * i * y) * np.sin(2 * np.pi * i * z) * np.sin(2 * np.pi * np.sqrt(2) * c * t * i)
            return s

        def solved_Ey(x, y, z, t):
            s = x * 0
            for i in range(1, n + 1):
                s -= np.cos(2 * np.pi * i * y) * np.sin(2 * np.pi * i * z) * np.cos(2 * np.pi * np.sqrt(2) * c * t * i)
            return s

        def solved_Ez(x, y, z, t):
            s = x * 0
            for i in range(1, n + 1):
                s += np.sin(2 * np.pi * i * y) * np.cos(2 * np.pi * i * z) * np.cos(2 * np.pi * np.sqrt(2) * c * t * i)
            return s

        def solved_Hx(x, y, z, t):
            s = 0 * x
            for i in range(1, n + 1):
                s -= np.cos(2 * np.pi * i * y) * np.cos(2 * np.pi * i * z) * np.sin(2 * np.pi * np.sqrt(2) * c * t * i)
            return np.sqrt(2 * eps / mu).item() * s

        def solved_Hy(x, y, z, t):
            s = 0 * x
            for i in range(1, n + 1):
                s += np.sin(2 * np.pi * i * y) * np.cos(2 * np.pi * i * z) * np.cos(2 * np.pi * np.sqrt(2) * c * t * i)
            return np.sqrt(eps / mu / 2).item() * s

        def solved_Hz(x, y, z, t):
            s = 0 * x
            for i in range(1, n + 1):
                s -= np.cos(2 * np.pi * i * y) * np.sin(2 * np.pi * i * z) * np.cos(2 * np.pi * np.sqrt(2) * c * t * i)
            return np.sqrt(eps / mu / 2).item() * s
    elif solution == 4:
        def solved_Ex(x, y, z, t):
            s = z * 0
            for i in range(1, n + 1):
                s += np.sin(2 * np.pi * i * x) * np.cos(2 * np.pi * i * y) * np.sin(2 * np.pi * np.sqrt(2) * c * t * i)
            return s

        def solved_Ey(x, y, z, t):
            s = z * 0
            for i in range(1, n + 1):
                s -= np.cos(2 * np.pi * i * x) * np.sin(2 * np.pi * i * y) * np.sin(2 * np.pi * np.sqrt(2) * c * t * i)
            return s

        def solved_Ez(x, y, z, t):
            s = z * 0
            for i in range(1, n + 1):
                s += np.sin(2 * np.pi * i * x) * np.sin(2 * np.pi * i * y) * np.cos(2 * np.pi * np.sqrt(2) * c * t * i)
            return s

        def solved_Hx(x, y, z, t):
            s = z * 0
            for i in range(1, n + 1):
                s -= np.sin(2 * np.pi * i * x) * np.cos(2 * np.pi * i * y) * np.sin(2 * np.pi * np.sqrt(2) * c * t * i)
            return np.sqrt(eps / mu / 2).item() * s

        def solved_Hy(x, y, z, t):
            s = z * 0
            for i in range(1, n + 1):
                s += np.cos(2 * np.pi * i * x) * np.sin(2 * np.pi * i * y) * np.sin(2 * np.pi * np.sqrt(2) * c * t * i)
            return np.sqrt(eps / mu / 2).item() * s

        def solved_Hz(x, y, z, t):
            s = z * 0
            for i in range(1, n + 1):
                s += np.sin(2 * np.pi * i * x) * np.sin(2 * np.pi * i * y) * np.cos(2 * np.pi * np.sqrt(2) * c * t * i)
            return np.sqrt(2 * eps / mu).item() * s
    elif solution == 5:
        def solved_Ex(x, y, z, t):
            s = x * 0
            for i in range(1, n + 1):
                s += np.cos(2 * np.pi * i * x) * np.sin(2 * np.pi * i * y) * np.sin(2 * np.pi * i * z) * np.cos(np.pi * np.sqrt(12).item() * c * i * t)
            return s

        def solved_Ey(x, y, z, t):
            s = x * 0
            for i in range(1, n + 1):
                s += np.sin(2 * np.pi * i * x) * np.cos(2 * np.pi * i * y) * np.sin(2 * np.pi * i * z) * np.cos(np.pi * np.sqrt(12).item() * c * i * t)
            return s / 2

        def solved_Ez(x, y, z, t):
            s = x * 0
            for i in range(1, n + 1):
                s -= np.sin(2 * np.pi * i * x) * np.sin(2 * np.pi * i * y) * np.cos(2 * np.pi * i * z) * np.cos(np.pi * np.sqrt(12).item() * c * i * t)
            return s / 2

        def solved_Hx(x, y, z, t):
            return 0 * x

        def solved_Hy(x, y, z, t):
            s = 0 * x
            for i in range(1, n + 1):
                s -= np.cos(2 * np.pi * i * x) * np.sin(2 * np.pi * i * y) * np.cos(2 * np.pi * i * z) * np.sin(np.pi * np.sqrt(12).item() * c * i * t)
            return np.sqrt(eps / 12 / mu) * s

        def solved_Hz(x, y, z, t):
            s = 0 * x
            for i in range(1, n + 1):
                s += np.cos(2 * np.pi * i * x) * np.cos(2 * np.pi * i * y) * np.sin(2 * np.pi * i * z) * np.sin(np.pi * np.sqrt(12).item() * c * i * t)
            return np.sqrt(eps / 12 / mu) * s

tt_times = []
basic_times = []
round_times = []
qr_times = []
svd_times = []
basic_errors = []
tt_errors = []
basic_sizes = []
tt_sizes = []
ranks = [[[] for j in range(6)] for i in range(2)]
grids = []
add_times = []
part_times = []

print("Started solving", flush = True)
while(grid_size <= end_size and (analytic or grid_size <= solver_size)):
    if boundary == "PEC":
        cell_size = 1 / (grid_size - 1)
    elif boundary == "Periodic":
        cell_size = 1 / grid_size
    t = 1 / grid_size / c / np.sqrt(3).item() #Courant limit
    #Reused constants
    eps1 = t / eps / cell_size
    mu1 = -t / mu / cell_size

    if boundary == "PEC":
        gu = np.linspace(0, 1, 2 * grid_size - 1, dtype = precision if npy else pre2)
        x = torch.linspace(0, 1, grid_size * 2 - 1, dtype = precision, device = device)
    elif boundary == "Periodic":
        gu = np.linspace(1 / grid_size / 2, 1, 2 * grid_size, dtype = precision if npy else pre2)
        x = torch.linspace(1 / grid_size / 2, 1, grid_size * 2, dtype = precision, device = device)
    X, Y, Z = torch.meshgrid(x, x, x, indexing = 'ij')
    
    #Assume appropirate yee cell for both H at each half integer timestep and E at each integer timestep
    if analytic:
        if npy:
            Exb = solved_Ex(X[1::2, ::2, ::2], Y[1::2, ::2, ::2], Z[1::2, ::2, ::2], 0)
            Eyb = solved_Ey(X[::2, 1::2, ::2], Y[::2, 1::2, ::2], Z[::2, 1::2, ::2], 0)
            Ezb = solved_Ez(X[::2, ::2, 1::2], Y[::2, ::2, 1::2], Z[::2, ::2, 1::2], 0)
            Hxb = solved_Hx(X[::2, 1::2, 1::2], Y[::2, 1::2, 1::2], Z[::2, 1::2, 1::2], -t / 2)
            Hyb = solved_Hy(X[1::2, ::2, 1::2], Y[1::2, ::2, 1::2], Z[1::2, ::2, 1::2], -t / 2)
            Hzb = solved_Hz(X[1::2, 1::2, ::2], Y[1::2, 1::2, ::2], Z[1::2, 1::2, ::2], -t / 2)
        else:
            Exb = torch.tensor(solved_Ex(X[1::2, ::2, ::2].cpu().numpy(), Y[1::2, ::2, ::2].cpu().numpy(), Z[1::2, ::2, ::2].cpu().numpy(), 0), device = device, dtype = precision)
            Eyb = torch.tensor(solved_Ey(X[::2, 1::2, ::2].cpu().numpy(), Y[::2, 1::2, ::2].cpu().numpy(), Z[::2, 1::2, ::2].cpu().numpy(), 0), device = device, dtype = precision)
            Ezb = torch.tensor(solved_Ez(X[::2, ::2, 1::2].cpu().numpy(), Y[::2, ::2, 1::2].cpu().numpy(), Z[::2, ::2, 1::2].cpu().numpy(), 0), device = device, dtype = precision)
            Hxb = torch.tensor(solved_Hx(X[::2, 1::2, 1::2].cpu().numpy(), Y[::2, 1::2, 1::2].cpu().numpy(), Z[::2, 1::2, 1::2].cpu().numpy(), -t / 2), device = device, dtype = precision)
            Hyb = torch.tensor(solved_Hy(X[1::2, ::2, 1::2].cpu().numpy(), Y[1::2, ::2, 1::2].cpu().numpy(), Z[1::2, ::2, 1::2].cpu().numpy(), -t / 2), device = device, dtype = precision)
            Hzb = torch.tensor(solved_Hz(X[1::2, 1::2, ::2].cpu().numpy(), Y[1::2, 1::2, ::2].cpu().numpy(), Z[1::2, 1::2, ::2].cpu().numpy(), -t / 2), device = device, dtype = precision)
    elif boundary == "PEC":
        if solution == 2:
            Exb = torch.zeros((grid_size - 1, grid_size, grid_size), device = device, dtype = precision)
            Hyb = np.sqrt(eps/mu) * 2 * torch.sin(np.pi * (X[1::2, ::2, 1::2] + c * t / 2)) * torch.sin(np.pi * (Y[1::2, ::2, 1::2] + c * t / 2)) * torch.sin(np.pi * (Z[1::2, ::2, 1::2] + c * t / 2))
            Hzb = torch.zeros((grid_size - 1, grid_size - 1, grid_size), device = device, dtype = precision)
        elif solution == 3:
            Exb = torch.zeros((grid_size - 1, grid_size, grid_size), device = device, dtype = precision)
            Hyb = torch.zeros((grid_size - 1, grid_size, grid_size - 1), device = device, dtype = precision)
            Hzb = torch.zeros((grid_size - 1, grid_size - 1, grid_size), device = device, dtype = precision)
        Eyb = torch.zeros((grid_size, grid_size - 1, grid_size), device = device, dtype = precision)
        Ezb = torch.zeros((grid_size, grid_size, grid_size - 1), device = device, dtype = precision)
        Hxb = torch.zeros((grid_size, grid_size - 1, grid_size - 1), device = device, dtype = precision)
    if boundary == "PEC":
        Exb[:, :, 0] = 0
        Exb[:, :, -1] = 0
        Exb[:, 0, :] = 0
        Exb[:, -1, :] = 0
        Eyb[:, :, 0] = 0
        Eyb[:, :, -1] = 0
        Eyb[0, :, :] = 0
        Eyb[-1, :, :] = 0
        Ezb[0, :, :] = 0
        Ezb[-1, :, :] = 0
        Ezb[:, 0, :] = 0
        Ezb[:, -1, :] = 0
    
    if not npy:
        del x, X, Y, Z
        torch.cuda.empty_cache()

    if simulation_type == 0:
        if not end_only:
            Ex = torch.zeros((*Exb.shape, grid_size * iters + 1))  
            Ey = torch.zeros((*Eyb.shape, grid_size * iters + 1)) 
            Ez = torch.zeros((*Ezb.shape, grid_size * iters + 1)) 
            Hx = torch.zeros((*Hxb.shape, grid_size * iters + 1))
            Hy = torch.zeros((*Hyb.shape, grid_size * iters + 1))
            Hz = torch.zeros((*Hzb.shape, grid_size * iters + 1))
            Ex[:, :, :, 0] = Exb
            Ey[:, :, :, 0] = Eyb
            Ez[:, :, :, 0] = Ezb
            Hx[:, :, :, 0] = Hxb
            Hy[:, :, :, 0] = Hyb
            Hz[:, :, :, 0] = Hzb
        if source and solution == 4:
            if npy:
                X = -np.pi * c * t * np.array([np.outer(np.sin(j * np.pi * gu[::2][1:-1]), np.sin(j * np.pi * gu[::2][1:-1])) for j in range(1, n + 1)])
            else:
                X = -np.pi * c * t * torch.tensor(np.array([np.outer(np.sin(j * np.pi * gu[::2][1:-1]), np.sin(j * np.pi * gu[::2][1:-1])) for j in range(1, n + 1)]), device = device, dtype = precision)
    else:
        Ext = TT.TTarray(Exb, error, caps[0])
        Eyt = TT.TTarray(Eyb, error, caps[1])
        Ezt = TT.TTarray(Ezb, error, caps[2])
        Hxt = TT.TTarray(Hxb, error, caps[3])
        Hyt = TT.TTarray(Hyb, error, caps[4])
        Hzt = TT.TTarray(Hzb, error, caps[5])
        if simulation_type == 1:
            if not npy:
                del Exb, Eyb, Ezb, Hxb, Hyb, Hzb
        if source:
            if solution == 3:
                P = TT.TTarray([torch.zeros((1, i, 1), device = device, dtype = precision) for i in [grid_size - 1, grid_size, grid_size]], error)
                P[0][0, grid_size // 2, 0] = 1
                P[2][0, grid_size // 2, 0] = 1
            elif solution == 4:
                P = TT.TTarray([torch.zeros(([1, 1, n][i], [grid_size - 1, grid_size, grid_size][i], [1, n, 1][i]), device = device, dtype = precision) for i in range(3)], error)
                P[0][0, :, 0] = -np.pi * c * t
                if npy:
                    X = -np.pi * c * t * np.array([np.outer(np.sin(j * np.pi * gu[::2][1:-1]), np.sin(j * np.pi * gu[::2][1:-1])) for j in range(1, n + 1)])
                else:
                    X = -np.pi * c * t * torch.tensor(np.array([np.outer(np.sin(j * np.pi * gu[::2][1:-1]), np.sin(j * np.pi * gu[::2][1:-1])) for j in range(1, n + 1)]), device = device, dtype = precision)
        lranks = [i.ranks()[1:3] for i in [Ext, Eyt, Ezt, Hxt, Hyt, Hzt]]
        tt_size = Hxt.nbytes() + Hyt.nbytes() + Hzt.nbytes() + Ext.nbytes() + Eyt.nbytes() + Ezt.nbytes()

    TT.roundTime = 0
    TT.qrTime = 0
    TT.svdTime = 0
    TT.addTime = 0
    basic_time = 0
    tt_time = 0
    part_time = 0
    if device == 'cuda':
        events = [torch.cuda.Event(enable_timing = True) for _ in range(4)] #Basic, TT
    for i in range(grid_size * iters):
        if i % iters == 0:
            pass
            print(f'{round(i / iters)}/{grid_size}', flush = True)
        if simulation_type in [0, 2]:
            if device == 'cuda':
                events[0].record()
            else:           
                basic_time -= time.time()
            if boundary == "Periodic":
                Hxb += mu1 * (roll(Ezb, -1, 1) - Ezb - roll(Eyb, -1, 2) + Eyb)
                Hyb += mu1 * (roll(Exb, -1, 2) - Exb - roll(Ezb, -1, 0) + Ezb)
                Hzb += mu1 * (roll(Eyb, -1, 0) - Eyb - roll(Exb, -1, 1) + Exb)
                Exb += eps1 * (Hzb - roll(Hzb, 1, 1) - Hyb + roll(Hyb, 1, 2))
                Eyb += eps1 * (Hxb - roll(Hxb, 1, 2) - Hzb + roll(Hzb, 1, 0))
                Ezb += eps1 * (Hyb - roll(Hyb, 1, 0) - Hxb + roll(Hxb, 1, 1))
            elif boundary == "PEC":
                Hxb += mu1 * (Ezb[:, 1:, :] - Ezb[:, :-1, :] - Eyb[:, :, 1:] + Eyb[:, :, :-1])
                Hyb += mu1 * (Exb[:, :, 1:] - Exb[:, :, :-1] - Ezb[1:, :, :] + Ezb[:-1, :, :])
                Hzb += mu1 * (Eyb[1:, :, :] - Eyb[:-1, :, :] - Exb[:, 1:, :] + Exb[:, :-1, :])
                Exb[:, 1:-1, 1:-1] += eps1 * (Hzb[:, 1:, 1:-1] - Hzb[:, :-1, 1:-1] - Hyb[:, 1:-1, 1:] + Hyb[:, 1:-1, :-1])
                Eyb[1:-1, :, 1:-1] += eps1 * (Hxb[1:-1, :, 1:] - Hxb[1:-1, :, :-1] - Hzb[1:, :, 1:-1] + Hzb[:-1, :, 1:-1])
                Ezb[1:-1, 1:-1, :] += eps1 * (Hyb[1:, 1:-1, :] - Hyb[:-1, 1:-1, :] - Hxb[1:-1, 1:, :] + Hxb[1:-1, :-1, :])
                if source:
                    if solution == 3:
                        if i < grid_size:
                            T = (i + 1) * 2 * np.pi / grid_size
                            x = (a[0] * np.sin(T) + a[1] * 2 * np.sin(2 * T) - (a[0] + 4 * a[1]) / 3 * np.sin(3 * T)) * grid_size * grid_size
                            Exb[grid_size // 2, grid_size // 2, grid_size // 2] += x
                    elif solution == 4:
                        part_time -= time.time()
                        x = sum([j * np.sin(2 * j * np.pi * c * t * (i + 0.5)) * X[j - 1, :, :] for j in range(1, n + 1)])
                        part_time += time.time()
                        Exb[:, 1:-1, 1:-1] += x
            if device == 'cuda':
                events[1].record()
            else:           
                basic_time += time.time()
            if simulation_type == 0:
                if not end_only:
                    Ex[:, :, :, i + 1] = Exb
                    Ey[:, :, :, i + 1] = Eyb
                    Ez[:, :, :, i + 1] = Ezb
                    Hx[:, :, :, i + 1] = Hxb
                    Hy[:, :, :, i + 1] = Hyb
                    Hz[:, :, :, i + 1] = Hzb
        if simulation_type > 0:
            if device == 'cuda':
                events[2].record()
            else:           
                tt_time -= time.time()
            if boundary == "Periodic":
                Hxt += mu1 * (Eyt.rollsum([[-1, 2, -1]]) - Ezt.rollsum([[-1, 1, -1]]))
                Hyt += mu1 * (Ezt.rollsum([[-1, 0, -1]]) - Ext.rollsum([[-1, 2, -1]]))
                Hzt += mu1 * (Ext.rollsum([[-1, 1, -1]]) - Eyt.rollsum([[-1, 0, -1]]))
                if not TT.roundInPlus:
                    Hxt = Hxt.round()
                    Hyt = Hyt.round()
                    Hzt = Hzt.round()
                Ext += eps1 * (Hzt.rollsum([[1, 1, -1]]) - Hyt.rollsum([[1, 2, -1]]))
                Eyt += eps1 * (Hxt.rollsum([[1, 2, -1]]) - Hzt.rollsum([[1, 0, -1]]))
                Ezt += eps1 * (Hyt.rollsum([[1, 0, -1]]) - Hxt.rollsum([[1, 1, -1]]))
                if not TT.roundInPlus:
                    Ext = Ext.round()
                    Eyt = Eyt.round()
                    Ezt = Ezt.round()
            elif boundary == "PEC":
                Hxt += mu1 * (Ezt.reducedSum(1, [(1, None), (0, -1)], [1, -1]) - Eyt.reducedSum(2, [(1, None), (0, -1)], [1, -1]))
                Hyt += mu1 * (Ext.reducedSum(2, [(1, None), (0, -1)], [1, -1]) - Ezt.reducedSum(0, [(1, None), (0, -1)], [1, -1]))
                Hzt += mu1 * (Eyt.reducedSum(0, [(1, None), (0, -1)], [1, -1]) - Ext.reducedSum(1, [(1, None), (0, -1)], [1, -1]))
                if not TT.roundInPlus:
                    Hxt = Hxt.round()
                    Hyt = Hyt.round()
                    Hzt = Hzt.round()
                Ext += eps1 * (Hzt.reduce([':', ':', (1, -1)]).reducedSum(1, [(1, None), (0, -1)], [1, -1]) - Hyt.reduce([':', (1, -1), ':']).reducedSum(2, [(1, None), (0, -1)], [1, -1])).pad([1, 2], [[2, 1], [2, 1]])
                Eyt += eps1 * (Hxt.reduce([(1, -1), ':', ':']).reducedSum(2, [(1, None), (0, -1)], [1, -1]) - Hzt.reduce([':', ':', (1, -1)]).reducedSum(0, [(1, None), (0, -1)], [1, -1])).pad([0, 2], [[2, 1], [2, 1]])
                Ezt += eps1 * (Hyt.reduce([':', (1, -1), ':']).reducedSum(0, [(1, None), (0, -1)], [1, -1]) - Hxt.reduce([(1, -1), ':', ':']).reducedSum(1, [(1, None), (0, -1)], [1, -1])).pad([0, 1], [[2, 1], [2, 1]])
                if not TT.roundInPlus:
                    Ext = Ext.round()
                    Eyt = Eyt.round()
                    Ezt = Ezt.round()
                if source:
                    if solution == 3:
                        if i < grid_size:
                            T = (i + 1) * 2 * np.pi / grid_size
                            x = (a[0] * np.sin(T) + a[1] * 2 * np.sin(2 * T) - (a[0] + 4 * a[1]) / 3 * np.sin(3 * T)) * grid_size * grid_size
                            P[1][0, grid_size // 2, 0] = x
                            Ext += P
                    elif solution == 4:
                        if npy:
                            P[1][0, 1:-1, :] = np.array([np.sin(2 * j * np.pi * c * t * (i + 0.5)) * np.sin(j * np.pi * gu[::2][1:-1]) for j in range(1, n + 1)]).T
                            P[2][:, 1:-1, 0] = np.array([j * np.sin(j * np.pi * gu[::2][1:-1]) for j in range(1, n + 1)])
                        else:
                            P[1][0, 1:-1, :] = torch.tensor(np.array([np.sin(2 * j * np.pi * c * t * (i + 0.5)) * np.sin(j * np.pi * gu[::2][1:-1]) for j in range(1, n + 1)]).T, device = device, dtype = precision)
                            P[2][:, 1:-1, 0] = torch.tensor(np.array([j * np.sin(j * np.pi * gu[::2][1:-1]) for j in range(1, n + 1)]), device = device, dtype = precision)
                        Ext += P
            if device == 'cuda':
                events[3].record()
            else:           
                tt_time += time.time()
            for j in range(6):
                eranks = [Ext, Eyt, Ezt, Hxt, Hyt, Hzt][j].ranks()
                lranks[j][0] = max(lranks[j][0], eranks[1])
                lranks[j][1] = max(lranks[j][1], eranks[2])
                tt_size = max(Hxt.nbytes() + Hyt.nbytes() + Hzt.nbytes() + Ext.nbytes() + Eyt.nbytes() + Ezt.nbytes(), tt_size)
        if device == 'cuda':
            torch.cuda.synchronize()
            if simulation_type != 1:
                basic_time += events[0].elapsed_time(events[1]) / 1000
            if simulation_type != 0:
                tt_time += events[2].elapsed_time(events[3]) / 1000
                for j in range(0, len(TT.events['Rounding']), 2):
                    TT.roundTime += TT.events['Rounding'][j].elapsed_time(TT.events['Rounding'][j + 1]) / 1000
                TT.events['Rounding'] = []
                for j in range(0, len(TT.events['qr']), 2):
                    TT.qrTime += TT.events['qr'][j].elapsed_time(TT.events['qr'][j + 1]) / 1000
                TT.events['qr'] = []
                for j in range(0, len(TT.events['svd']), 2):
                    TT.svdTime += TT.events['svd'][j].elapsed_time(TT.events['svd'][j + 1]) / 1000
                TT.events['svd'] = []
                for j in range(0, len(TT.events['Part']), 2):
                    TT.part_time += TT.events['Part'][j].elapsed_time(TT.events['Part'][j + 1]) / 1000
                TT.events['Part'] = []
                for j in range(0, len(TT.events['Addition']), 2):
                    TT.addTime += TT.events['Addition'][j].elapsed_time(TT.events['Addition'][j + 1]) / 1000
                TT.events['Addition'] = []
    if simulation_type == 0:
        print(basic_time, flush = True)
        if end_only:
            Ex = Exb
            Ey = Eyb
            Ez = Ezb
            Hx = Hxb
            Hy = Hyb
            Hz = Hzb
        if npy:
            np.save(f'Ex-{ending}-{grid_size}.npy', Ex)
            np.save(f'Ey-{ending}-{grid_size}.npy', Ey)
            np.save(f'Ez-{ending}-{grid_size}.npy', Ez)
            np.save(f'Hx-{ending}-{grid_size}.npy', Hx)
            np.save(f'Hy-{ending}-{grid_size}.npy', Hy)
            np.save(f'Hz-{ending}-{grid_size}.npy', Hz)
        else:
            np.save(f'Ex-{ending}-{grid_size}.npy', Ex.cpu().numpy())
            np.save(f'Ey-{ending}-{grid_size}.npy', Ey.cpu().numpy())
            np.save(f'Ez-{ending}-{grid_size}.npy', Ez.cpu().numpy())
            np.save(f'Hx-{ending}-{grid_size}.npy', Hx.cpu().numpy())
            np.save(f'Hy-{ending}-{grid_size}.npy', Hy.cpu().numpy())
            np.save(f'Hz-{ending}-{grid_size}.npy', Hz.cpu().numpy())
            del Exb, Eyb, Ezb, Hxb, Hyb, Hzb, Ex, Ey, Ez, Hx, Hy, Hz
            if source and solution == 4:
                del X
    else:
        if not analytic:
            if grid_size == solver_size and end_size == solver_size and len(grids) == 0:
                Exs = solved_Ex
                Eys = solved_Ey
                Ezs = solved_Ez
                Hxs = solved_Hx
                Hys = solved_Hy
                Hzs = solved_Hz
            else:
                Exs = solved_Ex(np.meshgrid(gu[1::2], gu[::2], gu[::2], indexing = 'ij'))
                Eys = solved_Ey(np.meshgrid(gu[::2], gu[1::2], gu[::2], indexing = 'ij'))
                Ezs = solved_Ez(np.meshgrid(gu[::2], gu[::2], gu[1::2], indexing = 'ij'))
                Hxs = solved_Hx(np.meshgrid(gu[::2], gu[1::2], gu[1::2], indexing = 'ij'))
                Hys = solved_Hy(np.meshgrid(gu[1::2], gu[::2], gu[1::2], indexing = 'ij'))
                Hzs = solved_Hz(np.meshgrid(gu[1::2], gu[1::2], gu[::2], indexing = 'ij'))
        else:
            Exs = solved_Ex(*np.meshgrid(gu[1::2], gu[::2], gu[::2], indexing = 'ij'), iters * grid_size * t)
            Eys = solved_Ey(*np.meshgrid(gu[::2], gu[1::2], gu[::2], indexing = 'ij'), iters * grid_size * t)
            Ezs = solved_Ez(*np.meshgrid(gu[::2], gu[::2], gu[1::2], indexing = 'ij'), iters * grid_size * t)
            Hxs = solved_Hx(*np.meshgrid(gu[::2], gu[1::2], gu[1::2], indexing = 'ij'), (iters * grid_size - 0.5) * t)
            Hys = solved_Hy(*np.meshgrid(gu[1::2], gu[::2], gu[1::2], indexing = 'ij'), (iters * grid_size - 0.5) * t)
            Hzs = solved_Hz(*np.meshgrid(gu[1::2], gu[1::2], gu[::2], indexing = 'ij'), (iters * grid_size - 0.5) * t)

        if npy:
            if simulation_type == 2:
                basic_errors.append(
                    np.sqrt(
                            (
                                (
                                    np.sum((Exb - Exs).astype(np.float64) ** 2) +
                                    np.sum((Eyb - Eys).astype(np.float64) ** 2) +
                                    np.sum((Ezb - Ezs).astype(np.float64) ** 2)
                                ) / (np.sum(Exs ** 2) + np.sum(Eys ** 2) + np.sum(Ezs ** 2)) + 
                                (
                                    np.sum((Hxb - Hxs).astype(np.float64) ** 2) +
                                    np.sum((Hyb - Hys).astype(np.float64) ** 2) +
                                    np.sum((Hzb - Hzs).astype(np.float64) ** 2)
                                ) / (np.sum(Hxs ** 2) + np.sum(Hys ** 2) + np.sum(Hzs ** 2))
                            )
                    ).item()
                )
            tt_errors.append(
                np.sqrt(
                        (
                            (
                                np.sum((Ext.raw(np.float64) - Exs) ** 2) +
                                np.sum((Eyt.raw(np.float64) - Eys) ** 2) +
                                np.sum((Ezt.raw(np.float64) - Ezs) ** 2)
                            ) / (np.sum(Exs ** 2) + np.sum(Eys ** 2) + np.sum(Ezs ** 2)) + 
                            (
                                np.sum((Hxt.raw(np.float64) - Hxs) ** 2) +
                                np.sum((Hyt.raw(np.float64) - Hys) ** 2) +
                                np.sum((Hzt.raw(np.float64) - Hzs) ** 2)
                            ) / (np.sum(Hxs ** 2) + np.sum(Hys ** 2) + np.sum(Hzs ** 2))
                        )
                    ).item()
                )
        else:
            if simulation_type == 2:
                basic_errors.append(
                    np.sqrt(
                            (
                                (
                                    np.sum((Exb.cpu().numpy() - Exs) ** 2) +
                                    np.sum((Eyb.cpu().numpy() - Eys) ** 2) +
                                    np.sum((Ezb.cpu().numpy() - Ezs) ** 2)
                                ) / (np.sum(Exs ** 2) + np.sum(Eys ** 2) + np.sum(Ezs ** 2)) + 
                                (
                                    np.sum((Hxb.cpu().numpy() - Hxs) ** 2) +
                                    np.sum((Hyb.cpu().numpy() - Hys) ** 2) +
                                    np.sum((Hzb.cpu().numpy() - Hzs) ** 2)
                                ) / (np.sum(Hxs ** 2) + np.sum(Hys ** 2) + np.sum(Hzs ** 2))
                            )
                    )
                )
            tt_errors.append(
                np.sqrt(
                        (
                            (
                                np.sum((Ext.raw().cpu().numpy() - Exs) ** 2) +
                                np.sum((Eyt.raw().cpu().numpy() - Eys) ** 2) +
                                np.sum((Ezt.raw().cpu().numpy() - Ezs) ** 2)
                            ) / (np.sum(Exs ** 2) + np.sum(Eys ** 2) + np.sum(Ezs ** 2)) + 
                            (
                                np.sum((Hxt.raw().cpu().numpy() - Hxs) ** 2) +
                                np.sum((Hyt.raw().cpu().numpy() - Hys) ** 2) +
                                np.sum((Hzt.raw().cpu().numpy() - Hzs) ** 2)
                            ) / (np.sum(Hxs ** 2) + np.sum(Hys ** 2) + np.sum(Hzs ** 2))
                        )
                )
            )
        tt_times.append(tt_time)
        round_times.append(TT.roundTime)
        qr_times.append(TT.qrTime)
        svd_times.append(TT.svdTime)
        if simulation_type == 2:
            basic_times.append(basic_time)
            basic_sizes.append(Exb.nbytes + Eyb.nbytes + Ezb.nbytes + Hxb.nbytes + Hyb.nbytes + Hzb.nbytes)
        if part_time != 0:
            part_times.append(part_time)
        elif TT.part_time != 0:
            part_times.append(TT.part_time)
        
        if source:
            tt_sizes.append(tt_size + P.nbytes())
        else:
            tt_sizes.append(tt_size)
        add_times.append(TT.addTime)
        for i in range(6):
            ranks[0][i].append(lranks[i][0])
            ranks[1][i].append(lranks[i][1])
        grids.append(grid_size)
        json_stuff = ["grids", "basic_times", "tt_times", "round_times", "qr_times", "svd_times", "add_times", "basic_errors", "tt_errors", "basic_sizes", "tt_sizes", "ranks"]
        if simulation_type == 1:
            json_stuff = [s for s in json_stuff if 'basic' not in s]

        if len(part_times) != 0:
            json_stuff.insert(7, "part_times")
        with open(f"{ending}.json", "w") as f:
            json.dump({i: eval(i) for i in json_stuff}, f)

        if len(grids) > 1:
            plt.clf()
            plt.title(f"Time to solve")
            plt.xlabel("grid size")
            plt.ylabel("time (seconds)") 
            plt.xscale("log") 
            plt.yscale("log")
            if simulation_type == 2:
                plt.plot(grids, basic_times, label = "Basic")
            plt.plot(grids, tt_times, label = "TT")
            plt.plot(grids, round_times, label = "Rounding", linestyle = '--')
            plt.plot(grids, qr_times, label = "QR", linestyle = '--')
            plt.plot(grids, svd_times, label = "SVD", linestyle = '--')
            plt.plot(grids, np.array(svd_times) + np.array(qr_times), label = "QR+SVD", linestyle = '--')
            plt.plot(grids, add_times, label = "Add", linestyle = '--')
            if len(part_times) != 0:
                plt.plot(grids, part_times, label = "Part")
            plt.legend()
            plt.savefig(f"plot1-{ending}.png")

            plt.clf()
            plt.title(f"Final error")
            plt.xlabel("grid size")
            plt.ylabel("relative error") 
            plt.xscale("log") 
            plt.yscale("log")
            if simulation_type == 2:
                plt.plot(grids, basic_errors, label = "Basic")
            plt.plot(grids, tt_errors, label  = "TT")
            plt.legend()
            plt.savefig(f"plot2-{ending}.png", bbox_inches='tight')

            plt.clf()
            plt.title(f"Storage needed")
            plt.xlabel("grid size")
            plt.ylabel("bytes")
            plt.xscale("log") 
            plt.yscale("log")
            if simulation_type == 2:
                plt.plot(grids, basic_sizes, label = "Basic")
            plt.plot(grids, tt_sizes, label = "TT")
            plt.legend()
            plt.savefig(f"plot3-{ending}.png")

            plt.clf()
            plt.title(f"Rank 1")
            plt.xlabel("grid size")
            plt.ylabel("size")
            plt.xscale("log") 
            plt.yscale("log")
            for i in range(6):
                plt.plot(grids, ranks[0][i], label = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"][i], linestyle = '--' if i > 2 else '-')
            plt.legend()
            plt.savefig(f"plot4-{ending}.png")

            plt.clf()
            plt.title(f"Rank 2")
            plt.xlabel("grid size")
            plt.ylabel("size")
            plt.xscale("log") 
            plt.yscale("log")
            for i in range(6):
                plt.plot(grids, ranks[1][i], label = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"][i], linestyle = '--' if i > 2 else '-')
            plt.legend()
            plt.savefig(f"plot5-{ending}.png")
        if not npy:
            if simulation_type == 2:
                del Exb, Eyb, Ezb, Hxb, Hyb, Hzb
            del Ext, Eyt, Ezt, Hxt, Hyt, Hzt, Exs, Eys, Ezs, Hxs, Hys, Hzs
            if source:
                del P
                if solution == 4:
                    del X
    if not npy:
        torch.cuda.empty_cache()
    print(f"{grid_size}: Finished solving", flush = True)
    grid_size *= 2