import numpy as np
import time
import matplotlib.pyplot as plt
import json
import os
from typing import Literal, Tuple, Union


#IMPORTANT: While the code itself may not be written as optimally as possible, the algorithms are
def multi_test(boundary: Literal["PEC", "Periodic"], solution: int, iters: int, sizes: Tuple[int], npy: bool, simulations: list[dict], param = None, eps: float = 8.854e-12, mu: float = np.pi * 4e-7, device: Literal['cpu', 'cuda'] = 'cuda', solver: bool = False, ignore_error: bool = False):
    #Initial setup
    c = 1/np.sqrt(mu * eps).item()
    def omega_num(fx, fy, fz, dx, dt):
        S = c * dt / dx
        s2 = np.sum([np.sin(np.array([fx, fy, fz]) * dx / 2) ** 2]).item()
        s = np.clip(S * np.sqrt(s2), 0, 1)
        return 2 / dt * np.arcsin(s)
    if len(sizes) > 2:
        solver_size = sizes[2]
        end_size = sizes[1]
    else:
        if len(sizes) == 2:
            end_size = sizes[1]
        else:
            end_size = sizes[0]
        solver_size = end_size
    grid_size = solver_size if solver else sizes[0] 
    n = param
    a = param
    p = param
    order = 'Ex,Ey,Ez,Hx,Hy,Hz'.split(',') #Ordering for operations done on everything
    yee = { #Yee grid and initial times
        'Ex': (1, 0, 0, 0),
        'Ey': (0, 1, 0, 0),
        'Ez': (0, 0, 1, 0),
        'Hx': (0, 1, 1, -1/2),
        'Hy': (1, 0, 1, -1/2),
        'Hz': (1, 1, 0, -1/2)
    }
    EHk = ['solving', 'solver', 'solved']
    EH = {k: {} for k in EHk}
    ixer = {k: tuple(slice(yee[k][i], None, 2) for i in range(3)) for k in order} #indexing when using half-spaced grid
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
    #Figuring out simulation from boundary and solution number
    analytic = True
    ending = f"{boundary}-{solution}-{iters}"
    if boundary == "PEC":
        if solution in [1, 5]:
            ending += f'-{n}'
        elif solution == 2:
            analytic = False
            ending += f'-{solver_size}-{n}'
        elif solution == 3:
            analytic = False
            ending += f'-{solver_size}-{a}'
        elif solution == 4:
            ending += f'-{n}'
    elif boundary == "Periodic":
        if solution in [3, 4, 5]:
            ending += f'-{n}'
        elif solution == 6:
            analytic = True
            ending += f'-{p}'
    if not analytic and not os.path.isdir(ending) and not solver and not ignore_error:
        multi_test(boundary, solution, iters, sizes, npy, simulations, param, eps, mu, device, True)
    if not solver:
        print(f'Simulation for: {ending} with {"numpy" if npy else "torch"}', flush = True)
        if not analytic: #Approximate non-analytic solutions using pre-solved solution using interpolation
            if not ignore_error:
                if boundary == 'PEC':
                    su = np.linspace(0, 1, 2 * solver_size + 1, dtype = precision if npy else pre2)
                elif boundary == 'Periodic':
                    su = np.linspace(1 / solver_size / 2, 1, 2 * solver_size, dtype = precision if npy else pre2)
                from scipy.interpolate import RegularGridInterpolator
                for d in order:
                    EH['solver'][d] = RegularGridInterpolator(tuple(su[ixer[d][i]] for i in range(3)), np.load(os.path.join(ending, f'{d}.npy')), method = "linear")
        else: #Otherwise, setup analytic solution
            if boundary == "PEC": 
                if solution == 1:
                    def solved(x, y, z, t, d, g, dt):
                        s = torch.zeros_like(x * y * z)
                        match d:
                            case 'Ex':
                                for i in range(1, n + 1):
                                    s += torch.cos(np.pi * i * x) * torch.sin(np.pi * i * y) * torch.sin(np.pi * i * z) * np.cos(omega_num(np.pi * i, np.pi * i, np.pi * i, 1 / g,  dt) * t)
                                return s
                            case 'Ey':
                                for i in range(1, n + 1):
                                    s -= torch.sin(np.pi * i * x) * torch.cos(np.pi * i * y) * torch.sin(np.pi * i * z) * np.cos(omega_num(np.pi * i, np.pi * i, np.pi * i, 1 / g,  dt) * t)
                                return s / 2
                            case 'Ez':
                                for i in range(1, n + 1):
                                    s -= torch.sin(np.pi * i * x) * torch.sin(np.pi * i * y) * torch.cos(np.pi * i * z) * np.cos(omega_num(np.pi * i, np.pi * i, np.pi * i, 1 / g,  dt) * t)
                                return s / 2
                            case 'Hx':
                                return s
                            case 'Hy':
                                for i in range(1, n + 1):
                                    s -= torch.cos(np.pi * i * x) * torch.sin(np.pi * i * y) * torch.cos(np.pi * i * z) * np.sin(omega_num(np.pi * i, np.pi * i, np.pi * i, 1 / g,  dt) * t)
                                return np.sqrt(3 * eps / (4 * mu)).item() * s
                            case 'Hz':
                                for i in range(1, n + 1):
                                    s += torch.cos(np.pi * i * x) * torch.cos(np.pi * i * y) * torch.sin(np.pi * i * z) * np.sin(omega_num(np.pi * i, np.pi * i, np.pi * i, 1 / g,  dt) * t)
                                return np.sqrt(3 * eps / (4 * mu)).item() * s
                        return s
                elif solution == 4:
                    def solved(x, y, z, t, d, g, dt):
                        s = np.zeros_like(x)
                        match d:
                            case 'Ex':
                                for i in range(1, n + 1):
                                    s += np.sin(np.pi * i * y) * np.sin(np.pi * i * z) * np.cos(np.pi * 2 * c * t * i)
                            case 'Hy':
                                for i in range(1, n + 1):
                                    s -= np.sin(np.pi * i * y) * np.cos(np.pi * i * z) * np.sin(np.pi * 2 * c * t * i)
                                return np.sqrt(eps / mu).item() * s / 2
                            case 'Hz':
                                for i in range(1, n + 1):
                                    s += np.cos(np.pi * i * y) * np.sin(np.pi * i * z) * np.sin(np.pi * 2 * c * t * i)
                                return np.sqrt(eps / mu).item() * s / 2
                        return s
                elif solution == 5:
                    def solved(x, y, z, t, d, g, dt):
                        s = np.zeros_like(z)
                        match d:
                            case 'Ez':
                                for i in range(1, n + 1):
                                    s += np.sin(np.pi * i * y) * np.sin(np.pi * i * x) * np.cos(np.pi * np.sqrt(2) * c * t * i)
                            case 'Hx':
                                for i in range(1, n + 1):
                                    s -= np.cos(np.pi * i * y) * np.sin(np.pi * i * x) * np.sin(np.pi * np.sqrt(2) * c * t * i)
                                return np.sqrt(eps / (2 * mu)).item() * s
                            case 'Hy':
                                for i in range(1, n + 1):
                                    s += np.sin(np.pi * i * y) * np.cos(np.pi * i * x) * np.sin(np.pi * np.sqrt(2) * c * t * i)
                                return np.sqrt(eps / (2 * mu)).item() * s
                        return s
                elif solution == 6:
                    def solved(x, y, z, t, d, g, dt):
                        s = np.zeros_like(x)
                        match d:
                            case 'Ex':
                                pass
                            case 'Ey':
                                pass
                            case 'Ez':
                                pass
                            case 'Hx':
                                pass
                            case 'Hy':
                                pass
                            case 'Hz':
                                pass
                        return s
            elif boundary == "Periodic":
                if solution == 1:
                    def solved(x, y, z, t, d, g, dt):
                        match d:
                            case 'Ex':
                                return np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t))
                            case 'Ey':
                                return -0.5 * np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t))
                            case 'Ez':
                                return 0.5 * np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t))
                            case 'Hx':
                                return np.zeros_like(x)
                            case 'Hy':
                                return -np.sqrt(3 * eps / mu) / 2  * np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t))
                            case 'Hz':
                                return -np.sqrt(3 * eps / mu) / 2  * np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t))
                elif solution == 2:
                    def solved(x, y, z, t, d, g, dt):
                        match d:
                            case 'Ex':
                                return np.cos(2 * np.pi * np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t)))
                            case 'Ey':
                                return -0.5 * np.cos(2 * np.pi * np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t)))
                            case 'Ez':
                                return 0.5 * np.cos(2 * np.pi * np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t)))
                            case 'Hx':
                                return np.zeros_like(x)
                            case 'Hy':
                                return -np.sqrt(3 * eps / mu).item() / 2  * np.cos(2 * np.pi * np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t)))
                            case 'Hz':
                                return -np.sqrt(3 * eps / mu).item() / 2  * np.cos(2 * np.pi * np.cos(2 * np.pi * (x + y - z - np.sqrt(3) * c * t)))
                elif solution == 3:
                    def solved(x, y, z, t, d, g, dt):
                        s = np.zeros_like(x)
                        match d:
                            case 'Ex':
                                for i in range(1, n + 1):
                                    s += np.sin(2 * np.pi * i * y) * np.sin(2 * np.pi * i * z) * np.sin(2 * np.pi * np.sqrt(2) * c * t * i)
                            case 'Ey':
                                for i in range(1, n + 1):
                                    s -= np.cos(2 * np.pi * i * y) * np.sin(2 * np.pi * i * z) * np.cos(2 * np.pi * np.sqrt(2) * c * t * i)
                            case 'Ez':
                                for i in range(1, n + 1):
                                    s += np.sin(2 * np.pi * i * y) * np.cos(2 * np.pi * i * z) * np.cos(2 * np.pi * np.sqrt(2) * c * t * i)
                            case 'Hx':
                                for i in range(1, n + 1):
                                    s -= np.cos(2 * np.pi * i * y) * np.cos(2 * np.pi * i * z) * np.sin(2 * np.pi * np.sqrt(2) * c * t * i)
                                return np.sqrt(2 * eps / mu).item() * s
                            case 'Hy':
                                for i in range(1, n + 1):
                                    s += np.sin(2 * np.pi * i * y) * np.cos(2 * np.pi * i * z) * np.cos(2 * np.pi * np.sqrt(2) * c * t * i)
                                return np.sqrt(eps / mu / 2).item() * s
                            case 'Hz':
                                for i in range(1, n + 1):
                                    s -= np.cos(2 * np.pi * i * y) * np.sin(2 * np.pi * i * z) * np.cos(2 * np.pi * np.sqrt(2) * c * t * i)
                                return np.sqrt(eps / mu / 2).item() * s
                        return s
                elif solution == 4:
                    def solved(x, y, z, t, d, g, dt):
                        s = np.zeros_like(z)
                        match d:
                            case 'Ex':
                                for i in range(1, n + 1):
                                    s += np.sin(2 * np.pi * i * x) * np.cos(2 * np.pi * i * y) * np.sin(2 * np.pi * np.sqrt(2) * c * t * i)
                            case 'Ey':
                                for i in range(1, n + 1):
                                    s -= np.cos(2 * np.pi * i * x) * np.sin(2 * np.pi * i * y) * np.sin(2 * np.pi * np.sqrt(2) * c * t * i)
                            case 'Ez':
                                for i in range(1, n + 1):
                                    s += np.sin(2 * np.pi * i * x) * np.sin(2 * np.pi * i * y) * np.cos(2 * np.pi * np.sqrt(2) * c * t * i)
                            case 'Hx':
                                for i in range(1, n + 1):
                                    s -= np.sin(2 * np.pi * i * x) * np.cos(2 * np.pi * i * y) * np.sin(2 * np.pi * np.sqrt(2) * c * t * i)
                                return np.sqrt(eps / mu / 2).item() * s
                            case 'Hy':
                                for i in range(1, n + 1):
                                    s += np.cos(2 * np.pi * i * x) * np.sin(2 * np.pi * i * y) * np.sin(2 * np.pi * np.sqrt(2) * c * t * i)
                                return np.sqrt(eps / mu / 2).item() * s
                            case 'Hz':
                                for i in range(1, n + 1):
                                    s += np.sin(2 * np.pi * i * x) * np.sin(2 * np.pi * i * y) * np.cos(2 * np.pi * np.sqrt(2) * c * t * i)
                                return np.sqrt(2 * eps / mu).item() * s
                        return s
                elif solution == 5:
                    def solved(x, y, z, t, d, g, dt):
                        s = np.zeros_like(z)
                        match d:
                            case 'Ex':
                                for i in range(1, n + 1):
                                    s += np.cos(2 * np.pi * i * x) * np.sin(2 * np.pi * i * y) * np.sin(2 * np.pi * i * z) * np.cos(np.pi * np.sqrt(12).item() * c * i * t)
                                return s
                            case 'Ey':
                                for i in range(1, n + 1):
                                    s -= np.sin(2 * np.pi * i * x) * np.cos(2 * np.pi * i * y) * np.sin(2 * np.pi * i * z) * np.cos(np.pi * np.sqrt(12).item() * c * i * t)
                                return s / 2
                            case 'Ez':
                                for i in range(1, n + 1):
                                    s -= np.sin(2 * np.pi * i * x) * np.sin(2 * np.pi * i * y) * np.cos(2 * np.pi * i * z) * np.cos(np.pi * np.sqrt(12).item() * c * i * t)
                                return s / 2
                            case 'Hx':
                                return s
                            case 'Hy':
                                for i in range(1, n + 1):
                                    s -= np.cos(2 * np.pi * i * x) * np.sin(2 * np.pi * i * y) * np.cos(2 * np.pi * i * z) * np.sin(np.pi * np.sqrt(12).item() * c * i * t)
                                return np.sqrt(eps / mu * 3 / 4) * s
                            case 'Hz':
                                for i in range(1, n + 1):
                                    s += np.cos(2 * np.pi * i * x) * np.cos(2 * np.pi * i * y) * np.sin(2 * np.pi * i * z) * np.sin(np.pi * np.sqrt(12).item() * c * i * t)
                                return np.sqrt(eps / mu * 3 / 4) * s
                elif solution == 6:
                    def solved(x, y, z, t, d, g, dt):
                        coefs = [] # fx,fy,fz,ft,ey,ez,hx,hy,hz
                        for i in range(1, max(p) + 1):
                            fx = min(i, p[0])
                            fy = i
                            fz = min(i, p[1])
                            ft = -np.sqrt(fx ** 2 + fy ** 2 + fz ** 2) * c
                            ey = -fx * np.sqrt(fy) / 2
                            ez = (np.sqrt(fy) * fy - 2) * fx / (2 * fz)
                            hx = (ey * fz - ez * fy) / (ft * mu)
                            hy = (ez * fx - fz) / (ft * mu)
                            hz = (fy - ey * fx) / (ft * mu)
                            coefs.append([fx,fy,fz,ft,ey,ez,hx,hy,hz])
                        match d:
                            case 'Ex':
                                return sum([np.cos(2 * np.pi * np.cos(2 * np.pi * (coefs[i - 1][0] * x + coefs[i - 1][1] * y + coefs[i - 1][2] * z + coefs[i - 1][3] * t))) for i in range(1, max(p) + 1)])
                            case 'Ey':
                                return sum([coefs[i - 1][4] * np.cos(2 * np.pi * np.cos(2 * np.pi * (coefs[i - 1][0] * x + coefs[i - 1][1] * y + coefs[i - 1][2] * z + coefs[i - 1][3] * t))) for i in range(1, max(p) + 1)])
                            case 'Ez':
                                return sum([coefs[i - 1][5] * np.cos(2 * np.pi * np.cos(2 * np.pi * (coefs[i - 1][0] * x + coefs[i - 1][1] * y + coefs[i - 1][2] * z + coefs[i - 1][3] * t))) for i in range(1, max(p) + 1)])
                            case 'Hx':
                                return sum([coefs[i - 1][6] * np.cos(2 * np.pi * np.cos(2 * np.pi * (coefs[i - 1][0] * x + coefs[i - 1][1] * y + coefs[i - 1][2] * z + coefs[i - 1][3] * t))) for i in range(1, max(p) + 1)])
                            case 'Hy':
                                return sum([coefs[i - 1][7] * np.cos(2 * np.pi * np.cos(2 * np.pi * (coefs[i - 1][0] * x + coefs[i - 1][1] * y + coefs[i - 1][2] * z + coefs[i - 1][3] * t))) for i in range(1, max(p) + 1)])
                            case 'Hz':
                                return sum([coefs[i - 1][8] * np.cos(2 * np.pi * np.cos(2 * np.pi * (coefs[i - 1][0] * x + coefs[i - 1][1] * y + coefs[i - 1][2] * z + coefs[i - 1][3] * t))) for i in range(1, max(p) + 1)])
            for d in order:
                EH['solver'][d] =lambda x,y,z,t,g,dt: solved(x,y,z,t,d,g,dt)
    info = {cat : [] for cat in 'sims,grids,times,round,svd,qr,add,norm,errors,sizes,rank1,rank2'.split(',')}
    while grid_size <= end_size or solver:
        #All solutions on [0, 1]^3
        if boundary == "PEC":
            cell_size = 1 / (grid_size)
        elif boundary == "Periodic":
            cell_size = 1 / grid_size
        #t = 1 / grid_size / c / np.sqrt(3).item() * 0.9330127
        t = 0.9 / grid_size / c / np.sqrt(3).item()
        #Reused constants
        eps1 = t / eps / cell_size
        mu1 = -t / mu / cell_size

        if not solver:
            info['grids'].append(grid_size)
            for key in info.keys():
                if key not in ['sims', 'grids']:
                    info[key].append([])

        if boundary == "PEC":
            gu = np.linspace(0, 1, 2 * grid_size + 1, dtype = precision if npy else pre2)
            x = np.linspace(0, 1, grid_size * 2 + 1, dtype = precision if npy else pre2)
        elif boundary == "Periodic":
            gu = np.linspace(1 / grid_size / 2, 1, 2 * grid_size, dtype = precision if npy else pre2)
            x = np.linspace(1 / grid_size / 2, 1, grid_size * 2, dtype = precision if npy else pre2)
        
        for sim in simulations:
            if not solver:
                simulation_type = sim['type']
                error = sim['error']
                zero_thres = sim['zero']
                if zero_thres:
                    group = True
                else:
                    group = False
                if simulation_type == 0:
                    print('Basic simulation', flush = True)
                    if grid_size == sizes[0]:
                        info['sims'].append('basic')
                else:
                    print(f'Error {error}, {f"group round with zero {zero_thres}" if group else "groupless"}', flush = True)
                    if grid_size == sizes[0]:
                        info['sims'].append(f'{error}-{zero_thres}')
            else:
                print(f'Solver for {ending} with {"numpy" if npy else "torch"}', flush = True)
                simulation_type = 0

            if npy:   
                X, Y, Z = np.meshgrid(x, x, x, indexing = 'ij')
            else:
                y = torch.tensor(x, device = device, dtype = precision)
        
            #Create Tensors; Assume appropriate yee cell for both H at each half integer timestep and E at each integer timestep
            if analytic:
                if npy:
                    for d in order:
                        EH['solving'][d] = EH['solver'][d](X[ixer[d]], Y[ixer[d]], Z[ixer[d]], yee[d][3] * t, grid_size)
                else:
                    for d in order:
                        #EH['solving'][d] = torch.tensor(EH['solver'][d](X[ixer[d]], Y[ixer[d]], Z[ixer[d]], yee[d][3] * t, grid_size), device = device, dtype = precision)
                        EH['solving'][d] = EH['solver'][d](y[ixer[d][0], None, None], y[None, ixer[d][1], None], y[None, None, ixer[d][2]], yee[d][3] * t, grid_size, t)
            elif boundary == "PEC":
                if solution == 2:
                    grid_size += 1
                    EH['solving']['Ex'] = torch.zeros((grid_size - 1, grid_size, grid_size), device = device, dtype = precision)
                    EH['solving']['Hz'] = torch.zeros((grid_size - 1, grid_size - 1, grid_size), device = device, dtype = precision)
                    EH['solving']['Hy'] = sum([np.sqrt(eps/mu) * 2 * np.cos(np.pi * min(i, p[0]) * (X[1::2, ::2, 1::2] + np.sqrt(3) * c * t / 2)) * np.cos(np.pi * i * (Y[1::2, ::2, 1::2] + np.sqrt(3) * c * t / 2)) * np.cos(np.pi * min(i, p[1]) * (Z[1::2, ::2, 1::2] + np.sqrt(3) * c * t / 2)) for i in range(1, max(p) + 1)])
                    EH['solving']['Ey'] = sum([np.sqrt(eps/mu) * 2 * np.cos(np.pi * min(i, p[0]) * (X[::2, 1::2, ::2])) * np.cos(np.pi * i * (Y[::2, 1::2, ::2])) * np.cos(np.pi * min(i, p[1]) * (Z[::2, 1::2, ::2])) for i in range(1, max(p) + 1)])
                    if not npy:
                        EH['solving']['Hy'] = torch.tensor(EH['solving']['Hy'], device = device, dtype = precision)
                        EH['solving']['Ey'] = torch.tensor(EH['solving']['Ey'], device = device, dtype = precision)
                    EH['solving']['Ez'] = torch.zeros((grid_size, grid_size, grid_size - 1), device = device, dtype = precision)
                    EH['solving']['Hx'] = torch.zeros((grid_size, grid_size - 1, grid_size - 1), device = device, dtype = precision)
                    grid_size -= 1
                elif solution == 3:
                    EH['solving']['Ex'] = torch.zeros((grid_size - 1, grid_size, grid_size), device = device, dtype = precision)
                    EH['solving']['Hy'] = torch.zeros((grid_size - 1, grid_size, grid_size - 1), device = device, dtype = precision)
                    EH['solving']['Hz'] = torch.zeros((grid_size - 1, grid_size - 1, grid_size), device = device, dtype = precision)
                    EH['solving']['Ey'] = torch.zeros((grid_size, grid_size - 1, grid_size), device = device, dtype = precision)
                    EH['solving']['Ez'] = torch.zeros((grid_size, grid_size, grid_size - 1), device = device, dtype = precision)
                    EH['solving']['Hx'] = torch.zeros((grid_size, grid_size - 1, grid_size - 1), device = device, dtype = precision)
            if boundary == "PEC":
                EH['solving']['Ex'][:, :, 0] = 0
                EH['solving']['Ex'][:, :, -1] = 0
                EH['solving']['Ex'][:, 0, :] = 0
                EH['solving']['Ex'][:, -1, :] = 0
                EH['solving']['Ey'][:, :, 0] = 0
                EH['solving']['Ey'][:, :, -1] = 0
                EH['solving']['Ey'][0, :, :] = 0
                EH['solving']['Ey'][-1, :, :] = 0
                EH['solving']['Ez'][0, :, :] = 0
                EH['solving']['Ez'][-1, :, :] = 0
                EH['solving']['Ez'][:, 0, :] = 0
                EH['solving']['Ez'][:, -1, :] = 0
            #torch.cuda.synchronize()
            if simulation_type == 1:
                for d in order:
                    EH['solving'][d] = TT.TTarray(EH['solving'][d], error)
                if not npy:
                    torch.cuda.empty_cache()
                lranks = [EH['solving'][d].ranks()[1:3] for d in order]
                size = sum([EH['solving'][d].nbytes() for d in order])                

                for key in TT.times.keys():
                    TT.times[key] = 0
            timing = 0
            if device == 'cuda':
                events = [torch.cuda.Event(enable_timing = True) for _ in range(2)] #start, end
            hOrder = 2
            if hOrder == 2:
                coefs = [-1, 1]
            elif hOrder == 4:
                coefs = [1/24, -27/24, 27/24, -1/24]
            def ecurl(d, rd):
                return sum([coefs[j] * roll(EH['solving'][d], (hOrder // 2) - 1 - j, rd) for j in range(hOrder)])
            def hcurl(d, rd):
                return sum([coefs[j] * roll(EH['solving'][d], (hOrder // 2) - j, rd) for j in range(hOrder)])
            #The actual simulation update loop
            for i in range(grid_size * iters):
                if i % (iters) == 0:
                    print(f'{round(i / iters)}/{grid_size}', flush = True)
                if simulation_type == 0:
                    if device == 'cuda':
                        events[0].record()
                    else:           
                        timing -= time.time()
                    if boundary == "Periodic":
                        EH['solving']['Hx'] += mu1 * (ecurl('Ez', 1) - ecurl('Ey', 2))
                        EH['solving']['Hy'] += mu1 * (ecurl('Ex', 2) - ecurl('Ez', 0))
                        EH['solving']['Hz'] += mu1 * (ecurl('Ey', 0) - ecurl('Ex', 1))
                        EH['solving']['Ex'] += eps1 * (hcurl('Hz', 1) - hcurl('Hy', 2))
                        EH['solving']['Ey'] += eps1 * (hcurl('Hx', 2) - hcurl('Hz', 0))
                        EH['solving']['Ez'] += eps1 * (hcurl('Hy', 0) - hcurl('Hx', 1))
                    elif boundary == "PEC":
                        EH['solving']['Hx'] += mu1 * (EH['solving']['Ez'][:, 1:, :] - EH['solving']['Ez'][:, :-1, :] - EH['solving']['Ey'][:, :, 1:] + EH['solving']['Ey'][:, :, :-1])
                        EH['solving']['Hy'] += mu1 * (EH['solving']['Ex'][:, :, 1:] - EH['solving']['Ex'][:, :, :-1] - EH['solving']['Ez'][1:, :, :] + EH['solving']['Ez'][:-1, :, :])
                        EH['solving']['Hz'] += mu1 * (EH['solving']['Ey'][1:, :, :] - EH['solving']['Ey'][:-1, :, :] - EH['solving']['Ex'][:, 1:, :] + EH['solving']['Ex'][:, :-1, :])
                        EH['solving']['Ex'][:, 1:-1, 1:-1] += eps1 * (EH['solving']['Hz'][:, 1:, 1:-1] - EH['solving']['Hz'][:, :-1, 1:-1] - EH['solving']['Hy'][:, 1:-1, 1:] + EH['solving']['Hy'][:, 1:-1, :-1])
                        EH['solving']['Ey'][1:-1, :, 1:-1] += eps1 * (EH['solving']['Hx'][1:-1, :, 1:] - EH['solving']['Hx'][1:-1, :, :-1] - EH['solving']['Hz'][1:, :, 1:-1] + EH['solving']['Hz'][:-1, :, 1:-1])
                        EH['solving']['Ez'][1:-1, 1:-1, :] += eps1 * (EH['solving']['Hy'][1:, 1:-1, :] - EH['solving']['Hy'][:-1, 1:-1, :] - EH['solving']['Hx'][1:-1, 1:, :] + EH['solving']['Hx'][1:-1, :-1, :])
                        #print([torch.linalg.norm(EH['solving'][d]).item() for d in order])
                    if device == 'cuda':
                        events[1].record()
                    else:           
                        timing += time.time()
                else:
                    if device == 'cuda':
                        events[0].record()
                    else:           
                        timing -= time.time()
                    if boundary == "Periodic":
                        EH['solving']['Hx'] += mu1 * (EH['solving']['Ey'].rollsum([[-1, 2, -1]]) - EH['solving']['Ez'].rollsum([[-1, 1, -1]]))
                        EH['solving']['Hy'] += mu1 * (EH['solving']['Ez'].rollsum([[-1, 0, -1]]) - EH['solving']['Ex'].rollsum([[-1, 2, -1]]))
                        EH['solving']['Hz'] += mu1 * (EH['solving']['Ex'].rollsum([[-1, 1, -1]]) - EH['solving']['Ey'].rollsum([[-1, 0, -1]]))
                        if not TT.roundInPlus:
                            if group:
                                EH['solving']['Hx'], EH['solving']['Hy'], EH['solving']['Hz'] = TT.group_round((EH['solving']['Hx'], EH['solving']['Hy'], EH['solving']['Hz']), error, zero_thres)
                            else:
                                for d in 'Hx,Hy,Hz'.split(','):
                                    EH['solving'][d] = EH['solving'][d].round()
                        EH['solving']['Ex'] += eps1 * (EH['solving']['Hz'].rollsum([[1, 1, -1]]) - EH['solving']['Hy'].rollsum([[1, 2, -1]]))
                        EH['solving']['Ey'] += eps1 * (EH['solving']['Hx'].rollsum([[1, 2, -1]]) - EH['solving']['Hz'].rollsum([[1, 0, -1]]))
                        EH['solving']['Ez'] += eps1 * (EH['solving']['Hy'].rollsum([[1, 0, -1]]) - EH['solving']['Hx'].rollsum([[1, 1, -1]]))
                        if not TT.roundInPlus:
                            if group:
                                EH['solving']['Ex'], EH['solving']['Ey'], EH['solving']['Ez'] = TT.group_round((EH['solving']['Ex'], EH['solving']['Ey'], EH['solving']['Ez']), error, zero_thres)
                            else:
                               for d in 'Ex,Ey,Ez'.split(','):
                                    EH['solving'][d] = EH['solving'][d].round() 
                    elif boundary == "PEC":
                        EH['solving']['Hx'] += mu1 * (EH['solving']['Ez'].reducedSum(1, [(1, None), (0, -1)], [1, -1]) - EH['solving']['Ey'].reducedSum(2, [(1, None), (0, -1)], [1, -1]))
                        EH['solving']['Hy'] += mu1 * (EH['solving']['Ex'].reducedSum(2, [(1, None), (0, -1)], [1, -1]) - EH['solving']['Ez'].reducedSum(0, [(1, None), (0, -1)], [1, -1]))
                        EH['solving']['Hz'] += mu1 * (EH['solving']['Ey'].reducedSum(0, [(1, None), (0, -1)], [1, -1]) - EH['solving']['Ex'].reducedSum(1, [(1, None), (0, -1)], [1, -1]))
                        if not TT.roundInPlus:
                            if group:
                                EH['solving']['Hx'], EH['solving']['Hy'], EH['solving']['Hz'] = TT.group_round((EH['solving']['Hx'], EH['solving']['Hy'], EH['solving']['Hz']), error, zero_thres)
                            else:
                                for d in 'Hx,Hy,Hz'.split(','):
                                    EH['solving'][d] = EH['solving'][d].round()
                        EH['solving']['Ex'] += eps1 * (EH['solving']['Hz'].reduce([':', ':', (1, -1)]).reducedSum(1, [(1, None), (0, -1)], [1, -1]) - EH['solving']['Hy'].reduce([':', (1, -1), ':']).reducedSum(2, [(1, None), (0, -1)], [1, -1])).pad([1, 2], [[2, 1], [2, 1]])
                        EH['solving']['Ey'] += eps1 * (EH['solving']['Hx'].reduce([(1, -1), ':', ':']).reducedSum(2, [(1, None), (0, -1)], [1, -1]) - EH['solving']['Hz'].reduce([':', ':', (1, -1)]).reducedSum(0, [(1, None), (0, -1)], [1, -1])).pad([0, 2], [[2, 1], [2, 1]])
                        EH['solving']['Ez'] += eps1 * (EH['solving']['Hy'].reduce([':', (1, -1), ':']).reducedSum(0, [(1, None), (0, -1)], [1, -1]) - EH['solving']['Hx'].reduce([(1, -1), ':', ':']).reducedSum(1, [(1, None), (0, -1)], [1, -1])).pad([0, 1], [[2, 1], [2, 1]])
                        if not TT.roundInPlus:
                            if group:
                                EH['solving']['Ex'], EH['solving']['Ey'], EH['solving']['Ez'] = TT.group_round((EH['solving']['Ex'], EH['solving']['Ey'], EH['solving']['Ez']), error, zero_thres)
                            else:
                               for d in 'Ex,Ey,Ez'.split(','):
                                    EH['solving'][d] = EH['solving'][d].round() 
                    if device == 'cuda':
                        events[1].record()
                    else:           
                        timing += time.time()
                    for j, d in enumerate(order):
                        eranks = EH['solving'][d].ranks()
                        lranks[j][0] = max(lranks[j][0], eranks[1])
                        lranks[j][1] = max(lranks[j][1], eranks[2])
                    size = max(sum([EH['solving'][d].nbytes() for d in order]), size)
                if device == 'cuda':
                    torch.cuda.synchronize()
                    timing += events[0].elapsed_time(events[1]) / 1000
                    if simulation_type == 1:
                        for key in TT.times.keys():                      
                            for j in range(0, len(TT.events[key]), 2):
                                TT.times[key] += TT.events[key][j].elapsed_time(TT.events[key][j + 1]) / 1000
                            TT.events[key] = []
            #After simulation; data fetching
            if solver:
                os.makedirs(ending, exist_ok = True)
                for d in order:
                    np.save(os.path.join(ending, d), EH['solving'][d].cpu().numpy())
                    del EH['solving'][d]
                del x
                torch.cuda.empty_cache()
                break
            if not analytic:
                if not ignore_error:
                    for d in order:
                        EH['solved'][d] = EH['solver'][d](np.stack(np.meshgrid(*(gu[ixer[d][i]] for i in range(3)), indexing = 'ij'), axis = -1))
            else:
                if npy:
                    for d in order:
                        EH['solved'][d] = EH['solver'][d](*np.meshgrid(*(gu[ixer[d][i]] for i in range(3)), indexing = 'ij'), (iters * grid_size + yee[d][3]) * t, grid_size)
                else:
                    y = torch.tensor(gu, device = device, dtype = precision)
                    for d in order:
                        EH['solved'][d] = EH['solver'][d](y[ixer[d][0], None, None], y[None, ixer[d][1], None], y[None, None, ixer[d][2]], (iters * grid_size + yee[d][3]) * t, grid_size, t).cpu().numpy()
            info['times'][-1].append(timing)
            info['rank1'][-1].append([])
            info['rank2'][-1].append([])
            if simulation_type == 0:
                info['sizes'][-1].append(sum([EH['solving'][d].nbytes for d in order]))
                for key in TT.times.keys():
                    info[key][-1].append([])
            else:
                #Error calculations based on https://www.sciencedirect.com/science/article/pii/S0378475423001313, as seen in 5.1
                for d in order:
                    EH['solving'][d] = EH['solving'][d].raw()
                info['sizes'][-1].append(size)
                for key in TT.times.keys():
                    info[key][-1].append(TT.times[key])
                for i in range(6):
                    info['rank1'][-1][-1].append(lranks[i][0])
                    info['rank2'][-1][-1].append(lranks[i][1])
            if not npy:
                for d in order:
                    EH['solving'][d] = EH['solving'][d].cpu().numpy()
            if not ignore_error or analytic:
                info['errors'][-1].append(
                    np.sqrt(
                        (
                            (
                                np.sum((EH['solving']['Ex'] - EH['solved']['Ex']) ** 2) +
                                np.sum((EH['solving']['Ey'] - EH['solved']['Ey']) ** 2) +
                                np.sum((EH['solving']['Ez'] - EH['solved']['Ez']) ** 2)
                            ) / (np.sum(EH['solved']['Ex'] ** 2) + np.sum(EH['solved']['Ey'] ** 2) + np.sum(EH['solved']['Ez'] ** 2))
                            # + (
                            #     np.sum((EH['solving']['Hx'] - EH['solved']['Hx']) ** 2) +
                            #     np.sum((EH['solving']['Hy'] - EH['solved']['Hy']) ** 2) +
                            #     np.sum((EH['solving']['Hz'] - EH['solved']['Hz']) ** 2)
                            # ) / (np.sum(EH['solved']['Hx'] ** 2) + np.sum(EH['solved']['Hy'] ** 2) + np.sum(EH['solved']['Hz'] ** 2))
                        )
                    )
                )
            if not npy:
                torch.cuda.empty_cache()
        grid_size *= 2
        with open(f"{ending}.json", "w") as f:
            json.dump(info, f)
        if solver:
            break