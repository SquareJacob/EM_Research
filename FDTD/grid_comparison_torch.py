import numpy as np
import time
import matplotlib.pyplot as plt
import json
import os
from typing import Literal, Tuple, Union

#IMPORTANT: While the code itself may not be written as optimally as possible, the algorithms are

def full_test(boundary: Literal["PEC", "Periodic"], solution: int, iters: int, sizes: Tuple[int], simulation_type: int, npy: bool, error: float = 0, caps: Union[list[list[int]], Literal['d']] = 'd', zero_thres: float = 0, m: int = 10, a: Tuple[int, int] = (-0.488, 0.145), device: Literal['cpu', 'cuda'] = 'cuda', save_type: int = 0, eps: float = 8.854e-12, mu: float = np.pi * 4e-7):
    """
    Full Test Run of FDTD, with assumed order of Ex, Ey, Ez, Hx, Hy, Hz.

    :param boundary: Boundary condition
    :param solution: Predetermined solution to use, see 'Solutions' directory for more details
    :param iters: Number of iterations; one iteration is sqrt(eps*mu/3)
    :param sizes: start size, end size and optionally solver size. Note that on each loop, grid size doubles
    :param simulation_type: 0 = Basic FDTD only, 1 = TT FDTD only, 2 = Both FDTD
    :param npy: Whether to use numpy or torch
    :param error: error threshold for TT
    :param caps: Caps to use for each TT-tensor. Each solution has default optimal caps
    :param zero_thres: Relative threshold needed for zeroing-out tensor
    :param m: the m for solutions it is used in
    :param a: the a for solutions it is used in
    :param device: what device to use if pytorch; cpu always used for numpy
    :param save_type: How to save results when simulation_type = 0; 0 = Nothing saved, 1 = End saved, 2 = All saved, 3 = All error saved
    :param eps: value for epsilon
    :param mu: value for mu
    """
    #Initial setup
    n = m
    c = 1/np.sqrt(mu * eps).item()
    grid_size = sizes[0]
    end_size = sizes[1]
    order = 'Ex,Ey,Ez,Hx,Hy,Hz'.split(',') #Ordering for operations done on everything
    yee = { #Yee grid and inital times
        'Ex': (1, 0, 0, 0),
        'Ey': (0, 1, 0, 0),
        'Ez': (0, 0, 1, 0),
        'Hx': (0, 1, 1, -1/2),
        'Hy': (1, 0, 1, -1/2),
        'Hz': (1, 1, 0, -1/2)
    }
    if len(sizes) > 2:
        solver_size = sizes[2]
    else:
        solver_size = end_size
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
    TT.zero_thres = zero_thres

    if caps == None:
        caps = [None for i in range(6)]
    if caps == 'd': #Default caps
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
                caps[3] = [1, 1]

    #Figuring out simulation from boundary and solution number
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

    EHk = ['solver', 'basic', 'TT', 'solved', 'saved']
    EH = {k: {} for k in EHk}
    ixer = {k: tuple(slice(yee[k][i], None, 2) for i in range(3)) for k in order} #indexing when using half-spaced grid

    if not analytic: #Approximate non-analytic solutions using pre-solved solution using interpolation
        if simulation_type > 0:
            if boundary == 'PEC':
                su = np.linspace(0, 1, 2 * solver_size - 1, dtype = precision if npy else pre2)
            elif boundary == 'Periodic':
                su = np.linspace(1 / solver_size / 2, 1, 2 * solver_size, dtype = precision if npy else pre2)
            from scipy.interpolate import RegularGridInterpolator
            for d in order:
                EH['solver'][d] = RegularGridInterpolator(tuple(su[ixer[d][i]] for i in range(3)), np.load(os.path.join(ending, f'{d}.npy')))
    else: #Otherwise, setup analytic solution
        if boundary == "PEC": 
            if solution == 1:
                def solved(x, y, z, t, d):
                    s = np.zeros_like(x)
                    match d:
                        case 'Ex':
                            for i in range(1, n + 1):
                                s += np.sin(np.pi * i * y) * np.sin(np.pi * i * z) * np.cos(np.pi * np.sqrt(2) * c * t * i)
                        case 'Hy':
                            for i in range(1, n + 1):
                                s -= np.sin(np.pi * i * y) * np.cos(np.pi * i * z) * np.sin(np.pi * np.sqrt(2) * c * t * i)
                            return np.sqrt(eps / (2 * mu)).item() * s
                        case 'Hz':
                            for i in range(1, n + 1):
                                s += np.cos(np.pi * i * y) * np.sin(np.pi * i * z) * np.sin(np.pi * np.sqrt(2) * c * t * i)
                            return np.sqrt(eps / (2 * mu)).item() * s
                    return s
            elif solution == 4:
                def solved(x, y, z, t, d):
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
                def solved(x, y, z, t, d):
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
        elif boundary == "Periodic":
            if solution == 1:
                def solved(x, y, z, t, d):
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
                def solved(x, y, z, t, d):
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
                def solved(x, y, z, t, d):
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
                def solved(x, y, z, t, d):
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
                def solved(x, y, z, t, d):
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
        for d in order:
            EH['solver'][d] =lambda x,y,z,t: solved(x,y,z,t,d)

    info = {
        i: [] for i in 'grids,basic_times,tt_times,round_times,qr_times,svd_times,add_times,norm_times,basic_errors,tt_errors,basic_sizes,tt_sizes,ranks'.split(',')
    }
    info["ranks"] = [[[] for j in range(6)] for i in range(2)]

    print("Started solving", flush = True)
    while(grid_size <= end_size and (analytic or grid_size <= solver_size)):
        #All solutions on [0, 1]^3
        if boundary == "PEC":
            cell_size = 1 / (grid_size - 1)
        elif boundary == "Periodic":
            cell_size = 1 / grid_size
        t = 1 / grid_size / c / np.sqrt(3).item() #Accounts for Courant limit
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
        
        #Create Tensors; Assume appropirate yee cell for both H at each half integer timestep and E at each integer timestep
        if analytic:
            if npy:
                for d in order:
                    EH['basic'][d] = EH['solver'][d](X[ixer[d]], Y[ixer[d]], Z[ixer[d]], yee[d][3] * t)
            else:
                for d in order:
                    EH['basic'][d] = torch.tensor(EH['solver'][d](X[ixer[d]].cpu().numpy(), Y[ixer[d]].cpu().numpy(), Z[ixer[d]].cpu().numpy(), yee[d][3] * t), device = device, dtype = precision)
        elif boundary == "PEC":
            if solution == 2:
                EH['basic']['Ex'] = torch.zeros((grid_size - 1, grid_size, grid_size), device = device, dtype = precision)
                EH['basic']['Hy'] = np.sqrt(eps/mu) * 2 * torch.sin(np.pi * (X[1::2, ::2, 1::2] + c * t / 2)) * torch.sin(np.pi * (Y[1::2, ::2, 1::2] + c * t / 2)) * torch.sin(np.pi * (Z[1::2, ::2, 1::2] + c * t / 2))
                EH['basic']['Hz'] = torch.zeros((grid_size - 1, grid_size - 1, grid_size), device = device, dtype = precision)
            elif solution == 3:
                EH['basic']['Ex'] = torch.zeros((grid_size - 1, grid_size, grid_size), device = device, dtype = precision)
                EH['basic']['Hy'] = torch.zeros((grid_size - 1, grid_size, grid_size - 1), device = device, dtype = precision)
                EH['basic']['Hz'] = torch.zeros((grid_size - 1, grid_size - 1, grid_size), device = device, dtype = precision)
            EH['basic']['Ey'] = torch.zeros((grid_size, grid_size - 1, grid_size), device = device, dtype = precision)
            EH['basic']['Ez'] = torch.zeros((grid_size, grid_size, grid_size - 1), device = device, dtype = precision)
            EH['basic']['Hx'] = torch.zeros((grid_size, grid_size - 1, grid_size - 1), device = device, dtype = precision)
        if boundary == "PEC":
            EH['basic']['Ex'][:, :, 0] = 0
            EH['basic']['Ex'][:, :, -1] = 0
            EH['basic']['Ex'][:, 0, :] = 0
            EH['basic']['Ex'][:, -1, :] = 0
            EH['basic']['Ey'][:, :, 0] = 0
            EH['basic']['Ey'][:, :, -1] = 0
            EH['basic']['Ey'][0, :, :] = 0
            EH['basic']['Ey'][-1, :, :] = 0
            EH['basic']['Ez'][0, :, :] = 0
            EH['basic']['Ez'][-1, :, :] = 0
            EH['basic']['Ez'][:, 0, :] = 0
            EH['basic']['Ez'][:, -1, :] = 0
        

        if not npy:
            del x, X, Y, Z
            torch.cuda.empty_cache()

        if simulation_type == 0:
            if save_type in [2, 3]:
                for d in order:
                    EH['saved'][d] = torch.zeros((iters * grid_size + 1, *EH['basic'][d].shape), device = device, dtype = precision)
                    EH['saved'][d][0, :, :, :] = EH['basic'][d]
            if source and solution == 4:
                if npy:
                    X = -np.pi * c * t * np.array([np.outer(np.sin(j * np.pi * gu[::2][1:-1]), np.sin(j * np.pi * gu[::2][1:-1])) for j in range(1, n + 1)])
                else:
                    X = -np.pi * c * t * torch.tensor(np.array([np.outer(np.sin(j * np.pi * gu[::2][1:-1]), np.sin(j * np.pi * gu[::2][1:-1])) for j in range(1, n + 1)]), device = device, dtype = precision)
        else:
            for i, d in enumerate(order):
                EH['TT'][d] = TT.TTarray(EH['basic'][d], error, caps[i])
            if simulation_type == 1:
                if not npy:
                    for d in order:
                        del EH['basic'][d]
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
            lranks = [EH['TT'][d].ranks()[1:3] for d in order]
            tt_size = sum([EH['TT'][d].nbytes() for d in order])

        for key in TT.times.keys():
            TT.times[key] = 0
        basic_time = 0
        tt_time = 0
        if device == 'cuda':
            events = [torch.cuda.Event(enable_timing = True) for _ in range(4)] #Basic, TT
        #The actual simulation update loop
        for i in range(grid_size * iters):
            if i % iters == 0:
                print(f'{round(i / iters)}/{grid_size}', flush = True)
            if simulation_type in [0, 2]:
                if device == 'cuda':
                    events[0].record()
                else:           
                    basic_time -= time.time()
                if boundary == "Periodic":
                    EH['basic']['Hx'] += mu1 * (roll(EH['basic']['Ez'], -1, 1) - EH['basic']['Ez'] - roll(EH['basic']['Ey'], -1, 2) + EH['basic']['Ey'])
                    EH['basic']['Hy'] += mu1 * (roll(EH['basic']['Ex'], -1, 2) - EH['basic']['Ex'] - roll(EH['basic']['Ez'], -1, 0) + EH['basic']['Ez'])
                    EH['basic']['Hz'] += mu1 * (roll(EH['basic']['Ey'], -1, 0) - EH['basic']['Ey'] - roll(EH['basic']['Ex'], -1, 1) + EH['basic']['Ex'])
                    EH['basic']['Ex'] += eps1 * (EH['basic']['Hz'] - roll(EH['basic']['Hz'], 1, 1) - EH['basic']['Hy'] + roll(EH['basic']['Hy'], 1, 2))
                    EH['basic']['Ey'] += eps1 * (EH['basic']['Hx'] - roll(EH['basic']['Hx'], 1, 2) - EH['basic']['Hz'] + roll(EH['basic']['Hz'], 1, 0))
                    EH['basic']['Ez'] += eps1 * (EH['basic']['Hy'] - roll(EH['basic']['Hy'], 1, 0) - EH['basic']['Hx'] + roll(EH['basic']['Hx'], 1, 1))
                elif boundary == "PEC":
                    EH['basic']['Hx'] += mu1 * (EH['basic']['Ez'][:, 1:, :] - EH['basic']['Ez'][:, :-1, :] - EH['basic']['Ey'][:, :, 1:] + EH['basic']['Ey'][:, :, :-1])
                    EH['basic']['Hy'] += mu1 * (EH['basic']['Ex'][:, :, 1:] - EH['basic']['Ex'][:, :, :-1] - EH['basic']['Ez'][1:, :, :] + EH['basic']['Ez'][:-1, :, :])
                    EH['basic']['Hz'] += mu1 * (EH['basic']['Ey'][1:, :, :] - EH['basic']['Ey'][:-1, :, :] - EH['basic']['Ex'][:, 1:, :] + EH['basic']['Ex'][:, :-1, :])
                    EH['basic']['Ex'][:, 1:-1, 1:-1] += eps1 * (EH['basic']['Hz'][:, 1:, 1:-1] - EH['basic']['Hz'][:, :-1, 1:-1] - EH['basic']['Hy'][:, 1:-1, 1:] + EH['basic']['Hy'][:, 1:-1, :-1])
                    EH['basic']['Ey'][1:-1, :, 1:-1] += eps1 * (EH['basic']['Hx'][1:-1, :, 1:] - EH['basic']['Hx'][1:-1, :, :-1] - EH['basic']['Hz'][1:, :, 1:-1] + EH['basic']['Hz'][:-1, :, 1:-1])
                    EH['basic']['Ez'][1:-1, 1:-1, :] += eps1 * (EH['basic']['Hy'][1:, 1:-1, :] - EH['basic']['Hy'][:-1, 1:-1, :] - EH['basic']['Hx'][1:-1, 1:, :] + EH['basic']['Hx'][1:-1, :-1, :])
                    if source:
                        if solution == 3:
                            if i < grid_size:
                                T = (i + 1) * 2 * np.pi / grid_size
                                x = (a[0] * np.sin(T) + a[1] * 2 * np.sin(2 * T) - (a[0] + 4 * a[1]) / 3 * np.sin(3 * T)) * grid_size * grid_size
                                EH['basic']['Ex'][grid_size // 2, grid_size // 2, grid_size // 2] += x
                        elif solution == 4:
                            x = sum([j * np.sin(2 * j * np.pi * c * t * (i + 0.5)) * X[j - 1, :, :] for j in range(1, n + 1)])
                            EH['basic']['Ex'][:, 1:-1, 1:-1] += x
                if device == 'cuda':
                    events[1].record()
                else:           
                    basic_time += time.time()
                if simulation_type == 0:
                    if save_type in [2, 3]:
                        for d in order:
                            EH['saved'][d][i + 1, :, :, :] = EH['basic'][d]
            if simulation_type > 0:
                if device == 'cuda':
                    events[2].record()
                else:           
                    tt_time -= time.time()
                if boundary == "Periodic":
                    EH['TT']['Hx'] += mu1 * (EH['TT']['Ey'].rollsum([[-1, 2, -1]]) - EH['TT']['Ez'].rollsum([[-1, 1, -1]]))
                    EH['TT']['Hy'] += mu1 * (EH['TT']['Ez'].rollsum([[-1, 0, -1]]) - EH['TT']['Ex'].rollsum([[-1, 2, -1]]))
                    EH['TT']['Hz'] += mu1 * (EH['TT']['Ex'].rollsum([[-1, 1, -1]]) - EH['TT']['Ey'].rollsum([[-1, 0, -1]]))
                    if not TT.roundInPlus:
                        EH['TT']['Hx'], EH['TT']['Hy'], EH['TT']['Hz'] = TT.group_round((EH['TT']['Hx'], EH['TT']['Hy'], EH['TT']['Hz']))
                    EH['TT']['Ex'] += eps1 * (EH['TT']['Hz'].rollsum([[1, 1, -1]]) - EH['TT']['Hy'].rollsum([[1, 2, -1]]))
                    EH['TT']['Ey'] += eps1 * (EH['TT']['Hx'].rollsum([[1, 2, -1]]) - EH['TT']['Hz'].rollsum([[1, 0, -1]]))
                    EH['TT']['Ez'] += eps1 * (EH['TT']['Hy'].rollsum([[1, 0, -1]]) - EH['TT']['Hx'].rollsum([[1, 1, -1]]))
                    if not TT.roundInPlus:
                        EH['TT']['Ex'], EH['TT']['Ey'], EH['TT']['Ez'] = TT.group_round((EH['TT']['Ex'], EH['TT']['Ey'], EH['TT']['Ez']))
                elif boundary == "PEC":
                    EH['TT']['Hx'] += mu1 * (EH['TT']['Ez'].reducedSum(1, [(1, None), (0, -1)], [1, -1]) - EH['TT']['Ey'].reducedSum(2, [(1, None), (0, -1)], [1, -1]))
                    EH['TT']['Hy'] += mu1 * (EH['TT']['Ex'].reducedSum(2, [(1, None), (0, -1)], [1, -1]) - EH['TT']['Ez'].reducedSum(0, [(1, None), (0, -1)], [1, -1]))
                    EH['TT']['Hz'] += mu1 * (EH['TT']['Ey'].reducedSum(0, [(1, None), (0, -1)], [1, -1]) - EH['TT']['Ex'].reducedSum(1, [(1, None), (0, -1)], [1, -1]))
                    if not TT.roundInPlus:
                        EH['TT']['Hx'], EH['TT']['Hy'], EH['TT']['Hz'] = TT.group_round((EH['TT']['Hx'], EH['TT']['Hy'], EH['TT']['Hz']))
                    EH['TT']['Ex'] += eps1 * (EH['TT']['Hz'].reduce([':', ':', (1, -1)]).reducedSum(1, [(1, None), (0, -1)], [1, -1]) - EH['TT']['Hy'].reduce([':', (1, -1), ':']).reducedSum(2, [(1, None), (0, -1)], [1, -1])).pad([1, 2], [[2, 1], [2, 1]])
                    EH['TT']['Ey'] += eps1 * (EH['TT']['Hx'].reduce([(1, -1), ':', ':']).reducedSum(2, [(1, None), (0, -1)], [1, -1]) - EH['TT']['Hz'].reduce([':', ':', (1, -1)]).reducedSum(0, [(1, None), (0, -1)], [1, -1])).pad([0, 2], [[2, 1], [2, 1]])
                    EH['TT']['Ez'] += eps1 * (EH['TT']['Hy'].reduce([':', (1, -1), ':']).reducedSum(0, [(1, None), (0, -1)], [1, -1]) - EH['TT']['Hx'].reduce([(1, -1), ':', ':']).reducedSum(1, [(1, None), (0, -1)], [1, -1])).pad([0, 1], [[2, 1], [2, 1]])
                    if source:
                        if solution == 3:
                            if i < grid_size:
                                T = (i + 1) * 2 * np.pi / grid_size
                                x = (a[0] * np.sin(T) + a[1] * 2 * np.sin(2 * T) - (a[0] + 4 * a[1]) / 3 * np.sin(3 * T)) * grid_size * grid_size
                                P[1][0, grid_size // 2, 0] = x
                                EH['TT']['Ex'] += P
                        elif solution == 4:
                            if npy:
                                P[1][0, 1:-1, :] = np.array([np.sin(2 * j * np.pi * c * t * (i + 0.5)) * np.sin(j * np.pi * gu[::2][1:-1]) for j in range(1, n + 1)]).T
                                P[2][:, 1:-1, 0] = np.array([j * np.sin(j * np.pi * gu[::2][1:-1]) for j in range(1, n + 1)])
                            else:
                                P[1][0, 1:-1, :] = torch.tensor(np.array([np.sin(2 * j * np.pi * c * t * (i + 0.5)) * np.sin(j * np.pi * gu[::2][1:-1]) for j in range(1, n + 1)]).T, device = device, dtype = precision)
                                P[2][:, 1:-1, 0] = torch.tensor(np.array([j * np.sin(j * np.pi * gu[::2][1:-1]) for j in range(1, n + 1)]), device = device, dtype = precision)
                            EH['TT']['Ex'] += P
                    if not TT.roundInPlus:
                        EH['TT']['Ex'], EH['TT']['Ey'], EH['TT']['Ez'] = TT.group_round((EH['TT']['Ex'], EH['TT']['Ey'], EH['TT']['Ez']))
                if device == 'cuda':
                    events[3].record()
                else:           
                    tt_time += time.time()
                for j, d in enumerate(order):
                    eranks = EH['TT'][d].ranks()
                    lranks[j][0] = max(lranks[j][0], eranks[1])
                    lranks[j][1] = max(lranks[j][1], eranks[2])
                tt_size = max(sum([EH['TT'][d].nbytes() for d in order]), tt_size)
            if device == 'cuda':
                torch.cuda.synchronize()
                if simulation_type != 1:
                    basic_time += events[0].elapsed_time(events[1]) / 1000
                if simulation_type != 0:
                    tt_time += events[2].elapsed_time(events[3]) / 1000
                    for key in TT.times.keys():                      
                        for j in range(0, len(TT.events[key]), 2):
                            TT.times[key] += TT.events[key][j].elapsed_time(TT.events[key][j + 1]) / 1000
                        TT.events[key] = []
        #After simulation; data fetching
        if simulation_type == 0:
            print(basic_time, flush = True)
            if save_type > 0:
                os.makedirs(ending, exist_ok = True)
            if save_type == 3:
                if analytic:
                    for d in order:
                        EH['solved'][d] = EH['solver'][d](*np.meshgrid(*(gu[ixer[d][i]] for i in range(3)), (np.arange(iters * grid_size + 1) + yee[d][3]) * t, indexing = 'ij')).transpose(3, 0, 1, 2)
            if not npy:
                if save_type == 1:
                    for d in order:
                        np.save(os.path.join(ending, d), EH['basic'][d].cpu().numpy())
                elif save_type == 2:
                    for d in order:
                        np.save(os.path.join(ending, d), EH['saved'][d].cpu().numpy())
                        del EH['saved'][d]
                elif save_type == 3:
                    np.save(os.path.join(ending, 'err'), (np.sqrt(
                        (
                            (
                                np.sum((EH['saved']['Ex'].cpu().numpy() - EH['solved']['Ex']) ** 2, axis = tuple(i + 1 for i in range(3))) +
                                np.sum((EH['saved']['Ey'].cpu().numpy() - EH['solved']['Ey']) ** 2, axis = tuple(i + 1 for i in range(3))) +
                                np.sum((EH['saved']['Ez'].cpu().numpy() - EH['solved']['Ez']) ** 2, axis = tuple(i + 1 for i in range(3)))
                            ) / (np.sum(EH['solved']['Ex'] ** 2, axis = tuple(i + 1 for i in range(3))) + np.sum(EH['solved']['Ey'] ** 2, axis = tuple(i + 1 for i in range(3))) + np.sum(EH['solved']['Ez'] ** 2, axis = tuple(i + 1 for i in range(3)))) + 
                            (
                                np.sum((EH['saved']['Hx'].cpu().numpy() - EH['solved']['Hx']) ** 2, axis = tuple(i + 1 for i in range(3))) +
                                np.sum((EH['saved']['Hy'].cpu().numpy() - EH['solved']['Hy']) ** 2, axis = tuple(i + 1 for i in range(3))) +
                                np.sum((EH['saved']['Hz'].cpu().numpy() - EH['solved']['Hz']) ** 2, axis = tuple(i + 1 for i in range(3)))
                            ) / (np.sum(EH['solved']['Hx'] ** 2, axis = tuple(i + 1 for i in range(3))) + np.sum(EH['solved']['Hy'] ** 2, axis = tuple(i + 1 for i in range(3))) + np.sum(EH['solved']['Hz'] ** 2, axis = tuple(i + 1 for i in range(3))))
                        )
                    )))
                    for d in order:
                        del EH['saved'][d]
                for d in order:
                    del EH['basic'][d]
                if source and solution == 4:
                    del X
            else:
                if save_type == 1:
                    for d in order:
                        np.save(os.path.join(ending, d), EH['basic'][d])
                elif save_type in [2, 3]:
                    for d in order:
                        np.save(os.path.join(ending, d), EH['saved'][d])
        else:
            if not analytic:
                for d in order:
                    EH['solved'][d] = EH['solver'][d](np.meshgrid(*(gu[ixer[d][i]] for i in range(3)), indexing = 'ij'))
            else:
                for d in order:
                    EH['solved'][d] = EH['solver'][d](*np.meshgrid(*(gu[ixer[d][i]] for i in range(3)), indexing = 'ij'), (iters * grid_size + yee[d][3]) * t)

            #Error calculations based on https://www.sciencedirect.com/science/article/pii/S0378475423001313, as seen in 5.1
            if npy:
                if simulation_type == 2:
                    info['basic_errors'].append(
                        np.sqrt(
                                (
                                    (
                                        np.sum((EH['basic']['Ex'] - EH['solved']['Ex']).astype(np.float64) ** 2) +
                                        np.sum((EH['basic']['Ey'] - EH['solved']['Ey']).astype(np.float64) ** 2) +
                                        np.sum((EH['basic']['Ez'] - EH['solved']['Ez']).astype(np.float64) ** 2)
                                    ) / (np.sum(EH['solved']['Ex'] ** 2) + np.sum(EH['solved']['Ey'] ** 2) + np.sum(EH['solved']['Ez'] ** 2)) + 
                                    (
                                        np.sum((EH['basic']['Hx'] - EH['solved']['Hx']).astype(np.float64) ** 2) +
                                        np.sum((EH['basic']['Hy'] - EH['solved']['Hy']).astype(np.float64) ** 2) +
                                        np.sum((EH['basic']['Hz'] - EH['solved']['Hz']).astype(np.float64) ** 2)
                                    ) / (np.sum(EH['solved']['Hx'] ** 2) + np.sum(EH['solved']['Hy'] ** 2) + np.sum(EH['solved']['Hz'] ** 2))
                                )
                        ).item()
                    )
                info['tt_errors'].append(
                    np.sqrt(
                            (
                                (
                                    np.sum((EH['TT']['Ex'].raw(np.float64) - EH['solved']['Ex']) ** 2) +
                                    np.sum((EH['TT']['Ey'].raw(np.float64) - EH['solved']['Ey']) ** 2) +
                                    np.sum((EH['TT']['Ez'].raw(np.float64) - EH['solved']['Ez']) ** 2)
                                ) / (np.sum(EH['solved']['Ex'] ** 2) + np.sum(EH['solved']['Ey'] ** 2) + np.sum(EH['solved']['Ez'] ** 2)) + 
                                (
                                    np.sum((EH['TT']['Hx'].raw(np.float64) - EH['solved']['Hx']) ** 2) +
                                    np.sum((EH['TT']['Hy'].raw(np.float64) - EH['solved']['Hy']) ** 2) +
                                    np.sum((EH['TT']['Hz'].raw(np.float64) - EH['solved']['Hz']) ** 2)
                                ) / (np.sum(EH['solved']['Hx'] ** 2) + np.sum(EH['solved']['Hy'] ** 2) + np.sum(EH['solved']['Hz'] ** 2))
                            )
                        ).item()
                    )
            else:
                if simulation_type == 2:
                    info['basic_errors'].append(
                        np.sqrt(
                                (
                                    (
                                        np.sum((EH['basic']['Ex'].cpu().numpy() - EH['solved']['Ex']) ** 2) +
                                        np.sum((EH['basic']['Ey'].cpu().numpy() - EH['solved']['Ey']) ** 2) +
                                        np.sum((EH['basic']['Ez'].cpu().numpy() - EH['solved']['Ez']) ** 2)
                                    ) / (np.sum(EH['solved']['Ex'] ** 2) + np.sum(EH['solved']['Ey'] ** 2) + np.sum(EH['solved']['Ez'] ** 2)) + 
                                    (
                                        np.sum((EH['basic']['Hx'].cpu().numpy() - EH['solved']['Hx']) ** 2) +
                                        np.sum((EH['basic']['Hy'].cpu().numpy() - EH['solved']['Hy']) ** 2) +
                                        np.sum((EH['basic']['Hz'].cpu().numpy() - EH['solved']['Hz']) ** 2)
                                    ) / (np.sum(EH['solved']['Hx'] ** 2) + np.sum(EH['solved']['Hy'] ** 2) + np.sum(EH['solved']['Hz'] ** 2))
                                )
                        )
                    )
                info['tt_errors'].append(
                    np.sqrt(
                            (
                                (
                                    np.sum((EH['TT']['Ex'].raw(torch.float64).cpu().numpy() - EH['solved']['Ex']) ** 2) +
                                    np.sum((EH['TT']['Ey'].raw(torch.float64).cpu().numpy() - EH['solved']['Ey']) ** 2) +
                                    np.sum((EH['TT']['Ez'].raw(torch.float64).cpu().numpy() - EH['solved']['Ez']) ** 2)
                                ) / (np.sum(EH['solved']['Ex'] ** 2) + np.sum(EH['solved']['Ey'] ** 2) + np.sum(EH['solved']['Ez'] ** 2)) + 
                                (
                                    np.sum((EH['TT']['Hx'].raw(torch.float64).cpu().numpy() - EH['solved']['Hx']) ** 2) +
                                    np.sum((EH['TT']['Hy'].raw(torch.float64).cpu().numpy() - EH['solved']['Hy']) ** 2) +
                                    np.sum((EH['TT']['Hz'].raw(torch.float64).cpu().numpy() - EH['solved']['Hz']) ** 2)
                                ) / (np.sum(EH['solved']['Hx'] ** 2) + np.sum(EH['solved']['Hy'] ** 2) + np.sum(EH['solved']['Hz'] ** 2))
                            )
                    )
                )
            info['tt_times'].append(tt_time)
            for key in TT.times.keys():
                info[f'{key}_times'].append(TT.times[key]) 
            if simulation_type == 2:
                info['basic_times'].append(basic_time)
                info['basic_sizes'].append(sum([EH['basic'][d].nbytes for d in order]))            
            if source:
                info['tt_sizes'].append(tt_size + P.nbytes())
            else:
                info['tt_sizes'].append(tt_size)
            for i in range(6):
                info['ranks'][0][i].append(lranks[i][0])
                info['ranks'][1][i].append(lranks[i][1])
            info['grids'].append(grid_size)
            with open(f"{ending}.json", "w") as f:
                json.dump(info, f)
            if not npy:
                if simulation_type == 2:
                    for d in order:
                        del EH['basic'][d]
                for d in order:
                    del EH['TT'][d]
                    del EH['solved'][d]
                if source:
                    del P
                    if solution == 4:
                        del X
        if not npy:
            torch.cuda.empty_cache()
        print(f"{grid_size}: Finished solving", flush = True)
        grid_size += 1