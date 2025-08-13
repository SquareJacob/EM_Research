from __future__ import annotations
import numpy as np
from typing import Union, Literal, Callable
import torch
import time
import traceback


class TT():
    roundInPlus = True
    firstFunc = lambda x, y: x
    class TTarray():   
        def __init__(self, A: Union[np.ndarray, list[np.ndarray], TT.TTarray], eps : float = 0, caps: list[int] = None):
            if isinstance(A, TT.TTarray):
                self.eps = A.eps
                self.caps = A.caps
                self.cores = []
                for a in A.cores:
                    self.cores.append(a.copy())
            else:
                self.eps = eps
                self.caps = caps
                if isinstance(A, list):
                    self.cores = []
                    for a in A:
                        self.cores.append(a.copy())
                else:
                    self.cores = TT.build(A, eps, caps)

        def __getitem__(self, i):
            return self.cores[i]
        
        def __setitem__(self, i, k):
            self.cores[i] = k
        
        def __len__(self):
            return len(self.cores)
        
        def __add__(self, other):
            if isinstance(other, TT.TTarray):
                if TT.roundInPlus:
                    T = TT.add(self, other)
                    return T.round()
                else:
                    return TT.add(self, other)
            else:
                raise TypeError("TTarray can only be added with another TTarray")
        
        def __iadd__(self, other):
            self = TT.add(self, other, epsFunc = TT.firstFunc, capsFunc= TT.firstFunc)
            if TT.roundInPlus:
                return self.round()
            return self
            
        def __mul__(self, other):
            if isinstance(other, (int, float)):
                copy = self.deepcopy()
                copy.cores[-1] *= other
                return copy
            else:
                raise TypeError("TTarray can only be multiplied with scalar number")
            
        def __rmul__(self, other):
            return self * float(other)
            
        def __imul__(self, other):
            self = self[0].dtype.type(other) * self
            return self
        
        def __sub__(self, other):
            return (-1 * other) + self
        
        def __isub__(self, other):
            self = self - other
            return self
        
        #https://github.com/oseledets/TT-Toolbox/blob/master/%40tt_tensor/full.m
        def raw(self, dtype = None) -> np.ndarray:
            dims = self.dims()
            ranks = self.ranks()
            if dtype is None:
                dtype = self[0].dtype
            A = self[0].astype(dtype)
            for i in range(1, len(dims)):
                A = A.reshape(-1, ranks[i]) @ self[i].reshape(ranks[i], -1).astype(dtype)         
            return A.reshape(dims)
        
        def norm(self, rank_split = None, dtype = None) -> Union[float, np.ndarray]:
            A = self
            if dtype is None:
                dtype = A[0].dtype
            if rank_split is None:
                rank_split = len(A) + 123
            if rank_split > 1:
                v = np.einsum('ijk,ljm->km', A[0].astype(dtype), A[0].astype(dtype))
            else:
                v = np.einsum('ijk,ijk->k', A[0].astype(dtype), A[0].astype(dtype))
            for i in range(1, len(A)):
                if i > rank_split:
                    v = np.einsum('inj,minl->mjl', A[i].astype(dtype), np.einsum('mik,knl->minl', v, A[i].astype(dtype)))
                elif i == rank_split:
                    v = np.einsum('inj,inl->ijl', A[i].astype(dtype), np.einsum('i,inl->inl', v, A[i].astype(dtype)))
                elif i == rank_split - 1:
                    v = np.einsum('inj,inj->j', A[i].astype(dtype), np.einsum('ik,knj->inj', v, A[i].astype(dtype)))
                else:
                    v = np.einsum('inj,iln->jl', A[i].astype(dtype), np.einsum('jk,knl->jln', v, A[i].astype(dtype)))
            v = np.sqrt(v)
            return v.item() if rank_split > i else v.reshape(-1)
        
        def size(self) -> int:
            total = 0
            for a in self:
                total += a.size
            return total
        
        def nbytes(self) -> int:
            total = 0
            for a in self:
                total += a.nbytes
            return total
        
        def ranks(self):
            return [1] + [g.shape[2] for g in self.cores]
        
        def round(self, eps: float = None, caps: list[int] = None) -> TT.TTarray:
            return TT.round(self, self.eps if eps is None else eps, self.caps if caps is None else caps)
        
        def dot(self, B: TT.TTarray) -> float:
            return TT.dot(self, B)
        
        def roll(self, shifts: np.ndarray[int]) -> TT.TTarray:
            return TT.roll(self, shifts)
        
        def dims(self):
            return [g.shape[1] for g in self.cores]
        
        def coreDims(self):
            return [g.shape for g in self.cores]
        
        def rollsum(self, shifts: np.ndarray[int], use_orig: bool = True, padding: bool = False) -> TT.TTarray:
            return TT.rollsum(self, shifts, use_orig, padding)
        
        def zeros(self) -> TT.TTarray:
            return TT.TTarray([np.zeros(a.shape, dtype = a.dtype) for a in self.cores], self.eps, self.eps, self.caps.copy() if self.caps else None)
        
        def zero(self, dims: list[int], indices: list[Union[list[int], int, tuple[int]]]) -> TT.TTarray:
            return TT.zero(self, dims, indices)
        
        def reduce(self, indices: list[Union[list[int], Literal[':'], int, tuple[int]]]) -> TT.TTarray:
            return TT.reduce(self, indices)
        
        def reducedSum(self, dim: int, indices: list[Union[list[int], Literal[':'], int, tuple[int]]], scalars: list[int]) -> TT.TTarray:
            return TT.reducedSum(self, dim, indices, scalars)
        
        def pad(self, dims: list[int], indices: list[list[int]]) -> TT.TTarray:
            return TT.pad(self, dims, indices)
        
        def deepcopy(self) -> TT.TTarray:
            x = TT.TTarray(self)
            A = x.cores
            x.cores  = []
            for a in A:
                x.cores.append(a.copy())
            if x.caps:
                x.caps = x.caps.copy()
            return x

    #https://users.math.msu.edu/users/iwenmark/Teaching/CMSE890/TENSOR_oseledets2011.pdf
    @staticmethod
    def build(A: np.ndarray, eps: float, caps: list[int] = None) -> list[np.ndarray]:
        dims = A.shape
        delta = eps / np.sqrt(len(dims) - 1) * np.linalg.norm(A)
        G = []
        r = [1] * (len(dims) + 1)
        C = A
        for i in range(len(dims) - 1):
            C = C.reshape(r[i] * dims[i], -1)
            U, S, V = np.linalg.svd(C, full_matrices= False)
            L = np.zeros_like(C)
            for j in range(len(S)):
                L += S[j] * np.dot(U[:, j:j + 1], V[j:j + 1, :])
                r[i + 1] = j + 1
                if np.linalg.norm(L - C, 'fro') <= delta:
                    break
            if caps:
                r[i + 1] = min(r[i + 1], caps[i])
            G.append(U[:, 0:r[i + 1]].reshape(r[i], dims[i], r[i + 1]))
            C = S[0:r[i + 1]][:, None] * V[0:r[i + 1], :]
        G.append((S[0:r[-2]][:, None] * V[0:r[-2], :]).reshape(r[-2], dims[-1], r[-1]))
        return G

   #https://users.math.msu.edu/users/iwenmark/Teaching/CMSE890/TENSOR_oseledets2011.pdf     
    @staticmethod
    def dot(A: TT, B: TT, dtype = None) -> float:
        if dtype is None:
            dtype = A[0].dtype
        v = np.einsum('ijk,ljm->ilkm', A[0].astype(dtype), B[0].astype(dtype)).reshape(-1, A[0].shape[2], B[0].shape[2])
        for i in range(1, len(A)):
            v = np.einsum('inj,kiln->kjl', A[i].astype(dtype), np.einsum('ijk,knl->ijln', v, B[i].astype(dtype)))
        return v.item()
  
    roundTime = 0
    svdTime = 0
    qrTime = 0
    part_time = 0
    oldRound = False
    @staticmethod
    def round(A: TTarray, eps: float, caps: list[int] = None) -> TTarray:
        if TT.oldRound:
            return TT.round1(A, eps, caps)
        TT.roundTime -= time.time()
        delta = eps / np.sqrt(len(A) - 1)
        r = [1]
        n = []
        B = []
        for g in A:
            s = g.shape
            r.append(s[2])
            n.append(s[1])
            B.append(g)
        rmax = np.prod(n).item()
        for i in range(len(A) - 1, 0, -1):
            TT.qrTime -= time.time()
            B[i], R = np.linalg.qr(B[i].reshape(r[i], n[i] * r[i + 1]).T, 'reduced')
            TT.qrTime += time.time()
            B[i] = B[i].T
            B[i - 1] = A[i - 1].reshape(r[i - 1] * n[i - 1], r[i]) @ R.T
            r[i] = R.shape[0]
        for i in range(len(A) - 1):
            TT.qrTime -= time.time()
            B[i], R = np.linalg.qr(B[i].reshape(r[i] * n[i], r[i + 1]), 'reduced')
            TT.qrTime += time.time()
            TT.svdTime -= time.time()
            try:
                U, S, V = np.linalg.svd(R, full_matrices = False)
            except Exception as e:
                print(e, flush = True)
                traceback.print_exc()
                print(R, flush = True)
                TT.svdTime += time.time()
                TT.roundTime += time.time()
                return A
            TT.svdTime += time.time()
            #https://github.com/oseledets/TT-Toolbox/blob/master/core/my_chop2.m
            s = np.linalg.norm(S)
            if s == 0:
                r1 = 1
            elif eps == 0:
                r1 = len(S)
            else:
                ep = delta * s
                C = np.cumsum(S[::-1] ** 2)
                ff = np.nonzero(C  < ep * ep)[0]
                if len(ff):
                    r1 = len(C) - ff[-1] - 1
                else:
                    r1 = len(C)
            r1 = min(r1, rmax)
            if caps:
                r1 =  min(r1, caps[i])
            B[i] = (B[i] @ U[:, :r1]).reshape(r[i], n[i], r1)
            B[i + 1] = (S[:r1, None] * V[:r1, :]) @ B[i + 1].reshape(r[i + 1], n[i + 1] * r[i + 2])
            r[i + 1] = r1
        B[-1] = B[-1].reshape(r[-2], n[-1], r[-1])
        TT.roundTime += time.time()
        return TT.TTarray(B, A.eps, A.caps)

    #https://github.com/oseledets/TT-Toolbox/blob/master/%40tt_tensor/round.m 
    @staticmethod
    def round1(A: TTarray, eps: float, caps: list[int] = None) -> TTarray:
        TT.roundTime -= time.time()
        delta = eps / np.sqrt(len(A) - 1)
        r = [1]
        n = []
        B = []
        for g in A:
            s = g.shape
            r.append(s[2])
            n.append(s[1])
            B.append(g)
        rmax = np.prod(n).item()
        nrm = np.zeros((len(A), 1), dtype = A[0].dtype)
        c1 = A[0]
        for i in range(len(A) - 1):
            TT.qrTime -= time.time()
            c0, R = np.linalg.qr(c1.reshape(r[i] * n[i], r[i + 1]), 'reduced')
            TT.qrTime += time.time()
            nrm[i + 1] = np.linalg.norm(R, "fro")
            if nrm[i + 1] != 0:
                R /= nrm[i + 1]
            c1 = np.dot(R, A[i + 1].reshape(r[i + 1], n[i + 1] * r[i + 2]))
            r[i + 1] = c0.shape[1]
            B[i] =  c0.copy()
            B[i + 1] = c1
        B[-1] = B[-1].reshape(r[-2], n[-1], r[-1])
        c1 = B[-1]
        for i in range(len(A) - 1, 0, -1):
            TT.svdTime -= time.time()
            try:
                U, S, V = np.linalg.svd(c1.reshape(r[i], n[i] * r[i + 1]), full_matrices = False)
            except Exception as e:
                print(e, flush = True)
                traceback.print_exc()
                print(c1.reshape(r[i], n[i] * r[i + 1]), flush = True)
                TT.svdTime += time.time()
                TT.roundTime += time.time()
                return A
            TT.svdTime += time.time()
			#https://github.com/oseledets/TT-Toolbox/blob/master/core/my_chop2.m
            s = np.linalg.norm(S)
            if s == 0:
                r1 = 1
            elif eps == 0:
                r1 = len(S)
            else:
                ep = delta * s
                C = np.cumsum(S[::-1] ** 2)
                ff = np.nonzero(C  < ep * ep)[0]
                if len(ff):
                    r1 = len(C) - ff[-1] - 1
                else:
                    r1 = len(C)
            r1 = min(r1, rmax)
            if caps:
                r1 =  min(r1, caps[i - 1])
            r[i] = r1
            c1 = np.dot(B[i - 1].reshape(r[i - 1] * n[i - 1], -1), U[:, :r1] * S[:r1][None, :])
            B[i] = V[:r1].reshape(r[i], n[i], r[i + 1])
        B[0] = c1
        nrm[0] = np.linalg.norm(B[0], 'fro')
        if nrm[0] != 0:
            B[0] /= nrm[0]
        B[0] = B[0].reshape(r[0], n[0], r[1])
        nrm0 = np.sum(np.log(np.abs(nrm)))
        nrm0 = np.exp(nrm0 / len(A)).item()
        for i in range(len(A)):
            B[i] *= nrm0
        TT.roundTime += time.time()
        return TT.TTarray(B, A.eps, A.caps)

    addTime = 0
    addEpsFunc = min
    addCapsFunc = lambda x, y: [max(x[i], y[i]) for i in range(len(x))] if x and y else None
    #https://users.math.msu.edu/users/iwenmark/Teaching/CMSE890/TENSOR_oseledets2011.pdf
    @staticmethod
    def add(A: TTarray, B: TTarray, epsFunc: Callable[[float, float], float] = None, capsFunc: Callable[[int, int], int] = None) -> TTarray:
        TT.addTime -= time.time()
        if not epsFunc:
            epsFunc = TT.addEpsFunc
        if not capsFunc:
            capsFunc = TT.addCapsFunc
        C = [np.concatenate((A[0], B[0]), 2)]
        for i in range(1, len(A) - 1):
            a, b = A[i], B[i]
            c = np.zeros((a.shape[0] + b.shape[0], a.shape[1], a.shape[2] + b.shape[2]), dtype = a.dtype)
            c[:a.shape[0], :, :a.shape[2]] = a
            c[a.shape[0]:, :, a.shape[2]:] = b
            C.append(c)
        C.append(np.concatenate((A[-1], B[-1]), 0))
        TT.addTime += time.time()
        return TT.TTarray(C, epsFunc(A.eps, B.eps), capsFunc(A.caps, B.caps))
    
    @staticmethod    
    def roll(A: TTarray, shifts: np.ndarray[int, int]) -> TTarray:
        B = A.deepcopy()
        for shift in shifts:
            B[shift[1]] = np.roll(B[shift[1]], shift[0], 1)
        return B
    
    @staticmethod
    def rollsum(A: TTarray, shifts: np.ndarray[int, int], use_orig: bool = True, padding: bool = False) -> TTarray:
        if use_orig:
            B = A.deepcopy()
            for shift in shifts:
                if padding:
                   if shift[0] < 0:
                       B[shift[1]][:, :shift[0], :] += shift[2] * B[shift[1]][:, -shift[0]:, :]
                   else:
                       B[shift[1]][:, shift[0]:, :] += shift[2] * B[shift[1]][:, :-shift[0], :]
                else:
                    B[shift[1]] += shift[2] * np.roll(B[shift[1]], shift[0], 1)

            return B
        else:
            raise RuntimeError("rollsum without original not implemented :(")
    
    @staticmethod
    def zeros(ranks: list[int], dims: list[int], eps: float, caps = None, dtype = np.float32) -> TTarray:
        return TT.TTarray([np.zeros((ranks[i], dims[i], ranks[i + 1]), dtype = dtype) for i in range(len(dims))], eps, caps)
    
    @staticmethod
    def zero(A: TTarray, dims: list[int], indices: list[Union[list[int], int, tuple[int]]]) -> TTarray:
        B = A.deepcopy()
        for i, index in enumerate(indices):
            if isinstance(index, list) or isinstance(index, int):
                B[dims[i]][:, index, :] = 0
            elif isinstance(index, tuple):
                B[dims[i]][:, index[0]:index[1], :] = 0
        return B
    
    @staticmethod
    def reduce(A: TTarray, indices: list[Union[list[int], Literal[':'], int, tuple[int]]]) -> TTarray:
        B = A.deepcopy()
        for i, index in enumerate(indices):
            if isinstance(index, list):
                B[i] = B[i][:, index, :]
            elif isinstance(index, int):
                B[i] = B[i][:, index:index+1, :]
            elif isinstance(index, tuple):
                B[i] = B[i][:, index[0]:index[1], :]
        return B
    
    @staticmethod
    def reducedSum(A: TTarray, dim: int, indices: list[Union[list[int], Literal[':'], int, tuple[int]]], scalars: list[int]) -> TTarray:
        B = A.deepcopy()
        for i, index in enumerate(indices):
            if i:
                if isinstance(index, list):
                    B[dim] += scalars[i] * A[dim][:, index, :]
                elif isinstance(index, int):
                    B[dim] += scalars[i] * A[dim][:, index:index+1, :]
                elif isinstance(index, tuple):
                    B[dim] += scalars[i] * A[dim][:, index[0]:index[1], :]
                elif index == ':':
                    B[dim] += scalars[i] * A[dim]
            else:
                if isinstance(index, list):
                    B[dim] = scalars[0] * B[dim][:, index, :]
                elif isinstance(index, int):
                    B[dim] = scalars[0] * B[dim][:, index:index+1, :]
                elif isinstance(index, tuple):
                    B[dim] = scalars[0] * B[dim][:, index[0]:index[1], :]
                else:
                    B[dim] *= scalars[0]
        return B
    
    @staticmethod
    def pad(A: TTarray, dims: list[int], indices: list[list[int]]) -> TTarray:
        B = A.deepcopy()
        for i, index in enumerate(indices):
            j = dims[i]
            s = A[j].shape
            B[j] = np.zeros((s[0], s[1] + index[0], s[2]), dtype = A[j].dtype)
            B[j][:, index[1]:index[1]+s[1], :] = A[j]
        return B


class torchTT():
    roundInPlus = True
    preallocated = False
    firstFunc = lambda x, y: x
    class TTarray():   
        def __init__(self, A: Union[torch.Tensor, list[torch.Tensor], torchTT.TTarray], eps: float = 0, caps: list[int] = None):
            if isinstance(A, torchTT.TTarray):
                self.eps = A.eps
                self.caps = A.caps
                self.cores = A.cores
            else:
                self.eps = eps
                self.caps = caps
                if isinstance(A, list):
                    self.cores = []
                    for a in A:
                        self.cores.append(a.clone())
                else:
                    self.cores = torchTT.build(A, eps, caps)

        def __getitem__(self, i):
            return self.cores[i]
        
        def __setitem__(self, i, k):
            self.cores[i] = k
        
        def __len__(self):
            return len(self.cores)
        
        def __add__(self, other):
            if isinstance(other, torchTT.TTarray):
                if torchTT.roundInPlus:
                    T = torchTT.add(self, other)
                    x = T.round()
                    del T
                    return x
                else:
                    return torchTT.add(self, other)
            else:
                raise TypeError("TTarray can only be added with another TTarray")
        
        def __iadd__(self, other):
            x = torchTT.add(self, other, epsFunc = torchTT.firstFunc, capsFunc = torchTT.firstFunc)
            if torchTT.roundInPlus:
                y = x.round()
                del x
                x = y
            del self
            self = x
            return self
            
        def __mul__(self, other):
            if isinstance(other, (int, float)):
                copy = self.deepcopy()
                copy.cores[-1] *= other
                return copy
            else:
                raise TypeError("TTarray can only be multiplied with scalar number")
            
        def __rmul__(self, other):
            return self * other
            
        def __imul__(self, other):
            x = other * self
            del self
            self = x
            return self
        
        def __sub__(self, other):
            return (-1 * other) + self
        
        def __isub__(self, other):
            x = self - other
            del self
            self = x
            return self
            
        def __del__(self):
            if hasattr(self, 'cores'):
                for a in self:
                    del a
        
        #https://github.com/oseledets/TT-Toolbox/blob/master/%40tt_tensor/full.m
        def raw(self) -> torch.Tensor:
            dims = self.dims()
            ranks = self.ranks()
            A = self[0]
            for i in range(1, len(dims)):
                A = A.reshape(-1, ranks[i]) @ self[i].reshape(ranks[i], -1)          
            return A.reshape(dims)
        
        def norm(self) -> float:
            return np.sqrt(torchTT.dot(self, self))
        
        def size(self) -> int:
            total = 0
            for a in self:
                total += a.numel()
            return total
            
        def nbytes(self) -> int:
            total = 0
            for a in self:
                total += a.nbytes
            return total
        
        def ranks(self):
            return [1] + [g.shape[2] for g in self.cores]
        
        def round(self, eps: float = None, caps: list[int] = None) -> torchTT.TTarray:
            return torchTT.round(self, self.eps if eps is None else eps, self.caps if caps is None else caps)
        
        def dot(self, B: torchTT.TTarray) -> float:
            return torchTT.dot(self, B)
        
        def roll(self, shifts: torch.Tensor[int]) -> torchTT.TTarray:
            return torchTT.roll(self, shifts)
        
        def dims(self):
            return [g.shape[1] for g in self.cores]
        
        def coreDims(self):
            return [g.shape for g in self.cores]
        
        def rollsum(self, shifts: torch.Tensor[int], use_orig: bool = True, padding: bool = False) -> torchTT.TTarray:
            return torchTT.rollsum(self, shifts, use_orig, padding)
        
        def zeros(self) -> torchTT.TTarray:
            return torchTT.TTarray([torch.zeros(a.shape, dtype = a.dtype, device = a.device) for a in self.cores], self.eps, self.caps.copy() if self.caps else None)
            
        def zero(self, dims: list[int], indices: list[Union[list[int], int, tuple[int]]]) -> torchTT.TTarray:
            return torchTT.zero(self, dims, indices)
        
        def reduce(self, indices: list[Union[list[int], Literal[':'], int, tuple[int]]]) -> torchTT.TTarray:
            return torchTT.reduce(self, indices)
            
        def reducedSum(self, dim: int, indices: list[Union[list[int], Literal[':'], int, tuple[int]]], scalars: list[int]) -> torchTT.TTarray:
            return torchTT.reducedSum(self, dim, indices, scalars)
        
        def pad(self, dims: list[int], indices: list[list[int]]) -> torchTT.TTarray:
            return torchTT.pad(self, dims, indices)
        
        def deepcopy(self) -> torchTT.TTarray:
            x = torchTT.TTarray(self)
            A = x.cores
            x.cores  = []
            for a in A:
                x.cores.append(a.clone())
            if x.caps:
                x.caps = x.caps.copy()
            return x

    #https://users.math.msu.edu/users/iwenmark/Teaching/CMSE890/TENSOR_oseledets2011.pdf
    @staticmethod
    def build(A: torch.Tensor, eps: float, caps: list[int] = None) -> list[torch.Tensor]:
        dims = A.shape
        delta = eps / np.sqrt(len(dims) - 1) * torch.linalg.norm(A)
        G = []
        r = [1] * (len(dims) + 1)
        C = A
        for i in range(len(dims) - 1):
            C = C.reshape(r[i] * dims[i], -1)
            if i:
                del U, S, V
            U, S, V = torch.linalg.svd(C, full_matrices= False)
            L = torch.zeros_like(C)
            for j in range(len(S)):
                L += S[j] * torch.matmul(U[:, j:j + 1], V[j:j + 1, :])
                r[i + 1] = j + 1
                if torch.linalg.norm(L - C, 'fro') <= delta:
                    del L
                    break
            if caps:
                r[i + 1] = min(r[i + 1], caps[i])
            G.append(U[:, 0:r[i + 1]].reshape(r[i], dims[i], r[i + 1]))
            #G.append(torch.reshape(U[:, 0:r[i + 1]], (r[i], dims[i], r[i + 1])))
            C = S[0:r[i + 1]][:, None] * V[0:r[i + 1], :]
            #C = torch.matmul(torch.diag(S[0:r[i + 1]]), V[0:r[i + 1], :])
        #G.append(torch.reshape(torch.matmul(torch.diag(S[0:r[-2]]), V[0:r[-2], :]), (r[-2], dims[-1], r[-1])))
        G.append((S[0:r[-2]][:, None] * V[0:r[-2], :]).reshape(r[-2], dims[-1], r[-1]))
        del U, S, V
        return G

   #https://users.math.msu.edu/users/iwenmark/Teaching/CMSE890/TENSOR_oseledets2011.pdf     
    @staticmethod
    def dot(A: TTarray, B: TTarray) -> float:
        v = torch.einsum('ijk,ljm->ilkm', A[0], B[0]).reshape(-1, A[0].shape[2], B[0].shape[2])
        for i in range(1, len(A)):
            v = torch.einsum('inj,kiln->kjl', A[i], torch.einsum('ijk,knl->ijln', v, B[i]))
        return v.item()
    
    roundTime = 0
    svdTime = 0
    qrTime = 0
    part_time = 0
    oldRound = False
    events = {'Rounding': [], 'qr': [], 'svd': [], 'Part': [], 'Addition': []}
    @staticmethod
    def round(A: TTarray, eps: float, caps: list[int] = None) -> TTarray:
        if torchTT.oldRound:
            return torchTT.round1(A, eps, caps)
        gpu = A[0].is_cuda
        if gpu:
            e = torch.cuda.Event(enable_timing = True)
            e.record()
            torchTT.events['Rounding'].append(e)
        else:
            torchTT.roundTime -= time.time()
        delta = eps / np.sqrt(len(A) - 1)
        r = [1]
        n = []
        B = []
        for g in A:
            s = g.shape
            r.append(s[2])
            n.append(s[1])
            B.append(g)
        rmax = np.prod(n).item()
        for i in range(len(A) - 1, 0, -1):
            if gpu:
                e = torch.cuda.Event(enable_timing = True)
                e.record()
                torchTT.events['qr'].append(e)
            else:
                torchTT.qrTime -= time.time()
            B[i], R = torch.linalg.qr(B[i].reshape(r[i], n[i] * r[i + 1]).t(), 'reduced')
            if gpu:
                e = torch.cuda.Event(enable_timing = True)
                e.record()
                torchTT.events['qr'].append(e)
            else:
                torchTT.qrTime += time.time()
            B[i] = B[i].t()
            B[i - 1] = A[i - 1].reshape(r[i - 1] * n[i - 1], r[i]) @ R.t()
            r[i] = R.shape[0]
        for i in range(len(A) - 1):
            if gpu:
                e = torch.cuda.Event(enable_timing = True)
                e.record()
                torchTT.events['qr'].append(e)
            else:
                torchTT.qrTime -= time.time()
            B[i], R = torch.linalg.qr(B[i].reshape(r[i] * n[i], r[i + 1]), 'reduced')
            if gpu:
                e = torch.cuda.Event(enable_timing = True)
                e.record()
                torchTT.events['qr'].append(e)
                e = torch.cuda.Event(enable_timing = True)
                e.record()
                torchTT.events['svd'].append(e)
            else:
                torchTT.qrTime += time.time()
                torchTT.svdTime -= time.time()
            U, S, V = torch.linalg.svd(R, full_matrices = False)
            if gpu:
                e = torch.cuda.Event(enable_timing = True)
                e.record()
                torchTT.events['svd'].append(e)
            else:
                torchTT.svdTime += time.time()
            #https://github.com/oseledets/TT-Toolbox/blob/master/core/my_chop2.m
            s = torch.linalg.norm(S)
            if s == 0:
                r1 = 1
            elif eps == 0:
                r1 = len(S)
            else:
                ep = delta * s
                C = torch.cumsum(S.square().flip(0), dim=0)
                r1 = C.numel() - torch.searchsorted(C, ep * ep, right=False)
                del C
            r1 = min(r1, rmax)
            if caps:
                r1 =  min(r1, caps[i])
            B[i] = (B[i] @ U[:, :r1]).reshape(r[i], n[i], r1)
            B[i + 1] = (S[:r1, None] * V[:r1, :]) @ B[i + 1].reshape(r[i + 1], n[i + 1] * r[i + 2])
            r[i + 1] = r1
        B[-1] = B[-1].reshape(r[-2], n[-1], r[-1])
        if gpu:
            e = torch.cuda.Event(enable_timing = True)
            e.record()
            torchTT.events['Rounding'].append(e)
        else:
            torchTT.roundTime += time.time()
        return torchTT.TTarray(B, A.eps, A.caps)
    
    #https://github.com/oseledets/TT-Toolbox/blob/master/%40tt_tensor/round.m
    @staticmethod
    def round1(A: TTarray, eps: float, caps: list[int] = None) -> TTarray:
        pre = torchTT.preallocated
        gpu = A[0].is_cuda
        if gpu:
            e = torch.cuda.Event(enable_timing = True)
            e.record()
            torchTT.events['Rounding'].append(e)
        else:
            torchTT.roundTime -= time.time()
        delta = eps / np.sqrt(len(A) - 1)
        r = [1]
        n = []
        B = []
        for g in A:
            s = g.shape
            r.append(s[2])
            n.append(s[1])
            B.append(g)
        rmax = np.prod(n)
        nrm = torch.zeros((len(A), 1), device = A[0].device, dtype = A[0].dtype)
        c1 = A[0]
        for i in range(len(A) - 1):
            if gpu:
                e = torch.cuda.Event(enable_timing = True)
                e.record()
                torchTT.events['qr'].append(e)
            else:
                torchTT.qrTime -= time.time()
            if torch.is_tensor(pre):
                c1 = c1.reshape(r[i] * n[i], r[i + 1])
                m1, m2 = c1.shape
                torch.linalg.qr(c1, 'reduced', out = (pre[:m1 * min(m1, m2)].view(m1, min(m1, m2)), pre[m1 * min(m1, m2):min(m1, m2) * (m2 + m1)].view(min(m1, m2), m2)))
                c0, R = (pre[:m1 * min(m1, m2)].view(m1, min(m1, m2)), pre[m1 * min(m1, m2):min(m1, m2) * (m2 + m1)].view(min(m1, m2), m2))
            else:
                c0, R = torch.linalg.qr(c1.reshape(r[i] * n[i], r[i + 1]), 'reduced')
            if gpu:
                e = torch.cuda.Event(enable_timing = True)
                e.record()
                torchTT.events['qr'].append(e)
            else:
                torchTT.qrTime += time.time()
            nrm[i + 1] = torch.linalg.norm(R, "fro")
            if nrm[i + 1] != 0:
                R /= nrm[i + 1].item()
            del c1
            c1 = torch.matmul(R, A[i + 1].reshape(r[i + 1], n[i + 1] * r[i + 2]))
            r[i + 1] = c0.shape[1]
            B[i] = c0.clone()
            B[i + 1] = c1
            if not torch.is_tensor(pre):
               del c0, R 
        B[-1] = B[-1].reshape(r[-2], n[-1], r[-1])
        c1 = B[-1]
        for i in range(len(A) - 1, 0, -1):
            if gpu:
                e = torch.cuda.Event(enable_timing = True)
                e.record()
                torchTT.events['svd'].append(e)
            else:
                torchTT.svdTime -= time.time()
            if torch.is_tensor(pre):
                c1 = c1.reshape(r[i], n[i] * r[i + 1])
                m1, m2 = c1.shape
                pre[:] = 0
                torch.linalg.svd(c1, full_matrices = False, out = (pre[:m1 * min(m1, m2)].view(m1, min(m1, m2)), pre[m1 * min(m1, m2):(m1 + 1) * min(m1, m2)], pre[(m1 + 1) * min(m1, m2):(m1 + 1 + m2) * min(m1, m2)].view(min(m1, m2), m2)))
                U, S, V = (pre[:m1 * min(m1, m2)].view(m1, min(m1, m2)), pre[m1 * min(m1, m2):(m1 + 1) * min(m1, m2)], pre[(m1 + 1) * min(m1, m2):(m1 + 1 + m2) * min(m1, m2)].view(min(m1, m2), m2))
            else:
                U, S, V = torch.linalg.svd(c1.reshape(r[i], n[i] * r[i + 1]), full_matrices = False)
            if gpu:
                e = torch.cuda.Event(enable_timing = True)
                e.record()
                torchTT.events['svd'].append(e)
            else:
                torchTT.svdTime += time.time()
            del c1
            #https://github.com/oseledets/TT-Toolbox/blob/master/core/my_chop2.m
            s = torch.linalg.norm(S)
            if s == 0:
                r1 = 1
            elif eps == 0:
                r1 = len(S)
            else:
                ep = delta * s
                C = torch.cumsum(S.square().flip(0), dim=0)
                r1 = C.numel() - torch.searchsorted(C, ep * ep, right=False)
                del C
            r1 = min(r1, rmax)
            if caps:
                r1 =  min(r1, caps[i - 1])
            r[i] = r1
            c1 = torch.matmul(B[i - 1].reshape(r[i - 1] * n[i - 1], -1), U[:, :r1] * S[:r1][None, :])
            #c1 = torch.matmul(B[i - 1].reshape(r[i - 1] * n[i - 1], -1), torch.matmul(U[:, :r1], torch.diag(S[:r1])))
            if torch.is_tensor(pre):
                B[i] = V[:r1, :].reshape(r[i], n[i], r[i + 1]).clone()
            else:
                B[i] = V[:r1, :].reshape(r[i], n[i], r[i + 1])
                del U, S, V
        B[0] = c1
        nrm[0] = torch.linalg.norm(B[0], 'fro')
        if nrm[0] != 0:
            B[0] /= nrm[0]
        B[0] = B[0].reshape(r[0], n[0], r[1])
        nrm0 = torch.sum(torch.log(torch.abs(nrm)))
        del nrm
        nrm0 = torch.exp(nrm0 / len(A))
        for i in range(len(A)):
            B[i] *= nrm0
        if gpu:
            e = torch.cuda.Event(enable_timing = True)
            e.record()
            torchTT.events['Rounding'].append(e)
        else:
            torchTT.roundTime += time.time()
        return torchTT.TTarray(B, A.eps, A.caps)

    addTime = 0
    addEpsFunc = min
    addCapsFunc = lambda x, y: [max(x[i], y[i]) for i in range(len(x))] if x and y else None
    #https://users.math.msu.edu/users/iwenmark/Teaching/CMSE890/TENSOR_oseledets2011.pdf
    @staticmethod
    def add(A: TTarray, B: TTarray, epsFunc: Callable[[float, float], float] = None, capsFunc: Callable[[int, int], int] = None) -> TTarray:
        if A[0].is_cuda:
            e = torch.cuda.Event(enable_timing = True)
            e.record()
            torchTT.events['Addition'].append(e)
        else:
            torchTT.addTime -= time.time()
        if not epsFunc:
            epsFunc = torchTT.addEpsFunc
        if not capsFunc:
            capsFunc = torchTT.addCapsFunc
        C = [torch.concatenate((A[0], B[0]), 2)]
        for i in range(1, len(A) - 1):
            a, b = A[i], B[i]
            c = torch.zeros((a.shape[0] + b.shape[0], a.shape[1], a.shape[2] + b.shape[2]), dtype = a.dtype, device = a.device)
            c[:a.shape[0], :, :a.shape[2]] = a
            c[a.shape[0]:, :, a.shape[2]:] = b
            C.append(c)
        C.append(torch.concatenate((A[-1], B[-1]), 0))
        if A[0].is_cuda:
            e = torch.cuda.Event(enable_timing = True)
            e.record()
            torchTT.events['Addition'].append(e)
        else:
            torchTT.addTime += time.time()
        return torchTT.TTarray(C, epsFunc(A.eps, B.eps), capsFunc(A.caps, B.caps))
    
    @staticmethod    
    def roll(A: TTarray, shifts: torch.Tensor[int, int]) -> TTarray:
        B = A.deepcopy()
        for shift in shifts:
            B[shift[1]] = torch.roll(B[shift[1]], shift[0], 1)
        return B
    
    @staticmethod
    def rollsum(A: TTarray, shifts: torch.Tensor[int, int], use_orig: bool = True, padding: bool = False) -> TTarray:
        if use_orig:
            B = A.deepcopy()
            for shift in shifts:
                if padding:
                   if shift[0] < 0:
                       B[shift[1]][:, :shift[0], :] += shift[2] * B[shift[1]][:, -shift[0]:, :]
                   else:
                       B[shift[1]][:, shift[0]:, :] += shift[2] * B[shift[1]][:, :-shift[0], :]
                else:
                    B[shift[1]] += shift[2] * torch.roll(B[shift[1]], shift[0], 1)

            return B
        else:
            raise RuntimeError("rollsum without original not implemented :(")
    
    @staticmethod
    def zeros(ranks: list[int], dims: list[int], eps: float, caps = None, dtype = torch.float32, device = 'cpu') -> TTarray:
        return torchTT.TTarray([torch.zeros((ranks[i], dims[i], ranks[i + 1]), dtype = dtype, device = device) for i in range(len(dims))], eps, caps)

    @staticmethod
    #https://github.com/oseledets/TT-Toolbox/blob/master/core/ten_conv.m
    def ten_conv(C: torch.Tensor, k: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        n1, n2, n3 = C.shape
        if k == 1:
            C = C.reshape(n1, n2 * n3)
            C = M.T @ C
            return C.reshape(-1, n2, n3)
        elif k == 2:
            C = torch.einsum('ijk->ikj', C).reshape(n1 * n3, n2) @ M
            return torch.einsum('ijk->ikj', C.reshape(n1, n3, -1))
        elif k == 3:
            C = C.reshape(n1 * n2, n3) @ M
            return C.reshape(n1, n2, -1)
    
    alsTime = 0
    alsQRTime = 0
    @staticmethod
    #https://github.com/oseledets/TT-Toolbox/blob/master/core/tt_als.m
    def als(A: TTarray, B: TTarray, sweeps: int) -> TTarray:
        """
        A is goal, B is starting point
        """
        torchTT.alsTime -= time.time()
        if np.abs(B.norm()) < 1e-7:
            B = torchTT.TTarray(torch.rand(*B.dims(), dtype = B[0].dtype, device =  B[0].device), B.eps)
        else:
            B = torchTT.TTarray(B.cores, B.eps)
        phi = [0] * len(A)
        torchTT.alsQRTime -= time.time()
        q, rv = torch.linalg.qr(B[0][0, :, :])
        torchTT.alsQRTime += time.time()
        B[0] = q
        phi[0] = q.T @ A[0][0, :, :]
        for i in range(1, len(A) - 1):
            B[i] = torchTT.ten_conv(torch.einsum('ijk->jik', B[i]), 2, rv.T)
            ncur, r2, r3 = B[i].shape
            core1 = B[i].reshape(ncur * r2, r3)
            torchTT.alsQRTime -= time.time()
            B[i], rv = torch.linalg.qr(core1)
            torchTT.alsQRTime += time.time()
            rnew = min(r3, ncur * r2)
            B[i] = B[i].reshape(ncur, r2, rnew)
            ra1, n, ra2 = A[i].shape
            _, rb1, rb2 = B[i].shape
            phi[i] = phi[i - 1] @ A[i].reshape(ra1, n * ra2)
            phi[i] = torch.einsum('ijk->jik', phi[i].reshape(rb1, n, ra2))
            phi[i] = B[i].reshape(n * rb1, rb2).T @ phi[i].reshape(n * rb1, ra2)
        for _ in range(sweeps):
            B[-1] = A[-1][:, :, 0].T @ phi[-2].T
            torchTT.alsQRTime -= time.time()
            q, rv = torch.linalg.qr(B[-1])
            torchTT.alsQRTime += time.time()
            B[-1] = q
            phi[-1] = q.T @ A[-1][:, :, 0].T
            for i in range(len(A) - 2, 0, -1):
                phi0 = torchTT.ten_conv(torch.einsum('ijk->jik', A[i]), 3, phi[i + 1].T)
                B[i] = torchTT.ten_conv(phi0, 2, phi[i - 1].T)
                ncur, r2, r3 = B[i].shape
                core1 = torch.einsum('ijk->ikj', B[i]).reshape(ncur * r3, r2)
                torchTT.alsQRTime -= time.time()
                B[i], rv = torch.linalg.qr(core1)
                torchTT.alsQRTime += time.time()
                rnew = min(r2, ncur * r3)
                B[i] = torch.einsum('ijk->ikj', B[i].reshape(ncur, r3, rnew))
                ra1, ncur, _ = A[i].shape
                rb1 = B[i].shape[1]
                rb2 = phi[i + 1].shape[0]
                phi0 = torch.einsum('ijk->ikj', phi0.reshape(ncur, ra1, rb2))
                phi[i] = torch.einsum('ijk->ikj', B[i]).reshape(ncur * rb2, rb1).T @ phi0.reshape(ncur * rb2, ra1)
            B[0] = A[0][0, :, :] @ phi[1].T
            torchTT.alsQRTime -= time.time()
            q, rv = torch.linalg.qr(B[0])
            torchTT.alsQRTime += time.time()
            B[0] = q
            phi[0] = q.T @ A[0][0, :, :]
            for i in range(1, len(A) - 1):
                B[i] = torchTT.ten_conv(torch.einsum('ijk->jik', A[i]), 3, phi[i + 1].T)
                B[i] = torchTT.ten_conv(B[i], 2, phi[i - 1].T)
                ncur, r2, r3 = B[i].shape
                core1 = B[i].reshape(ncur * r2, r3)
                B[i], rv = torch.linalg.qr(core1)
                rnew = min(r3, ncur * r2)
                B[i] = B[i].reshape(ncur, r2, rnew)
                ra1, n, ra2 = A[i].shape
                _, rb1, rb2 = B[i].shape
                phi[i] = phi[i - 1] @ A[i].reshape(ra1, n * ra2)
                phi[i] = torch.einsum('ijk->jik', phi[i].reshape(rb1, n, ra2))
                phi[i] = B[i].reshape(n * rb1, rb2).T @ phi[i].reshape(n * rb1, ra2)
        B[-1] = A[-1][:, :, 0].T @ phi[-2].T
        B[0] = B[0].reshape(1, B[0].shape[0], B[0].shape[1])
        for i in range(1, len(A) - 1):
            B[i] = torch.einsum('ijk->jik', B[i])
        B[-1] = B[-1].T.reshape(B[-1].shape[1], B[-1].shape[0], 1)
        torchTT.alsTime += time.time()
        return B
        
    @staticmethod
    def zero(A: TTarray, dims: list[int], indices: list[Union[list[int], int, tuple[int]]]) -> TTarray:
        B = A.deepcopy()
        for i, index in enumerate(indices):
            if isinstance(index, list) or isinstance(index, int):
                B[dims[i]][:, index, :] = 0
            elif isinstance(index, tuple):
                B[dims[i]][:, index[0]:index[1], :] = 0
        return B
    
    @staticmethod
    def reduce(A: TTarray, indices: list[Union[list[int], Literal[':'], int, tuple[int]]]) -> TTarray:
        B = A.deepcopy()
        for i, index in enumerate(indices):
            if isinstance(index, list):
                B[i] = B[i][:, index, :]
            elif isinstance(index, int):
                B[i] = B[i][:, index:index+1, :]
            elif isinstance(index, tuple):
                B[i] = B[i][:, index[0]:index[1], :]
        return B
        
    @staticmethod
    def reducedSum(A: TTarray, dim: int, indices: list[Union[list[int], Literal[':'], int, tuple[int]]], scalars: list[int]) -> TTarray:
        B = A.deepcopy()
        for i, index in enumerate(indices):
            if i:
                if isinstance(index, list):
                    B[dim] += scalars[i] * A[dim][:, index, :]
                elif isinstance(index, int):
                    B[dim] += scalars[i] * A[dim][:, index:index+1, :]
                elif isinstance(index, tuple):
                    B[dim] += scalars[i] * A[dim][:, index[0]:index[1], :]
                elif index == ':':
                    B[dim] += scalars[i] * A[dim]
            else:
                if isinstance(index, list):
                    B[dim] = scalars[0] * B[dim][:, index, :]
                elif isinstance(index, int):
                    B[dim] = scalars[0] * B[dim][:, index:index+1, :]
                elif isinstance(index, tuple):
                    B[dim] = scalars[0] * B[dim][:, index[0]:index[1], :]
                else:
                    B[dim] *= scalars[0]
        return B
    
    @staticmethod
    def pad(A: TTarray, dims: list[int], indices: list[list[int]]) -> TTarray:
        B = A.deepcopy()
        for i, index in enumerate(indices):
            j = dims[i]
            s = A[j].shape
            B[j] = torch.zeros(s[0], s[1] + index[0], s[2], dtype = A[j].dtype, device = A[j].device)
            B[j][:, index[1]:index[1]+s[1], :] = A[j]
        return B
    
    @staticmethod
    def preallocate(size: int, dtype: torch.dtype = None, device: torch.device = None) -> None:
        torchTT.preallocated = torch.zeros((size), dtype = dtype, device = device).contiguous()