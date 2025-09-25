import torch
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module, init_device_mesh

device_mesh = init_device_mesh("cuda", (4,))
x = distribute_tensor(torch.rand(100, 100, device = "cuda"), device_mesh=device_mesh)
print(x, flush = True)