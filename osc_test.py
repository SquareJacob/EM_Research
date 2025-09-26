import os
import torch
from torch.distributed.tensor import distribute_tensor, init_device_mesh, Shard
import torch.distributed as dist

dist.init_process_group(backend = "nccl")
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
print(f"Running example on {rank} in a world with {world_size}", flush = True)

device_mesh = init_device_mesh("cuda", (world_size,))
x = distribute_tensor(torch.rand(8, 8, device = "cuda"), device_mesh=device_mesh, placements = [Shard(0)])
print(f"ORIGINAL:{x.full_tensor()}", flush = True)
dist.barrier()
print(f"SHAPE:{x.shape}", flush = True)
dist.barrier()
print(f"ADDED:{(x + x.T).full_tensor()}", flush = True)
dist.barrier()
#x[:, 0] = 0
local = x.to_local()
zero = 7
if rank == zero // (x.shape[0] / world_size):
    local[int(zero % (x.shape[0] / world_size)), :].zero_()
#local[:, 3].zero_()
dist.barrier()
print(f"FULL ZERO:{x.full_tensor()}", flush = True)
torch.distributed.destroy_process_group()