import os
import torch
from torch.distributed.tensor import distribute_tensor, init_device_mesh, Shard

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
print(f"Running example on {rank} in a world with {world_size}", flush = True)

device_mesh = init_device_mesh("cuda", (4,))
x = distribute_tensor(torch.rand(100, 100, device = "cuda"), device_mesh=device_mesh, placements = [Shard(0)])
print(x, flush = True)
torch.distributed.destroy_process_group()