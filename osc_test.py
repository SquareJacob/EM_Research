import os
import torch
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module, init_device_mesh

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
print(f"Running example on {rank} in a world with {world_size}", flush = True)

X = []
for i in range(4):
    X.append(torch.rand(10, 10, device = f"cuda:{i}"))
for x in X:
    print(x.device)

# device_mesh = init_device_mesh("cuda", (1,4))
# x = distribute_tensor(torch.rand(100, 100, device = "cuda"), device_mesh=device_mesh)
# print(x, flush = True)