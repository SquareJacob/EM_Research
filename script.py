from FDTD.grid_comparison_torch import full_test

for b in ["PEC", "Periodic"]:
    for s in range(5):
        full_test(b, s + 1, 50, [4, 512, 512], 2, False, 1e-4)