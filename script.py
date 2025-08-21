from FDTD.grid_comparison_torch import full_test
#Boundary, Solution, Simulation Type
trials = [[b, s + 1, 1] for b in ["PEC", "Periodic"] for s in range (5)]
for trial in trials:
    full_test(trial[0], trial[1], 50, [512, 512, 512], trial[2], False, 1e-4, None, 1e-3)