from FDTD.grid_comparison_torch import full_test
full_test('PEC', 3, 4, [256, 256], 2, False, 1e-4, zero_thres = 0, tsubstep = 1, save_type = 0, caps = None, ignore_error = False, p = [6, 2])