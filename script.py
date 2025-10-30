from FDTD.grid_comparison_torch import full_test
from FDTD.multi_test import multi_test
import numpy as np
multi_test('Periodic', 1, 3, [16, 512], False, 
           [j for i in [1e-4, 1e-5, 1e-6, 1e-7] for j in
            [{'error': i, 'zero': 0, 'type': 0}, {'error': i, 'zero': 0, 'type': 1}, {'error': i, 'zero': i, 'type': 1}]], 
           param = 1, mu = 1, eps = 1, start_t = 0, noise = [20, 20])