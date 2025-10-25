from FDTD.grid_comparison_torch import full_test
from FDTD.multi_test import multi_test
import numpy as np
multi_test('PEC', 1, 3, [16, 512], False, 
           [{'error': 1e-7, 'zero': 1e-7, 'type': 0}, {'error': 1e-7, 'zero': 0, 'type': 1}, {'error': 1e-7, 'zero': 1e-7, 'type': 1}], 
           param = 1, mu = 1, eps = 1, start_t = 0, noise = [0, 20])