from FDTD.grid_comparison_torch import full_test
from FDTD.multi_test import multi_test
import numpy as np
multi_test('PEC', 1, 10, [8, 512], False, 
           [{'error': 1e-4, 'zero': 1e-4, 'type': 0}] + 
           [{'error': float(f'1e{-i}'), 'zero': 0, 'type': 1} for i in range(3, 7)] +
           [{'error': float(f'1e{-j}'), 'zero': float(f'1e{-i}'), 'type': 1} for i in range(3, 7) for j in range(3, 7)], 
           param =  20, mu = 1, eps = 1, start_t = 0.2)