from FDTD.grid_comparison_torch import full_test
from FDTD.multi_test import multi_test
import numpy as np
multi_test('PEC', 1, np.sqrt(2537).item(), [16, 256], False, 
           #[{'error': 1e-4, 'zero': 1e-6, 'type': 0}, {'error': 1e-4, 'zero': 0, 'type': 1}] +
           #[{'error': 1e-4, 'zero': float(f'1e{-i}'), 'type': 1} for i in range(3, 7)],
           [{'error': 0, 'zero': 0, 'type': 0}],
           param =  80, mu = 1, eps = 1)