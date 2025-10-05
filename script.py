from FDTD.grid_comparison_torch import full_test
from FDTD.multi_test import multi_test
multi_test('Periodic', 6, 10, [16, 512], False, 
           [{'error': 1e-4, 'zero': 1e-6, 'type': 0}, {'error': 1e-4, 'zero': 0, 'type': 1}] +
           [{'error': 1e-4, 'zero': float(f'1e{-i}'), 'type': 1} for i in range(3, 7)],
           param = [10, 10])