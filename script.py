from FDTD.grid_comparison_torch import full_test
from FDTD.multi_test import multi_test
multi_test('Periodic', 6, 10, [4, 512], False, 
           [{'error': 1e-3, 'zero': 0, 'type': 1}, {'error': 1e-3, 'zero': 1e-4, 'type': 1}], 
           param = [10, 10])