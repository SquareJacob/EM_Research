from FDTD.grid_comparison_torch import full_test
from FDTD.multi_test import multi_test
multi_test('PEC', 2, 4, [128, 128, 512], False, [{'error': 1e-4, 'zero': 0, 'type': 1}, {'error': 1e-4, 'zero': 0, 'type': 1}, {'error': 1e-4, 'zero':1e-4, 'type': 1}, {'error': 1e-4, 'zero':1e-3, 'type': 1}], [0, 15], ignore_error = False)