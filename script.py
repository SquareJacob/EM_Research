from FDTD.grid_comparison_torch import full_test
from FDTD.multi_test import multi_test
multi_test('PEC', 1, 10, [4, 256], False, [{'error': 1e-4, 'zero': 1e-4, 'type': 0}, {'error': 1e-4, 'zero': 0, 'type': 1}, {'error': 1e-4, 'zero': 1e-4, 'type': 1}, {'error': 1e-4, 'zero': 1e-3, 'type': 1}, {'error': 1e-3, 'zero': 1e-3, 'type': 1}], param = 80)