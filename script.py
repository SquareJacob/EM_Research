from FDTD.grid_comparison_torch import full_test
from FDTD.multi_test import multi_test
multi_test('PEC', 2, 10, [256, 256, 512], False, [{'error': 1e-4, 'zero': 1e-3, 'type': 0}], param = 10)