from FDTD.grid_comparison_torch import full_test
from FDTD.multi_test import multi_test
multi_test('Periodic', 3, 10, [4, 512], False, [{'error': 1e-6, 'zero': 1e-6, 'type': 1}], param = 80)