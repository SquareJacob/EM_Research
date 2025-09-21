from FDTD.grid_comparison_torch import full_test
from FDTD.multi_test import multi_test
#multi_test('PEC', 2, 8, [128, 128, 512], False, [{'error': 1e-4, 'zero': 0, 'type': 0}, {'error': 1e-4, 'zero': 0, 'type': 1}, {'error': 1e-4, 'zero':1e-4, 'type': 1}, {'error': 1e-4, 'zero':1e-3, 'type': 1}], [0, 15], ignore_error = False)
#multi_test('Periodic', 6, 10, [512, 512], False, [{'error': 0, 'zero': 0, 'type': 0},{'error': 1e-4, 'zero': 0, 'type': 1},{'error': 1e-4, 'zero': 1e-4, 'type': 1},{'error': 1e-4, 'zero': 1e-3, 'type': 1}], param = [10, 10])
multi_test('Periodic', 6, 10, [4, 512, 512], False, [{'error': 0, 'zero': 0, 'type': 0}], param = [10, 10])