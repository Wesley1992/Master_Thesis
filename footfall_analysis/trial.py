import numpy as np


with open(
        r'C:\Users\Hao\Desktop\Master_Thesis\modal_span_depth_thickness\mdl_span10_l2d12_5_gamma0_1\mdl_span10_l2d12_5_gamma0_1-results.json',
        'rb') as mdl_results:
    f_n = np.array(mdl_results['step_modal']['frequencies'])
    m_n = np.array(mdl_results['step_modal']['masses'])

print(f_n)