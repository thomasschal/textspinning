import pandas as pd
from openpyxl.workbook import Workbook

data_de = pd.read_json('bin\\eval_de_data-ba889fef-656b-4f64-a65e-16219e5d422e.json')
data_cn = pd.read_json('bin\\eval_cn_data-ac5f464b-aee3-4c88-b4f7-dd27c79f7754.json')

mean_de = data_de.mean(axis=0)
mean_cn = data_cn.mean(axis=0)

median_de = data_de.median(axis=0)
median_cn = data_cn.median(axis=0)

min_de = data_de.min(axis=0)
min_cn = data_cn.min(axis=0)

max_de = data_de.max(axis=0)
max_cn = data_cn.max(axis=0)

#data_de.to_excel(excel_writer='bin\\de_data.xlsx')
#data_cn.to_excel(excel_writer='bin\\cn_data.xlsx')

print('Hello World')