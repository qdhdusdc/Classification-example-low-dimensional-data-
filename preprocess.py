import glob
import pandas as pd
import os

folder_path = r'data/datasets'
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

dmbj_files = []
dmh_files = []
dmo_files = []
dmr_files = []

# 遍历所有csv文件，按照文件名前缀分类
for file_name in csv_files:
    if 'DMBJ' in file_name:
        dmbj_files.append(file_name)
    elif 'DMH' in file_name:
        dmh_files.append(file_name)
    elif 'DMO' in file_name:
        dmo_files.append(file_name)
    elif 'DMR' in file_name:
        dmr_files.append(file_name)
        
dmbj_dataset = []
for file_path in dmbj_files:
    data = pd.read_csv(file_path,header=None,usecols=[0,1],skiprows=2)
    data = data.dropna()
    dmbj_dataset.append(data)
dmbj_dataset = sum(dmbj_dataset)/len(dmbj_dataset)
dmbj_dataset[2] = 0 

dmh_dataset = []
for file_path in dmh_files:
    data = pd.read_csv(file_path,header=None,usecols=[0,1],skiprows=2)
    data = data.dropna()
    dmh_dataset.append(data)
dmh_dataset = sum(dmh_dataset)/len(dmh_dataset)
dmh_dataset[2] = 1

dmo_dataset = []
for file_path in dmo_files:
    data = pd.read_csv(file_path,header=None,usecols=[0,1],skiprows=2)
    data = data.dropna()
    dmo_dataset.append(data)
dmo_dataset = sum(dmo_dataset)/len(dmo_dataset)
dmo_dataset[2] = 2

dmr_dataset = []
for file_path in dmr_files:
    data = pd.read_csv(file_path,header=None,usecols=[0,1],skiprows=2)
    data = data.dropna()
    dmr_dataset.append(data)
dmr_dataset[1][1] = dmr_dataset[1][1].astype(float)
dmr_dataset = sum(dmr_dataset)/len(dmr_dataset)
dmr_dataset[2] = 3

data = pd.concat([dmbj_dataset,dmh_dataset,dmo_dataset,dmr_dataset], axis=0)

data.to_csv(r'data\data.csv', index=False)
