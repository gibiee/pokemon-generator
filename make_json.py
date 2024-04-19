import pandas as pd
import json

csv_path = 'dataset/info.csv'
json_save_path = 'dataset/dataset.json'

df = pd.read_csv(csv_path, index_col=0)
print(f'df load... {len(df)} items')
label_list = []
for i in range(len(df)) :
    item = df.iloc[i]

    img_path = item.img_path
    idxs = [idx for idx, val in enumerate(item[1:]) if val == 1]
    for idx in idxs :
        label_list.append([img_path, idx])

print(f'label_list items : {len(label_list)} items')

label = {
    "labels": label_list
}
with open(json_save_path, 'w') as f :
    json.dump(label, f, indent=2)