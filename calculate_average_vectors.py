import pandas as pd
import os
import numpy as np

df = pd.read_csv('dataset/info.csv', index_col=0)

data = {}
for i in range(len(df)) :
    row = df.iloc[i]
    for class_name, value in row.items() :
        if value == 1 :
            img_path = row['img_path']
            base = os.path.basename(img_path)
            fn, _ = os.path.splitext(base)

            vector_path = f"projection/{fn}_projected_w.npz"
            vector = np.load(vector_path)['w']
            if data.get(class_name) == None :
                data[class_name] = [vector, 1]
            else :
                data[class_name][0] += vector
                data[class_name][1] += 1

average_vectors = {}
for class_name, (vector_sum, count) in data.items() :
    average_vectors[class_name] = vector_sum / count

os.makedirs('./average_vectors', exist_ok=True)
for class_name, avg_vector in average_vectors.items() :
    np.save(f'average_vectors/{class_name}.npy', avg_vector)