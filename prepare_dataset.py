import os
import polars as pl

path = '/home/linhdang/workspace/PAPER_Material/FINAL-DATASET/train/train/labels'

output = []

for name in os.listdir(path):
    img_path = os.path.join(path, name)
    with open(img_path, 'r') as f:
        label = int(f.read())
    print(label)
    
    img_name = name.split('.')[0] + '.jpg'
    output.append({
        'name': img_name,
        'answer': label
    })

df = pl.DataFrame(output)
df.write_csv('classes.csv')