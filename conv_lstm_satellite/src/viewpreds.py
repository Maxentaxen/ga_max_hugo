import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
import os
folders = [
    Path('predictions/preds'),
    Path('predictions/targets'),
    Path('predictions/diffs')
]
columns = []
os.chdir('..')

print(os.getcwd())
for folder in folders:
    imgs = []
    for img_path in sorted(folder.iterdir()):
          imgs.append(Image.open(img_path).convert("RGB"))
    columns.append(imgs)

n_cols = len(columns)
n_rows = max(len(col) for col in columns)
titles = ['Targets', 'Predictions', 'Difference']
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

for col_idx, imgs in enumerate(columns):
    for row_idx in range(n_rows):
        ax = axes[row_idx, col_idx]
        if row_idx < len(imgs):
            lum_img = np.asarray(imgs[row_idx])
            ax.imshow(lum_img)
        ax.axis("off")
max_values = []

for image in columns[2]:
    max_values.append(1 - np.mean(image)/(255))
for ax, title in zip(axes[0], titles):
    ax.set_title(title)
print(f'Mean blackness: {np.mean(max_values)}')
plt.show()
