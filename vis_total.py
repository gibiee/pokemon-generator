import glob
import os
from PIL import Image

vis_paths = sorted(glob.glob("stylegan3/training-runs/00003-stylegan3-r-dataset-gpus1-batch4-gamma8/fakes*.png"))
vis_paths = [vis_path for vis_path in vis_paths if not "init" in vis_path]
for vis_path in vis_paths :
    img = Image.open(vis_path).convert('RGB')
    fn, _ = os.path.splitext(os.path.basename(vis_path))
    img.save(f"vis/train1_{fn}.jpg")
    print(f"train1_{fn}... saved!")

vis_paths = sorted(glob.glob("stylegan3/training-runs/00005-stylegan3-r-dataset-gpus1-batch4-gamma8/fakes*.png"))
vis_paths = [vis_path for vis_path in vis_paths if not "init" in vis_path]
for vis_path in vis_paths :
    img = Image.open(vis_path).convert('RGB')
    fn, _ = os.path.splitext(os.path.basename(vis_path))
    img.save(f"vis/train2_{fn}.jpg")
    print(f"train2_{fn}... saved!")