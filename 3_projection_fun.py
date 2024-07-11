import glob
import os

img_paths = sorted(glob.glob('./dataset/images/*.png'))

for img_path in img_paths[:10] :
    os.system(f"""python stylegan3-fun/projector.py \
            --network=./stylegan3/training-runs/selection_train2/network-snapshot-009600.pkl \
            --cfg=stylegan3-r \
            --target={img_path} \
            --outdir=./projections/projection-fun \
            """)
        
    os.system(f"""python stylegan3-fun/projector.py \
            --network=./stylegan3/training-runs/selection_train2/network-snapshot-009600.pkl \
            --cfg=stylegan3-r \
            --target={img_path} \
            --outdir=./projections/projection-fun-stablize \
            --stabilize-projection
            """)
