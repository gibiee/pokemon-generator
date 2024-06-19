import glob
import os

img_paths = sorted(glob.glob('./dataset/images/*.png'))

for img_path in img_paths[:10] :
    os.system(f"""python stylegan3-fun/projector.py \
            --network=./stylegan3/training-runs/00005-stylegan3-r-dataset-gpus1-batch4-gamma8/network-snapshot-005000.pkl \
            --cfg=stylegan3-r \
            --target={img_path} \
            --outdir=./projections-fun \
            """)
        
    os.system(f"""python stylegan3-fun/projector.py \
            --network=./stylegan3/training-runs/00005-stylegan3-r-dataset-gpus1-batch4-gamma8/network-snapshot-005000.pkl \
            --cfg=stylegan3-r \
            --target={img_path} \
            --outdir=./projections-fun-stablize \
            --stabilize-projection
            """)
