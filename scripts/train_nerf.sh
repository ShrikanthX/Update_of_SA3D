export CUDA_VISIBLE_DEVICES=0

python run.py --config=configs/llff/fern.py --stop_at=20000 --render_video --i_weights=10000



#For detecting center for mesh generation

python detect_center.py --config configs/nerf_unbounded/seg_bonsai.py
# For extracting mesh 

python mesh_nerf.py --config configs/nerf_unbounded/seg_bonsai.py --center -0.0858 -0.6554 0.2442  --limit .9

# python run.py --config=configs/nerf_unbounded/garden.py --stop_at=40000 --render_video --i_weights=20000

# python run.py --config=configs/lerf/figurines.py --stop_at=40000 --render_video --i_weights=20000