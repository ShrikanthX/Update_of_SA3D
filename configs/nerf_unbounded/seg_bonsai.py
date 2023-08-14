_base_ = './seg_nerf_unbounded_default.py'

expname = 'dcvgo_bonsai_unbounded'

data = dict(
    datadir='./data/360_v2/bonsai',
    factor=8, # 1559x1039
    movie_render_kwargs=dict(
        shift_x=0.0,  # positive right
        shift_y=0, # negative down
        shift_z=0,
        scale_r=1.0,
        pitch_deg=-30, # negative look downward
    ),
)

