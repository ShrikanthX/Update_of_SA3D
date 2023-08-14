_base_ = './nerf_unbounded_default.py'

expname = 'dcvgo_windmill_unbounded'

data = dict(
    datadir='./data/360_v2/windmill',
    factor=4, # 1558x1039
    bd_factor=None,
    movie_render_kwargs=dict(
        shift_y=-0.0,
        scale_r=0.9,
        pitch_deg=-20,
    ),
)
