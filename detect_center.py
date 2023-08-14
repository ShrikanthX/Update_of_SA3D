import argparse
import os
os.environ['PATH'] += ':/home/linyf/miniconda3/bin/'
import numpy as np
import torch
import mmcv
from importlib import import_module
from skimage import measure
from lib import utils, dvgo, dcvgo, dmpigo
from tools.nerf_helpers import export_obj, batchify


def config_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--N_iters", type=int, default=4)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--density_thres", type=float, default=1e-2)

    return parser


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load model for rendering
    ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_segmentation_gui.tar')
    ckpt_name = ckpt_path.split('/')[-1][:-4]
    if cfg.data.ndc:
        model_class = dmpigo.DirectMPIGO
    elif cfg.data.unbounded_inward:
        model_class = dcvgo.DirectContractedVoxGO
    else:
        model_class = dvgo.DirectVoxGO
    model = utils.load_model(model_class, ckpt_path).to(device)
    stepsize = cfg.fine_model_and_render.stepsize
    render_kwargs = {
        'bg': 1 if cfg.data.white_bkgd else 0,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }

    center = torch.zeros(3)
    radius = args.radius
    scaling = (args.radius // 2) ** (1 / (args.N_iters - 1))
    results = torch.zeros(0, 4)
    for i in range(args.N_iters):
        tiles = [torch.linspace(center[0].item() - radius, center[0].item() + radius, args.res),
                 torch.linspace(center[1].item() - radius, center[1].item() + radius, args.res),
                 torch.linspace(center[2].item() - radius, center[2].item() + radius, args.res)]
        samples = torch.stack(torch.meshgrid(*tiles), -1).view(-1, 3).float()
        density = model.sample_density(samples, **render_kwargs).view(-1, 1)
        samples = torch.cat([samples, density], dim=-1)
        results = torch.cat([results, samples], dim=0)
        # recalculate center and radius
        normalizer = results[:, -1].sum()
        center = (results[:, :3] * results[:, 3:]).sum(dim=0) / normalizer
        radius = radius * scaling
    print(center)
    results[:, :3] -= center
    mask = results[:, 3] > args.density_thres
    radius = results[mask][:, :3].abs().amax()
    print(radius)
