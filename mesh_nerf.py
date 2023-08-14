# this code file was borrowed and modified from https://github.com/qway/nerfmeshes.git

import argparse
import os
import numpy as np
import torch
# import mmcv
from lib.config_loader import Config
from importlib import import_module
from skimage import measure
from lib import utils, seg_dvgo, seg_dcvgo, dmpigo
from tools.nerf_helpers import export_obj, batchify


def config_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument(
        "--iso-level", type=float, default=32,
        help="Iso-level value for triangulation",
    )
    parser.add_argument(
        "--limit", type=float, default=0.5
    )
    parser.add_argument(
        "--center", type=float, nargs='+', default=[0., 0., 0.]
    )
    parser.add_argument(
        "--res", type=int, default=512,
        help="Sampling resolution for marching cubes, increase it for higher level of detail.",
    )

    return parser


def extract_density(model, args, device, nums, render_kwargs):
    assert (isinstance(nums, tuple) or isinstance(nums, list) or isinstance(nums, int)), \
        "Nums arg should be either iterable or int."

    if isinstance(nums, int):
        nums = (nums,) * 3
    else:
        assert (len(nums) == 3), "Nums arg should be of length 3, number of axes for 3D"

    # Create sample tiles
    tiles = [torch.linspace(args.center[0] - args.limit, args.center[0] + args.limit, nums[0]),
             torch.linspace(args.center[1] - args.limit, args.center[1] + args.limit, nums[1]),
             torch.linspace(args.center[2] - args.limit, args.center[2] + args.limit, nums[2])]

    # Generate 3D samples
    samples = torch.stack(torch.meshgrid(*tiles), -1).view(-1, 3).float()

    density = model.sample_density(samples, **render_kwargs)

    # Radiance 3D grid (rgb + density)
    density = density.view(*nums).contiguous().cpu().numpy()

    return density


def extract_iso_level(density, args):
    # Adaptive iso level
    iso_value = density.mean()
    print(f"Querying based on iso level: {iso_value}")

    return iso_value


def extract_geometry(model, device, args, render_kwargs):
    # Sample points based on the grid
    density = extract_density(model, args, device, args.res, render_kwargs)

    # Adaptive iso level
    iso_value = extract_iso_level(density, args)

    # Extracting iso-surface triangulated
    results = measure.marching_cubes(density, iso_value)

    # Use contiguous tensors
    vertices, triangles, normals, _ = [torch.from_numpy(np.ascontiguousarray(result)) for result in results]

    # Use contiguous tensors
    normals = torch.from_numpy(np.ascontiguousarray(normals))
    vertices = torch.from_numpy(np.ascontiguousarray(vertices))
    triangles = torch.from_numpy(np.ascontiguousarray(triangles))

    base = torch.tensor(args.center) - args.limit
    vertices = base[None].cpu() + vertices * (2 * args.limit / args.res)

    return vertices, triangles, normals, density


def export_marching_cubes(model, args, cfg, render_kwargs, device,segmentation_mask):
    print("Generating mesh geometry...")

    # Modify the rendering process to use the segmentation mask
    def modified_render_fn(target, direction, **render_kwargs):
        result = model(target.to(device), direction.to(device), direction.to(device), **render_kwargs)
        return result['rgb_marched'] * segmentation_mask  # Apply segmentation mask
    
    render_kwargs['render_fn'] = modified_render_fn

    # Extract model geometry
    vertices, triangles, normals, density = extract_geometry(model, device, args, render_kwargs)

    # Target mesh path
    mesh_path = os.path.join(cfg.basedir, cfg.expname, 'mesh.obj')

    # Export model
    rgb = []
    targets, directions = vertices, -normals
    for target, dir in zip(targets.split(8192), directions.split(8192)):
        result = model(target.to(device), dir.to(device), dir.to(device), **render_kwargs)
        rgb.append(result['rgb_marched'])
    rgb = torch.cat(rgb, dim=0).cpu()
    export_obj(vertices, triangles, rgb, normals, mesh_path)


if __name__ == "__main__":
    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    # init environment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    segmentation_mask = torch.ones((args.res, args.res, args.res), dtype=torch.float32).to(device)    

    # load model for rendering
    ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_segmentation_gui.tar')
    ckpt_name = ckpt_path.split('/')[-1][:-4]
    if cfg.data.ndc:
        model_class = dmpigo.DirectMPIGO
    elif cfg.data.unbounded_inward:
        model_class = seg_dcvgo.DirectContractedVoxGO
    else:
        model_class = seg_dvgo.DirectVoxGO
    # model_class = dcvgo.DirectContractedVoxGO
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

    with torch.no_grad():
        # Perform marching cubes and export the mesh
        export_marching_cubes(model, args, cfg, render_kwargs, device,segmentation_mask)

