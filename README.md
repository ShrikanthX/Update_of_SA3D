# An Extension of Segment Anything in 3D with NeRFs (SA3D)

This project is an extension of the implementation of the paper 'Segment Anything in 3D with Nerfs'.
This has [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) as the segmentation network and 3D colored mesh extraction option available.The ideas is to experiment the viability of this pipeline in an animation production environment for asset development.


## Overall Pipeline

With input prompts, Mobile SAM cuts out the target object from the according view. The obtained 2D segmentation mask is projected onto 3D mask grids via density-guided inverse rendering. 2D masks from other views are then rendered, which are mostly uncompleted but used as cross-view self-prompts to be fed into SAM again. Complete masks can be obtained and projected onto mask grids. This procedure is executed via an iterative manner while accurate 3D masks can be finally learned. The script mesh_nerf.py extracts the colored 3D mesh as .obj file.


## Installation

```
git clone https://github.com/ShrikanthX/Update_of_SA3D.git
cd Update_of_SA3D

conda create -n sa3d -c conda-forge colmap python==3.10    # COLMAP for generating camera poses from custom datasets
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
pip install -r requirements.txt

```

### Install SAM , mobile SAM and Grounding-DINO:

```
# Installing SAM
mkdir dependencies; cd dependencies 
mkdir sam_ckpt; cd sam_ckpt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
git clone git@github.com:facebookresearch/segment-anything.git 
cd segment-anything; pip install -e .

# Installing Mobile SAM
git clone git@github.com:ChaoningZhang/MobileSAM.git
cd MobileSAM; pip install -e .
mv weights/mobile_sam.pt ../sam_ckpt/

# Installing Grounding-DINO
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/; pip install -e .
mkdir weights; cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## Download Data
We now release the configs on these datasets:
* *360 unbounded:* [mip-NeRF360](https://jonbarron.info/mipnerf360/), 
* *Buildings drone captures:* [Pix4D](https://support.pix4d.com/hc/en-us/articles/360000235126-Example-projects-real-photogrammetry-data#label2),


### Data structure:  
<details>
  <summary> (click to expand) </summary>

    data
    ├── 360_v2             # Link: https://jonbarron.info/mipnerf360/
    │   └── [bicycle|bonsai|counter|garden|kitchen|room|stump]
    │       ├── poses_bounds.npy
    │       └── [images|images_2|images_4|images_8]
    
</details>

## Usage
- Train NeRF
  ```bash
  python run.py --config=configs/nerf_unbounded/bonsai.py --stop_at=20000 --render_video --i_weights=10000
  ```
- Run SA3D with mobile_SAM in GUI
  ```bash
  python run_seg_gui.py --config=configs/nerf_unbounded/seg_bonsai.py --segment \
  --sp_name=_gui --num_prompts=20 \
  --render_opt=train --save_ckpt --mobile_sam
  ```
- Render and Save Fly-through Videos
  ```bash
  python run_seg_gui.py --config=configs/nerf_unbounded/seg_bonsai.py --segment \
  --sp_name=_gui --num_prompts=20 \
  --render_only --render_opt=video --dump_images \
  --seg_type seg_img seg_density
  ```
- Detect Center for Mesh Extraction
  ```bash
  python detect_center.py --config=configs/nerf_unbounded/seg_bonsai.py 
  ```
- Extract mesh ( copy the center coordinates and radius generated from output log of detect_center.py and paste in --center and --limit below)
  ```bash
  python mesh_nerf.py --config configs/nerf_unbounded/seg_bonsai.py --center -0.0858 -0.6554 0.2442  --limit .9
  ```
 - The code will generate a colored 3D model named mesh.obj in ``./log/nerf_unbounded/dcvgo_bonsai_unbounded``


 ## Using a Custom dataset
- Choose a static object which you want to use (e.g a small toy)
- Take photos covering 360 degrees of the object in an orbital way (at least 50 images are preferred)
- Put all the images into the folder ``./tools/images/source`` like below
    <details>
        <summary> (click to expand) </summary>
      
          tools
          ├── images            
          │   └── source
          │       └── [image_1,image_2,image_3,images_4......]
            
    <details>

- Extract camera poses using COLMAP by running.
  ```bash
  cd tools
  python imgs2poses.py  images
  ```
- Move the 'dense' folder created inside the ``./tools/images`` folder to ``./data/360_v2``
- Rename 'dense' folder to a name as per the subject in the photo. e.g toy. The folder should look like below
    <details>
      <summary> (click to expand) </summary>

                data
                ├── 360_v2             
                │   └── [toy]
                │       ├── poses_bounds.npy
                │       └── [images|images_2|images_4|images_8]
                
    </details>
- Now create a copy of an existing config file in ``config/nerf_unbounded`` and rename it to the same name as before (e.g 'toy.py')
- Create a copy of an existing seg_config file in ``config/nerf_unbounded`` and rename it (e.g 'seg_toy.py')
- Train with 
  ```bash
  python run.py --config=configs/nerf_unbounded/toy.py --stop_at=20000 --render_video --i_weights=10000
  ```
  and - Run SA3D with mobile_SAM in GUI
  ```bash
  python run_seg_gui.py --config=configs/nerf_unbounded/seg_toy.py --segment \
  --sp_name=_gui --num_prompts=20 \
  --render_opt=train --save_ckpt --mobile_sam
  ```

## Acknowledgements
Most of the code in this project are based on the following GitHub repositories
- [Segment Anything in 3D with NeRFs](https://github.com/Jumpat/SegmentAnythingin3D)
- [Mobile SAM](https://github.com/ChaoningZhang/MobileSAM.git)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [DVGO](https://github.com/sunset1995/DirectVoxGO)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO.git)

Mesh extraction code is a modified version from the following Github repo 
- [nerfmeshes](https://github.com/qway/nerfmeshes.git)


