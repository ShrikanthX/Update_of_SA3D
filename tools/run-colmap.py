import sys
import os
from subprocess import check_output

if __name__=='__main__':

    rootdir = str(sys.argv[1])
    stereo=1
    if(len(sys.argv)>2):
        stereo = 0 #disabled
    
    print(rootdir)
    
    cmds = []
    
    cmds.append("colmap exhaustive_matcher --database_path " + rootdir+"db.db")
    cmds.append("mkdir " + rootdir +"sparse")
    cmds.append("colmap mapper --database_path " + rootdir + "db.db --image_path " + rootdir + "rectify --output_path " + rootdir + "sparse --Mapper.num_threads 16 --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0 --Mapper.init_min_tri_angle 4 --Mapper.multiple_models 0 --Mapper.extract_colors 0")
    
    if(stereo):
        cmds.append("mkdir " + rootdir +"sparse\\0")
        cmds.append("colmap rig_bundle_adjuster --input_path " + rootdir + "sparse\\0_norba --output_path " + rootdir + "sparse\\0 --rig_config_path config.json  --BundleAdjustment.refine_focal_length 0  --BundleAdjustment.refine_extra_params 0")
    cmds.append("python imgs2poses.py " + rootdir)
    
    for i in range(0,len(cmds)):
        print(cmds[i])
        check_output(cmds[i], shell=True)
        if(i==2 and stereo):
            os.rename(os.path.abspath(rootdir + "sparse\\0"),os.path.abspath(rootdir + "sparse\\0_norba"))
        


# 447.10834867,447.10834867,416.51002884,235.38794327
 
# [[447.10834867   0.         416.51002884 -22.46616889]
 # [  0.         447.10834867 235.38794327   0.        ]
 # [  0.           0.           1.           0.        ]]

# colmap point_triangulator \
    # --database_path $PROJECT_PATH/database.db \
    # --image_path $PROJECT_PATH/images
    # --input_path path/to/manually/created/sparse/model \
    # --output_path path/to/triangulated/sparse/model
 
