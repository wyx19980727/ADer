# # # python3 runs_single_class.py -d realiad -c configs/invad/invad_realiad.py -n 1 -m -1 -g 0
# # # python runs_single_class.py -d realiad -c configs/invad/invad_lite/lite_realiad.py -n 1 -m -1 -g 0
# # # python3 runs_single_class.py -d realiad -c configs/vitad/vitad_realiad.py -n 1 -m -1 -g 0

# # # CUDA_VISIBLE_DEVICES=0 python run.py -c configs/z_realiad/rd_256_100e.py -m train
# # # python runs_single_class.py -d realiad -c configs/z_realiad/rd_256_100e.py -n 1 -m -1 -g 0

# # CUDA_VISIBLE_DEVICES=0 python run.py -c configs/z_realiad/simplenet_256_100e.py -m train
# # python runs_single_class.py -d realiad -c configs/z_realiad/simplenet_256_100e.py -n 1 -m -1 -g 0

# # # python runs_single_class.py -d realiad -c configs/z_realiad/cflow_256_100e.py -n 1 -m -1 -g 0

# # # python runs_single_class.py -d realiad -c configs/z_realiad/rd++_256_100e.py -n 1 -m -1 -g 0

# # # python runs_single_class.py -d realiad -c configs/z_realiad/cfa_256_100e.py -n 1 -m -1 -g 0

# # sleep 7200
# # # CUDA_VISIBLE_DEVICES=0 python run.py -c configs/z_realiad/mambaad_256_100e.py -m train
# # # python runs_single_class.py -d realiad -c configs/z_realiad/mambaad_256_100e.py -n 1 -m -1 -g 0

# # python runs_single_class.py -d realiad -c configs/z_realiad/pyramidflow_256_100e.py -n 1 -m -1 -g 0

# # python runs_single_class.py -d realiad -c configs/z_realiad/draem_256_100e.py -n 1 -m -1 -g 0

# # python runs_single_class.py -d realiad -c configs/z_realiad/patchcore_256_100e.py -n 1 -m -1 -g 0

# # python runs_single_class.py -d realiad -c configs/z_realiad/destseg_256_100e.py -n 1 -m -1 -g 0

# # python runs_single_class.py -d realiad -c configs/z_realiad/realnet_256_100e.py -n 1 -m -1 -g 0

# # python runs_single_class.py -d realiad -c configs/z_realiad/realnet_256_100e.py -n 1 -m -1 -g 0


# # CUDA_VISIBLE_DEVICES=0 python run.py -c configs/z_realiad_subsample/uniad_realiad.py -m train
# # CUDA_VISIBLE_DEVICES=0 python run.py -c configs/z_realiad_subsample/rd_256_100e.py -m train
# # CUDA_VISIBLE_DEVICES=0 python run.py -c configs/z_realiad_subsample/rdmv_256_100e.py -m train

# # CUDA_VISIBLE_DEVICES=0 python run.py -c configs/z_realiad/lgc_256_100e.py -m train
# # python runs_single_class.py -d realiad -c configs/z_realiad/lgc_256_100e.py -n 1 -m -1 -g 0

# CUDA_VISIBLE_DEVICES=0 python run.py -c configs/z_realiad/rd_256_100e_multiviewcontrast.py -m train
# CUDA_VISIBLE_DEVICES=0 python run.py -c configs/z_realiad/rd_256_100e.py -m train

# python runs_single_class.py -d realiad -c configs/z_realiad/rd_256_100e_multiviewcontrast.py -n 1 -m -1 -g 0

##################

# #CUDA_VISIBLE_DEVICES=0 python run.py -c configs/z_realiad/patchcore_256_100e.py -m train
# CUDA_VISIBLE_DEVICES=0 python run.py -c configs/z_realiad/draem_256_100e.py -m train
# #CUDA_VISIBLE_DEVICES=0 python run.py -c configs/z_realiad/cfa_256_100e.py -m test
# CUDA_VISIBLE_DEVICES=0 python run.py -c configs/z_realiad/cflow_256_100e.py -m train
# #CUDA_VISIBLE_DEVICES=0 python run.py -c configs/z_realiad/patchcore_256_100e.py -m train

# CUDA_VISIBLE_DEVICES=0 python run.py -c runs_compare/rd/rd_256_100e.py -m test vis=True vis_dir=albus/visualization

sleep 14400
CUDA_VISIBLE_DEVICES=0 python run.py -c configs/z_realiad_subsample/rdepipolar_256_100e.py -m train


