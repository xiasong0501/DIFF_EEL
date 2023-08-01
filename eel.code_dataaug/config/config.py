import os
import numpy
from easydict import EasyDict

C = EasyDict()
config = C
cfg = C

C.seed = 666

"""Root Directory Config"""
C.repo_name = 'ood_seg'
C.root_dir = os.path.realpath("")

"""Data Dir and Weight Dir"""
C.city_root_path = 'datasets/cityscapes_large'
C.coco_root_path = 'datasets/COCO'
C.fishy_root_path = 'datasets/fishyscapes'
C.segment_me_root_path = 'datasets/SMIC'
C.road_anomaly_root_path = 'datasets/RoadAnomaly'
C.diff_amy_root_path = 'datasets/Diff_amy'
C.diff_amy_lowres_root_path='datasets/Diff_amy_lowres'

#################################################
###########
C.rpl_weight_path = os.path.join(C.root_dir, 'ckpts', 'exp', 'checkpoint_best.pth')  #6500 7500 8500 9500
# C.rpl_weight_path = os.path.join(C.root_dir, 'ckpts', 'exp', 'rpl.code+corocl/iter_3200.pth') 
# C.rpl_weight_path = os.path.join(C.root_dir, 'ckpts', '27_withbg0.5_withmse/iter_3000.pth') 
C.measure_way = "energy"
C.proj_name = "7.20_Diff_EEL_p=0.50"
C.lr = 5e-5
C.experiment_name = "Diff_EEL_p=0.50"
#################################################
############
C.saved_dir = os.path.join("ckpts/exp", C.experiment_name)
C.pretrained_weight_path = os.path.join(C.root_dir, 'ckpts', 'pretrained_ckpts', 'cityscapes_best.pth')

"""Network Config"""
C.fix_bias = True 
C.bn_eps = 1e-5
C.bn_momentum = 0.1

"""Image Config"""
C.num_classes = 19
C.outlier_exposure_idx = 254  # NOTE: it starts from 0

C.image_mean = numpy.array([0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
C.image_std = numpy.array([0.229, 0.224, 0.225])

C.city_image_height = 800
C.city_image_width = 800

C.ood_image_height = C.city_image_height
C.ood_image_width = C.city_image_width

# C.city_train_scale_array = [0.5, 0.75, 1, 1.5, 1.75, 2.0]
C.ood_train_scale_array = [.25, .5, .5, .75, .1, .125]

C.num_train_imgs = 2975
C.num_eval_imgs = 500

"""Train Config"""
#C.lr = 7.5e-4
C.batch_size = 3
C.energy_weight = .05

C.lr_power = 2
C.momentum = 0.9
C.weight_decay = 1e-4

C.nepochs = 30
C.niters_per_epoch = C.num_train_imgs // C.batch_size

C.num_workers = 2
C.void_number = 5
C.warm_up_epoch = 0

"""Eval Config"""
C.eval_iter = int(C.niters_per_epoch *2)
#C.measure_way = "energy"
C.eval_stride_rate = 1 / 3
C.eval_scale_array = [1., ]
C.eval_flip = True
C.eval_crop_size = 1024

"""Display Config"""
C.record_info_iter = 20
C.display_iter = 50

"""Wandb Config"""
# Specify you wandb environment KEY; and paste here
C.wandb_key = "f3200214b40213dd3fe34d912d68738f9f79e25a"

# Your project [work_space] name
#C.proj_name = "OoD_Segmentation_7"

# reg1:=entropy dis-similarity; reg2:=res_ neg-entropy

#C.experiment_name = "rpl.code+corocl"

# half pretrained_ckpts-loader upload images; loss upload every iteration
C.upload_image_step = [0, int((C.num_train_imgs / C.batch_size) / 2)]

# False for debug; True for visualize
C.wandb_online = True

"""Save Config"""
#C.saved_dir = os.path.join("ckpts/exp", C.experiment_name)

if not os.path.exists(C.saved_dir):
    os.mkdir(C.saved_dir)

