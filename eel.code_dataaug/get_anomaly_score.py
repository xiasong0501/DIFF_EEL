import argparse
from collections import OrderedDict
import torch.optim
from config.config import config
from dataset.training.cityscapes import Cityscapes
from dataset.validation.fishyscapes import Fishyscapes
from dataset.validation.lost_and_found import LostAndFound
from dataset.validation.road_anomaly import RoadAnomaly
from dataset.validation.segment_me_if_you_can import SegmentMeIfYouCan
from engine.engine import Engine
from model.network import Network
from utils.img_utils import Compose, Normalize, ToTensor
from utils.wandb_upload import *
from utils.logger import *
from engine.evaluator import SlidingEval
from valid import valid_anomaly, valid_epoch, final_test_inlier
import os
import torch
import torchvision
import cv2
from PIL import Image, UnidentifiedImageError
import numpy as np
from torch.cuda.amp import autocast as autocast
def compute_anomaly_score(score, mode='energy'):
    score = score.squeeze()[:19]
    if mode == 'energy':
        prob = torch.softmax(score, dim=0)

        top2_prob=torch.topk(prob,2,dim=0)[0]
        prob_diff=top2_prob[0,:,:]-top2_prob[1,:,:]
        prob_max=top2_prob[0,:,:]
        entorpy_score = -torch.sum(prob * torch.log(prob), dim=0)
        normalized_entropy=entorpy_score
        normalized_entropy=normalized_entropy-normalized_entropy.mean()
        energy=torch.log(torch.sum(torch.exp(score),dim=0))
        reg_energy=torch.sum(torch.exp(score+2*prob),dim=0)
        normalize_energy1=(energy-energy.min())/(energy.max()-energy.min())*5
        normalize_energy=energy-energy.mean()
        sort_normalized_entropy=normalized_entropy.flatten().sort().values
        thre_indx=int(len(sort_normalized_entropy)/10)
        threshold_entropy=sort_normalized_entropy[thre_indx]
        normalized_entropy=torch.nn.functional.relu(normalized_entropy-threshold_entropy)+0.01
        plus_entropy=1/normalized_entropy
        # anomaly_score=entorpy_score
        anomaly_score = -(1. * torch.logsumexp(score, dim=0))*1+entorpy_score*1
        #anomaly_score = -normalize_energy1*1+normalized_entropy*0.2
       # anomaly_score=-torch.max(prob,dim=0)[0] #max_softmax
        # anomaly_score = -torch.log(reg_energy)
        # anomaly_score=-(1. * torch.log((plus_entropy)*torch.sum(torch.exp(score),dim=0)))
        
        
        # one = torch.ones_like(anomaly_score)*anomaly_score.min()
        # anomaly_score= torch.where(normalized_entropy < 0.0001, one, anomaly_score)
        # anomaly_score=-(1. * torch.log((prob_diff+0.01)*torch.sum(torch.exp(score),dim=0)))
        # anomaly_score=-(1. * torch.log((1/(entorpy_score+0.01))*torch.sum(torch.exp(score),dim=0)))

    elif mode == 'entropy':
        prob = torch.softmax(score, dim=0)
        anomaly_score=-torch.max(prob,dim=0)[0]
        #anomaly_score=-(torch.max(prob,dim=0)[0]*(1. * torch.logsumexp(score, dim=0)))
        #anomaly_score = -torch.sum(prob * torch.log(prob), dim=0) / torch.log(torch.tensor(19.))
    else:
        raise NotImplementedError

    # regular gaussian smoothing
    anomaly_score=anomaly_score.detach().cpu().numpy()
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # kernel_1 = np.ones((4, 4), dtype=np.uint8)
    # anomaly_score = cv2.erode(anomaly_score, kernel, 1)
    # anomaly_score = cv2.dilate(anomaly_score, kernel, 1)
    # anomaly_score = cv2.dilate(anomaly_score, kernel_1, 1)
    anomaly_score = torch.from_numpy(anomaly_score)
    anomaly_score = anomaly_score.unsqueeze(0)
    anomaly_score = torchvision.transforms.GaussianBlur(7, sigma=1)(anomaly_score)
    anomaly_score = anomaly_score.squeeze(0)
    return anomaly_score

def get_anomaly_detector(ckpt_path):
    """
    Get Network Architecture based on arguments provided
    """
    ckpt_name = ckpt_path
    model = Network(config.num_classes)
    state_dict = torch.load(ckpt_name)
    state_dict = state_dict['model'] if 'model' in state_dict.keys() else state_dict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=True)
    return model


def main(gpu, ngpus_per_node, config, args):
    args.local_rank = gpu
    logger = logging.getLogger("ours")
    logger.propagate = False

    engine = Engine(custom_arg=args, logger=logger,
                    continue_state_object=config.pretrained_weight_path)

    transform = Compose([ToTensor(), Normalize(config.image_mean, config.image_std)])

    #cityscapes_val = Cityscapes(root=config.city_root_path, split="val", transform=transform)
    #cityscapes_test = Cityscapes(root=config.city_root_path, split="test", transform=transform)
    #evaluator = SlidingEval(config, device=0 if engine.local_rank < 0 else engine.local_rank)
    # fishyscapes_ls = Fishyscapes(split='LostAndFound', root=config.fishy_root_path, transform=transform)
    # fishyscapes_static = Fishyscapes(split='Static', root=config.fishy_root_path, transform=transform)
    # segment_me_anomaly = SegmentMeIfYouCan(split='road_anomaly', root=config.segment_me_root_path, transform=transform)
    # segment_me_obstacle = SegmentMeIfYouCan(split='road_obstacle', root=config.segment_me_root_path,
    #                                         transform=transform)
    # road_anomaly = RoadAnomaly(root=config.road_anomaly_root_path, transform=transform)
    model = get_anomaly_detector(config.rpl_weight_path)
    #vis_tool = Tensorboard(config=config)

    if engine.distributed:
        torch.cuda.set_device(engine.local_rank)
        model.cuda(engine.local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[engine.local_rank],
                                                          find_unused_parameters=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    # dataset_path="datasets/SMIC/dataset_AnomalyTrack/images"
    dataset_name='dataset_AnomalyTrack'
    # dataset_name='dataset_ObstacleTrack'
    dataset_path="datasets/SMIC/"+dataset_name+"/images"
    image_names=os.listdir(dataset_path)
    i=0
    with torch.no_grad():
         with autocast(dtype=torch.float16):
            model.eval()
            for image_name in image_names:
                i=i+1
                image=Image.open(dataset_path+'/'+image_name).convert('RGB')
                image, mask = transform(image, image)
                img = image.cuda(non_blocking=True)
                _, logits = model.module(img) if engine.distributed else model(img)
                #assert 1==0, logits.shape
                logits=logits.to(torch.float32)
                anomaly_score = compute_anomaly_score(logits, mode='energy').cpu()   
                anomaly_score=anomaly_score.numpy()
                if dataset_name=='dataset_ObstacleTrack':
                    np_name='outputs_obs/'+image_name[0:-5]+'.npy'
                    print(i)
                if dataset_name=='dataset_AnomalyTrack':
                    np_name='outputs_amy/'+image_name[0:-4]+'.npy'
                    print(i)
                np.save(np_name,anomaly_score)
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomaly Segmentation')
    parser.add_argument('--gpus', default=4,
                        type=int,
                        help="gpus in use")
    parser.add_argument("--ddp", action="store_true",
                        help="distributed data parallel training or not;"
                             "MUST SPECIFIED")
    parser.add_argument('-l', '--local_rank', default=-1,
                        type=int,
                        help="distributed or not")
    parser.add_argument('-n', '--nodes', default=1,
                        type=int,
                        help="distributed or not")

    args = parser.parse_args()
    args.world_size = args.nodes * args.gpus

    # we enforce the flag of ddp if gpus >= 2;
    args.ddp = True if args.world_size > 1 else False
    if args.gpus <= 1:
        main(-1, 1, config=config, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=args.gpus, args=(args.gpus, config, args))