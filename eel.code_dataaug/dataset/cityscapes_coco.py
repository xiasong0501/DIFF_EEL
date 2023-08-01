import torch
import numpy
import random
from PIL import Image
from dataset.training.cityscapes import Cityscapes
from dataset.training.coco import COCO
import time 
from PIL import ImageFile
import os

def return_ids(obj_class):
    if obj_class== 'bus':
        return([15,28])
    if obj_class== 'truck':
        return([14,27])
    if obj_class== 'bicycle':
        return([18,33])
    else:
        return([19,34])

class CityscapesCocoMix(torch.utils.data.Dataset):
    def __init__(self, split, preprocess, cs_root='', coco_root="", cs_split=None, coco_split=None):

        self._split_name = split
        self.preprocess = preprocess

        if cs_split is None or coco_split is None:
            self.cs_split = split
            self.coco_split = split
        else:
            self.cs_split = cs_split
            self.coco_split = coco_split

        self.city = Cityscapes(root=cs_root, split=self.cs_split)
        self.coco = COCO(root=coco_root, split=self.coco_split, proxy_size=int(len(self.city)))

        self.city_number = len(self.city.images)
        self.ood_number = len(self.coco.images)
        self.train_id_out = self.coco.train_id_out
        self.num_classes = self.city.num_train_ids
        self.mean = self.city.mean
        self.std = self.city.std
        self.void_ind = self.city.ignore_in_eval_ids

    def __getitem__(self, idx):
        city_idx, anomaly_mix_or_not = idx[0], idx[1]
        # city_idx = idx
        """Return raw image, ground truth in PIL format and absolute path of raw image as string"""
        ImageFile.LOAD_TRUNCATED_IMAGES = True     
        try:
              
            city_image = numpy.array(Image.open(self.city.images[city_idx]).convert('RGB'), dtype=float)
        except:
            # if city_idx>10:
            #     city_idx=city_idx-1
            # if city_idx<=10:
            #     city_idx=city_idx+1  
            #time.sleep(4)
            city_image = numpy.array(Image.open(self.city.images[city_idx]).convert('RGB'), dtype=float)
        try:   
            city_target = numpy.array(Image.open(self.city.targets[city_idx]).convert('L'), dtype=int) #如果你不需要训练color，那你只需要返回这两项提前生成好的东西，但是对于实时添加位置和处理，那还需要新的准备；
        except:
            #time.sleep(4)
            city_target = numpy.array(Image.open(self.city.targets[city_idx]).convert('L'), dtype=int)       
        one = numpy.ones_like(city_target)*254
        city_target = numpy.where(city_target == 19, one, city_target)
        CoCo_diff=random.randint(0, 1)
        if CoCo_diff==1:
            ood_idx = random.randint(0, self.ood_number-1)
            ood_image = numpy.array(Image.open(self.coco.images[ood_idx]).convert('RGB'), dtype=float)
            ood_target = numpy.array(Image.open(self.coco.targets[ood_idx]).convert('L'), dtype=int)  #add a random selection in this place, including coco and diffusion
        else:
            object_list=os.listdir('DALLE2 object')
            obj_num=random.randint(0,int(len(object_list)/1)-1)#随机选取一个object类
            obj_class=object_list[obj_num]
            if obj_class== 'bus' or 'truck' or 'bicycle':
                obj_class= 'mixed_vehicle'
            obj_names=os.listdir('DALLE2 object/'+obj_class) 
            item_nums=int(len(obj_names)/3) 
            item_num=random.randint(0,item_nums-1)
            # obj_name= 'DALLE2 object/'+obj_class + '/'+ str(item_num)+ '.png'
            mask_name= 'DALLE2 object/'+obj_class + '/'+ str(item_num)+ 'mask.png'
            result_name= 'DALLE2 object/'+obj_class + '/'+ str(item_num)+ '.png'
            ood_image = numpy.array(Image.open(result_name).convert('RGB'), dtype=float)
            ood_target = numpy.array(Image.open(mask_name).convert('L'), dtype=int)
            mask_token=numpy.ones_like(ood_target)*254
            ood_target= numpy.where(ood_target >200, mask_token, ood_target)
            #ood_target= (ood_target/255)*254
        # city_image, city_target= self.preprocess(city_image, city_target)
        city_image, city_target, city_mix_image, city_mix_target, \
           ood_image, ood_target = self.preprocess(city_image, city_target, ood_image, ood_target,
                                                   anomaly_mix_or_not=anomaly_mix_or_not)

        return torch.tensor(city_image, dtype=torch.float), torch.tensor(city_target, dtype=torch.long), \
            torch.tensor(city_mix_image, dtype=torch.float), torch.tensor(city_mix_target, dtype=torch.long), \
            torch.tensor(ood_image, dtype=torch.float), torch.tensor(ood_target, dtype=torch.long)
        
        # return torch.tensor(city_image, dtype=torch.float), torch.tensor(city_target, dtype=torch.long)

    def __len__(self):
        """Return total number of images in the whole dataset."""
        return len(self.city.images)

    def __repr__(self):
        """Return number of images in each dataset."""
        fmt_str = 'Cityscapes Split: %s\n' % self.cs_split
        fmt_str += '----Number of images: %d\n' % len(self.city)
        fmt_str += 'COCO Split: %s\n' % self.coco_split
        #fmt_str += '----Number of images: %d\n' % len(self.coco)
        return fmt_str.strip()
