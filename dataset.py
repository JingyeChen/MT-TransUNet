import os
import cv2
import random
import itertools
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from torchvision import transforms
import torchvision.transforms.functional as tf

from utils.sampler import TwoStreamBatchSampler

################# Dataset for Seg
class MyDataSet_seg(data.Dataset):
    def __init__(self, root_path, list_path, root_path_coarsemask=None, crop_size=(224, 224), max_iters=None, label=True, fold=None):
        self.root_path = root_path
        self.root_path_coarsemask = root_path_coarsemask
        self.list_path = list_path
        self.crop_w, self.crop_h = crop_size
        self.label = label

        self.ids = os.listdir(os.path.join(self.root_path, 'Images'))
        self.img_ids = [f'/Images/{i} /Annotations/{i}' for i in self.ids]

        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        if self.label:
            for index, name in enumerate(self.img_ids):
                img_file = name[0:name.find(' ')]
                label_file = name[name.find(' ')+1:]
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })
        else:
            for name in self.img_ids:
                img_file = name[0:name.find(' ')]
                # label_file = name[name.find(' ')+1:]
                self.files.append({
                    "img": img_file,
                    "label": img_file,
                    "name": name
                })


        self.train_augmentation = transforms.Compose(
            [
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.ToPILImage(),
             transforms.Resize(self.crop_w)
             ])

        self.train_coarsemask_augmentation = transforms.Compose(
            [transforms.RandomAffine(degrees=10, translate=(0, 0.1), scale=(0.9, 1.1), shear=5.729578),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.ToPILImage(),
             transforms.Resize(self.crop_w)
             ])

        self.train_gt_augmentation = transforms.Compose(
            [
            # transforms.RandomAffine(degrees=10, translate=(0, 0.1), scale=(0.9, 1.1), shear=5.729578),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.ToPILImage(),
             transforms.Resize(self.crop_w)
             ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(self.root_path + datafiles["img"]).convert('RGB')
        if self.label:
            label = Image.open(self.root_path + datafiles["label"])
        else:
            label = Image.open(self.root_path + datafiles["label"]).convert('L')

        is_crop = [0,1]
        random.shuffle(is_crop)
        if is_crop[0] == 0:
            [WW, HH] = image.size
            p_center = [int(WW / 2), int(HH / 2)]
            crop_num = np.array(range(30, int(np.mean(p_center) / 2), 30))

            random.shuffle(crop_num)
            crop_p = crop_num[0]
            rectangle = (crop_p, crop_p, WW - crop_p, HH - crop_p)
            image = image.crop(rectangle)
            label = label.crop(rectangle)

            image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
            label = label.resize((self.crop_w, self.crop_h), Image.NEAREST)

        else:
            image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
            label = label.resize((self.crop_w, self.crop_h), Image.NEAREST)

        seed = np.random.randint(2147483647)
        random.seed(seed)
        image = self.train_augmentation(image)

        random.seed(seed)
        label = self.train_gt_augmentation(label)

        image = np.array(image) / 255.
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)

        label = np.array(label)
        label = np.float32(label > 0)
        name = datafiles["img"].split('/')[-1]

        return image.copy(), label.copy(), name




class MyValDataSet_seg(data.Dataset):
    def __init__(self, root_path, list_path, root_path_coarsemask=None, crop_size=(224, 224)):
        self.root_path = root_path
        self.root_path_coarsemask = root_path_coarsemask
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size

        self.ids = os.listdir(os.path.join(self.root_path, 'Images'))
        self.img_ids = [f'/Images/{i} /Annotations/{i}' for i in self.ids]

        self.files = []
        for name in self.img_ids:
            img_file = name[0:name.find(' ')]
            label_file = name[name.find(' ')+1:]
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(self.root_path + datafiles["img"]).convert('RGB')
        label = Image.open(self.root_path + datafiles["label"])

        image = image.resize((self.crop_h, self.crop_w), Image.BICUBIC)
        label = label.resize((self.crop_h, self.crop_w), Image.NEAREST)

        image = np.array(image) / 255.
        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32)

        label = np.array(label)

        name = datafiles["img"].split('/')[0]

        return image.copy(), label.copy(), name



class MyTestDataSet_seg(data.Dataset):
    def __init__(self, root_path, list_path, root_path_coarsemask=None, crop_size=(224, 224), fold=None):
        self.root_path = root_path
        self.root_path_coarsemask = root_path_coarsemask
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size

        self.ids = os.listdir(os.path.join(self.root_path, 'Images'))
        self.img_ids = [f'/Images/{i} /Annotations/{i}' for i in self.ids]

        self.files = []
        for index, name in enumerate(self.img_ids):

            if fold is not None:
                if not (index >= fold * 50 and index < (fold + 1) * 50):
                    continue

            img_file = name[0:name.find(' ')]
            label_file = name[name.find(' ')+1:]
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(self.root_path + datafiles["img"]).convert('RGB')
        label = Image.open(self.root_path + datafiles["label"])

        image0 = image.resize((self.crop_h, self.crop_w), Image.BICUBIC)
        image0 = np.array(image0) / 255.
        image0 = image0.transpose(2, 0, 1).astype(np.float32)

        image1 = image.resize((self.crop_h + 32, self.crop_w + 32), Image.BICUBIC)
        image1 = np.array(image1) / 255.
        image1 = image1.transpose(2, 0, 1).astype(np.float32)

        image2 = image.resize((self.crop_h + 64, self.crop_w + 64), Image.BICUBIC)
        image2 = np.array(image2) / 255.
        image2 = image2.transpose(2, 0, 1).astype(np.float32)


        label0 = label.resize((self.crop_h, self.crop_w), Image.NEAREST)
        label0 = np.array(label0)

        label1 = label.resize((self.crop_h + 32, self.crop_w + 32), Image.NEAREST)
        label1 = np.array(label1)

        label2 = label.resize((self.crop_h+64, self.crop_w+64), Image.NEAREST)
        label2 = np.array(label2)

        # name = datafiles["img"][7:23]
        name = datafiles["img"].split('/')[-1]


        return image0.copy(), image1.copy(), image2.copy(), label0.copy(), name


def get_data(args):

    data_str = args.data

    if 'isic' in data_str.lower():
        ############# Load training data
        data_train_root = 'root for /ISIC2017/Training'
        data_train_add_root = 'root for /ISIC2017/Training_addition'
        train_dataset_label = MyDataSet_seg(data_train_root, None, crop_size=(args.w, args.h))
        train_dataset_unlabel = MyDataSet_seg(data_train_add_root, None, crop_size=(args.w, args.h),
                                              label=False)
        train_data = torch.utils.data.ConcatDataset([train_dataset_label, train_dataset_unlabel])
        labeled_idxs = list(range(args.label_data))
        unlabeled_idxs = list(range(args.label_data, args.label_data + args.unlabel_data))

        batch_sampler = TwoStreamBatchSampler(
            labeled_idxs, unlabeled_idxs, args.batch_size, int(args.batch_size / 2))
        trainloader = data.DataLoader(train_data, batch_sampler=batch_sampler, num_workers=8, pin_memory=True)

        ############# Load val data
        data_val_root = 'root for /ISIC2017/Validation'
        valloader = data.DataLoader(MyValDataSet_seg(data_val_root, None, crop_size=(args.w, args.h)),
                                    batch_size=1, shuffle=False,
                                    num_workers=8,
                                    pin_memory=True)

        ############# Load testing data
        data_test_root = 'root for /ISIC2017/Testing'
        testloader = data.DataLoader(
            MyTestDataSet_seg(data_test_root, None, crop_size=(args.w, args.h)), batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True)

        return {
            'trainloader': trainloader,
            'valloader': valloader,
            'testloader': testloader
        }

    elif 'ph' in args.data_str.lower():
        data_test_ph2_root = 'root for /PH2Dataset/PH2Dataset/PH2 Dataset images/'
        # data_test_root_mask = 'Coarse_masks/Testing_EnhancedSN/'
        data_test_list = 'root for /ISIC/ph2_test.txt'
        testloader = data.DataLoader(
            MyTestDataSet_seg(data_test_ph2_root, None, crop_size=(args.w, args.h)), batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True)

        return {
            'trainloader': None,
            'valloader': None,
            'testloader': testloader
        }


