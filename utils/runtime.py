import torch
import numpy as np

def random_resize(args, image, label):

    if not args.random_resize:
        return image, label

    size = [192, 224, 256, 288, 320][np.random.randint(0, 5)]
    image = torch.nn.functional.interpolate(image, (size, size), mode='bilinear')

    label = label.unsqueeze(0)
    label = torch.nn.functional.interpolate(label, (size, size), mode='bilinear')
    label = label.squeeze(0)

    label[label >= 0.5] = 1
    label[label < 0.5] = 0

    return image, label


def get_cls_label(args):

    if 'ph' in args.data:
        dic = {}
        cls_file = '/home/db/Joint-seg-cls-jhu/joint-seg-cls-dataset/PH2/cls_label.txt'
        lines = open(cls_file, 'r').readlines()
        for line in lines:
            line = line.strip()
            image_id, label = line.split()
            dic[image_id] = label
        return dic

    elif 'isic' in args.data:
        dic = {}
        cls_file = '/home/db/Joint-seg-cls-jhu/joint-seg-cls-dataset/ISIC2017/cls_label.txt'
        lines = open(cls_file, 'r').readlines()
        for line in lines:
            line = line.strip()
            image_id, label = line.split()
            dic[image_id] = label
        return dic


def name_list_to_cls_label(name_list, label_dic):
    # determine what attribute to use
    tensor_list = []
    for name in name_list:
        tensor_list.append(int(label_dic[name.split('.')[0]]))
    return torch.Tensor(tensor_list).long().cuda()