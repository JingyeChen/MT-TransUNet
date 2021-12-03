import os
import cv2
import time
import numpy as np

import torch
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

from utils.runtime import get_cls_label, name_list_to_cls_label
from utils.metrics import cla_evaluate

def val_mode_seg_multi_scale(args, valloader, model, path, test=False, visualize=False, ph2=False, logging=None, cls_dic=None):
    dice = []
    sen = []
    spe = []
    acc = []
    jac_score = []

    if not os.path.exists(os.path.join(path, 'image')):
        os.mkdir(os.path.join(path, 'image'))
    f = open(os.path.join(path, 'image', 'result_analysis.txt'), 'w+')
    to_img = transforms.ToPILImage()

    # cls_dic = get_cls_label(test, ph2)

    nevus = [0, 0]
    melanoma = [0, 0]
    seborrheic = [0, 0]
    melanoma_correct = 0
    seborrheic_correct = 0

    m_gt = []
    m_pred_binary = []
    m_pred_prob = []

    s_gt = []
    s_pred_binary = []
    s_pred_prob = []

    att_collection = []

    for image_index, batch in enumerate(valloader):

        if image_index % 50 == 0:
            print('Val {} / {}'.format(image_index, len(valloader)))

        data0, data1, data2, mask, name = batch
        cls_gt = name_list_to_cls_label(name, cls_dic)  # 这个0,1,2代表三类

        data0 = data0.cuda()
        data1 = data1.cuda()
        data2 = data2.cuda()
        mask = mask[0].data.numpy()
        val_mask = np.int64(mask > 0)

        model.eval()
        result = 0
        cls_result = 0

        time0 = time.time()
        with torch.no_grad():
            data = data0
            rot_90 = torch.rot90(data, 1, [2, 3])
            rot_180 = torch.rot90(data, 2, [2, 3])
            rot_270 = torch.rot90(data, 3, [2, 3])
            hor_flip = torch.flip(data, [-1])
            ver_flip = torch.flip(data, [-2])
            data = torch.cat([data, rot_90, rot_180, rot_270, hor_flip, ver_flip], dim=0)  # 做了六种数据增强
            pred, _, cls_logits, att_maps, _ = model(data)

            now_att_maps = []
            # print(len(att_maps))
            # print(att_maps)
            for i in att_maps:
                # print(i.shape)
                now_att_maps.append(torch.mean(i[0, ...], 0).cpu())
            att_collection.append(now_att_maps)

            pred = F.interpolate(pred, size=(args.w, args.h), mode='bicubic')
            result += pred[0:1] + torch.rot90(pred[1:2], 3, [2, 3]) + torch.rot90(pred[2:3], 2, [2, 3]) + torch.rot90(
                pred[3:4], 1, [2, 3]) + torch.flip(pred[4:5], [-1]) + torch.flip(pred[5:6], [-2])
            cls_result += cls_logits

            data = data1
            rot_90 = torch.rot90(data, 1, [2, 3])
            rot_180 = torch.rot90(data, 2, [2, 3])
            rot_270 = torch.rot90(data, 3, [2, 3])
            hor_flip = torch.flip(data, [-1])
            ver_flip = torch.flip(data, [-2])
            data = torch.cat([data, rot_90, rot_180, rot_270, hor_flip, ver_flip], dim=0)  # 做了六种数据增强
            pred, _, cls_logits, att_maps, _ = model(data)
            pred = F.interpolate(pred, size=(args.w, args.h), mode='bicubic')
            result += pred[0:1] + torch.rot90(pred[1:2], 3, [2, 3]) + torch.rot90(pred[2:3], 2, [2, 3]) + torch.rot90(
                pred[3:4], 1, [2, 3]) + torch.flip(pred[4:5], [-1]) + torch.flip(pred[5:6], [-2])
            cls_result += cls_logits

            data = data2
            rot_90 = torch.rot90(data, 1, [2, 3])
            rot_180 = torch.rot90(data, 2, [2, 3])
            rot_270 = torch.rot90(data, 3, [2, 3])
            hor_flip = torch.flip(data, [-1])
            ver_flip = torch.flip(data, [-2])
            data = torch.cat([data, rot_90, rot_180, rot_270, hor_flip, ver_flip], dim=0)  # 做了六种数据增强
            pred, _, cls_logits, att_maps, _ = model(data)
            pred = F.interpolate(pred, size=(args.w, args.h), mode='bicubic')
            result += pred[0:1] + torch.rot90(pred[1:2], 3, [2, 3]) + torch.rot90(pred[2:3], 2, [2, 3]) + torch.rot90(
                pred[3:4], 1, [2, 3]) + torch.flip(pred[4:5], [-1]) + torch.flip(pred[5:6], [-2])
            cls_result += cls_logits

        time1 = time.time()
        if args.print_time:
            print(f'time: {time1 - time0}')

        pred = result
        pred = torch.softmax(pred, dim=1).cpu().data.numpy()
        pred_arg = np.argmax(pred[0], axis=0)

        # y_pred
        y_true_f = val_mask.reshape(val_mask.shape[0] * val_mask.shape[1], order='F')
        y_pred_f = pred_arg.reshape(pred_arg.shape[0] * pred_arg.shape[1], order='F')

        intersection = np.float(np.sum(y_true_f * y_pred_f))
        dice.append((2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f)))
        sen.append(intersection / np.sum(y_true_f))
        intersection0 = np.float(np.sum((1 - y_true_f) * (1 - y_pred_f)))
        spe.append(intersection0 / np.sum(1 - y_true_f))
        acc.append(accuracy_score(y_true_f, y_pred_f))
        jac_score.append(intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection))

        cls_result = torch.mean(cls_result, 0)
        index = torch.argmax(cls_result)
        cls_prob = torch.softmax(cls_result, 0)

        di = (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))
        ja = intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection)
        f.write(
            f'Index: {image_index} Dice: {di} JA: {ja} CLS_PRED: {index} CLS_GT: {int(cls_gt)} CLS_PROB: {str(cls_prob)}\n')

        if cls_gt == 0:
            m_gt.append(0)
            s_gt.append(0)
            nevus[1] += 1
            if index == 0:
                m_pred_binary.append(0)
                m_pred_prob.append(cls_prob[1])
                s_pred_binary.append(0)
                s_pred_prob.append(cls_prob[2])
                nevus[0] += 1
                melanoma_correct += 1
                seborrheic_correct += 1
            elif index == 2:
                m_pred_binary.append(0)
                m_pred_prob.append(cls_prob[1])
                s_pred_binary.append(1)
                s_pred_prob.append(cls_prob[2])
                melanoma_correct += 1
            elif index == 1:
                m_pred_binary.append(1)
                m_pred_prob.append(cls_prob[1])
                s_pred_binary.append(0)
                s_pred_prob.append(cls_prob[2])
                seborrheic_correct += 1
        elif cls_gt == 1:
            m_gt.append(1)
            s_gt.append(0)
            melanoma[1] += 1
            if index == 1:
                m_pred_binary.append(1)
                m_pred_prob.append(cls_prob[1])
                s_pred_binary.append(0)
                s_pred_prob.append(cls_prob[2])
                melanoma[0] += 1
                seborrheic_correct += 1
                melanoma_correct += 1
            elif index == 0:
                m_pred_binary.append(0)
                m_pred_prob.append(cls_prob[1])
                s_pred_binary.append(0)
                s_pred_prob.append(cls_prob[2])
                seborrheic_correct += 1
            elif index == 2:
                m_pred_binary.append(0)
                m_pred_prob.append(cls_prob[1])
                s_pred_binary.append(1)
                s_pred_prob.append(cls_prob[2])
        elif cls_gt == 2:
            m_gt.append(0)
            s_gt.append(1)

            seborrheic[1] += 1
            if index == 2:
                m_pred_binary.append(0)
                m_pred_prob.append(cls_prob[1])
                s_pred_binary.append(1)
                s_pred_prob.append(cls_prob[2])
                seborrheic[0] += 1
                melanoma_correct += 1
                seborrheic_correct += 1
            elif index == 0:
                m_pred_binary.append(0)
                m_pred_prob.append(cls_prob[1])
                s_pred_binary.append(0)
                s_pred_prob.append(cls_prob[2])
                melanoma_correct += 1
            elif index == 1:
                m_pred_binary.append(1)
                m_pred_prob.append(cls_prob[1])
                s_pred_binary.append(0)
                s_pred_prob.append(cls_prob[2])


        index_str = str(int(image_index))
        while len(index_str) < 4:
            index_str = '0' + index_str

        if True:
            data = data[0].cpu()
            img = to_img(data)
            img.save(os.path.join(path, 'image/{}.png'.format(name[0].split('.')[0])))
            cv2.imwrite(os.path.join(path, 'image/{}_pred.png'.format(name[0].split('.')[0])), pred_arg * 255)
            cv2.imwrite(os.path.join(path, 'image/{}_gt.png'.format(name[0].split('.')[0])), val_mask * 255)
            f.write('{} {}\n'.format(index_str, (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))))

    print('Test nevus: {} / {}'.format(nevus[0], nevus[1]))
    print('Test melanoma: {} / {}'.format(melanoma[0], melanoma[1]))
    print('Test seborrheic: {} / {}'.format(seborrheic[0], seborrheic[1]))

    total_acc = (nevus[0] + melanoma[0] + seborrheic[0]) / (nevus[1] + melanoma[1] + seborrheic[1])
    m_acc = (melanoma_correct) / (nevus[1] + melanoma[1] + seborrheic[1])
    s_acc = (seborrheic_correct) / (nevus[1] + melanoma[1] + seborrheic[1])
    print('Total acc', total_acc)
    print('M acc', m_acc)
    print('S acc', s_acc)

    dic = {}

    acc, auc, AP, sens, spec = cla_evaluate(np.array(m_gt), np.array(m_pred_binary), np.array(m_pred_prob))
    if logging is not None:
        logging.info(f'macc: {acc} | mauc: {auc} | mAP: {AP} | msens: {sens} | mspec: {spec}')

    dic['macc'], dic['mauc'], dic['msens'], dic['mspec'] = acc, auc, sens, spec

    if not ph2:
        acc, auc, AP, sens, spec = cla_evaluate(np.array(s_gt), np.array(s_pred_binary), np.array(s_pred_prob))
        logging.info(f'sacc: {acc} | sauc: {auc} | sAP: {AP} | ssens: {sens} | sspec: {spec}')

    dic['sacc'], dic['sauc'], dic['ssens'], dic['sspec'] = acc, auc, sens, spec

    f.close()
    print('successfully record!')

    return np.array(acc), np.array(dice), np.array(sen), np.array(spe), np.array(
        jac_score), total_acc, m_acc, s_acc, dic


def val_mode_seg(valloader, model, path, epoch, test=False, visualize=False):
    dice = []
    sen = []
    spe = []
    acc = []
    jac_score = []

    if not os.path.exists(os.path.join(path, 'image')):
        os.mkdir(os.path.join(path, 'image'))
    f = open(os.path.join(path, 'image', 'record_dice.txt'), 'w+')
    to_img = transforms.ToPILImage()

    for index, batch in enumerate(valloader):
        if index % 50 == 0:
            print('Val {} / {}'.format(index, len(valloader)))

        data, mask, name = batch
        data = data.cuda()
        mask = mask[0].data.numpy()
        val_mask = np.int64(mask > 0)

        model.eval()

        with torch.no_grad():
            if test:
                rot_90 = torch.rot90(data, 1, [2, 3])
                rot_180 = torch.rot90(data, 2, [2, 3])
                rot_270 = torch.rot90(data, 3, [2, 3])
                hor_flip = torch.flip(data, [-1])
                ver_flip = torch.flip(data, [-2])
                data = torch.cat([data, rot_90, rot_180, rot_270, hor_flip, ver_flip], dim=0)  # 做了六种数据增强
                pred, _, _, att_maps, _ = model(data)
                pred = pred[0:1] + torch.rot90(pred[1:2], 3, [2, 3]) + torch.rot90(pred[2:3], 2, [2, 3]) + torch.rot90(
                    pred[3:4], 1, [2, 3]) + torch.flip(pred[4:5], [-1]) + torch.flip(pred[5:6], [-2])
            else:
                pred, _, _, _, _ = model(data)

        pred = torch.softmax(pred, dim=1).cpu().data.numpy()
        pred_arg = np.argmax(pred[0], axis=0)

        y_true_f = val_mask.reshape(val_mask.shape[0] * val_mask.shape[1], order='F')
        y_pred_f = pred_arg.reshape(pred_arg.shape[0] * pred_arg.shape[1], order='F')

        intersection = np.float(np.sum(y_true_f * y_pred_f))
        dice.append((2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f)))
        sen.append(intersection / np.sum(y_true_f))
        intersection0 = np.float(np.sum((1 - y_true_f) * (1 - y_pred_f)))
        spe.append(intersection0 / np.sum(1 - y_true_f))
        acc.append(accuracy_score(y_true_f, y_pred_f))
        jac_score.append(intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection))

        index_str = str(index)
        while len(index_str) < 4:
            index_str = '0' + index_str

        if visualize:
            data = data[0].cpu()
            img = to_img(data)
            img.save(os.path.join(path, 'image/{}.jpg'.format(index_str)))
            cv2.imwrite(os.path.join(path, 'image/{}_pred.jpg'.format(index_str)), pred_arg * 255)
            cv2.imwrite(os.path.join(path, 'image/{}_gt.jpg'.format(index_str)), val_mask * 255)
            f.write('{} {}\n'.format(index_str, (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))))

    return np.array(acc), np.array(dice), np.array(sen), np.array(spe), np.array(jac_score)