import sys
import time
import logging
import numpy as np
from tqdm import tqdm
from dataset import get_data

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from utils.optimizer import get_optim
from utils.prepare import clean, backup, get_month_day
from utils.metrics import compute_sdf, dice_loss
from utils.learning_rate import adjust_learning_rate, get_current_consistency_weight
from utils.runtime import random_resize, get_cls_label, name_list_to_cls_label
from utils.validate import val_mode_seg, val_mode_seg_multi_scale
from utils.loss import MyLabelSmoothingLoss


# get loss function
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()
ce_loss = nn.CrossEntropyLoss()
weight_ce_loss_for_class_m = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 1.0, 1.0])).cuda()  # 第1类的权重调高
label_smooth_ce_loss = MyLabelSmoothingLoss(classes=2, smoothing=0.0)

def init_program(args):
    clean(args.exp_name)
    backup(args.exp_name)

    writer = SummaryWriter('./history/{}/'.format(get_month_day()) + args.exp_name)
    logging.basicConfig(filename="./history/{}/{}/log.txt".format(get_month_day(), args.exp_name), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    cudnn.enabled = True
    cudnn.benchmark = True

    return writer


# a function that trains the model
def trainer(args, model):
    writer = init_program(args)

    if args.pretrain != '':
        pretrained_dict = torch.load(args.pretrain)
        model.load_state_dict(pretrained_dict)
        print('Load Pre-trained Model Finish!')

    optimizer = get_optim(args, model.parameters())
    dataloader_group = get_data(args)
    trainloader, valloader, testloader = dataloader_group['trainloader'], dataloader_group['valloader'], \
                                         dataloader_group['testloader']

    label_dic = get_cls_label(args)

    best_val_jac = 0
    best_cls = 0
    iter_num = 0
    for epoch in range(args.epoch):
        for i_iter, batch in tqdm(enumerate(trainloader)):

            test_flag = False
            if test_flag:
                print('*** Only Test ***')
                break
            test_flag = False

            model.train()
            images, labels, name = batch

            images, labels = random_resize(args, images, labels)

            images = images.cuda()
            labels = labels.cuda().squeeze(1)

            optimizer.zero_grad()
            lr = adjust_learning_rate(args, optimizer, iter_num)
            preds, dt_preds, cls_logits, attention_maps, ds_mask_logits = model(images)
            cls_label = name_list_to_cls_label(name, label_dic)

            ### attention loss
            last_att_maps = attention_maps[0]
            last_att_maps = last_att_maps[:int(args.batch_size / 2), ...]
            last_att_maps = torch.mean(last_att_maps, 1)[:, 0, 1:].view(int(args.batch_size / 2), 14, 14)  # 4 196
            labels_resize = torch.nn.functional.interpolate(labels.unsqueeze(1), size=(14, 14), mode='nearest')[
                            :int(args.batch_size / 2), ...]
            labels_resize = labels_resize.squeeze(1)
            attention_loss = ((1 - labels_resize) * last_att_maps).sum()

            ### cls seg consistency loss
            last_att_maps = attention_maps[-1]
            last_att_maps = torch.mean(last_att_maps, 1)[:, 0, 1:]  # 8 196
            last_att_maps = last_att_maps * (1 / (last_att_maps.sum() + 1e-3))
            preds_resize = torch.nn.functional.interpolate(preds[:, 1, ...].unsqueeze(1), size=(14, 14),
                                                           mode='bilinear')  # 8 14 14
            preds_resize = preds_resize.squeeze(1).view(args.batch_size, -1)  # 8 196
            preds_resize = torch.softmax(preds_resize, 1)
            cs_loss = ((last_att_maps - preds_resize) * (last_att_maps - preds_resize)).sum()

            ### activate consistent loss
            preds_softmax = torch.softmax(preds, 1)
            preds_resize = torch.nn.functional.interpolate(preds_softmax[:, 1, ...].unsqueeze(1), size=(14, 14),
                                                           mode='bilinear')  # 8 14 14
            preds_resize = preds_resize.squeeze(1).view(args.batch_size, -1)  # 8 196
            # preds_resize = torch.softmax(preds_resize, 1)
            ac_loss = - (last_att_maps * preds_resize).sum()

            ### deep supervision loss
            label_resize = torch.nn.functional.interpolate(labels.unsqueeze(1), size=(14, 14),
                                                           mode='nearest')  # 8 14 14
            label_resize = label_resize.squeeze(1)
            deep_loss_seg = ce_loss(
                ds_mask_logits[:int(args.batch_size / 2), ...], label_resize[:int(args.batch_size / 2), ...].long())

            ### sdf seg dice loss
            with torch.no_grad():
                gt_dis = compute_sdf(labels[:int(args.batch_size / 2)].cpu(
                ).numpy(), dt_preds[:int(args.batch_size / 2), 0, ...].shape)
                gt_dis = torch.from_numpy(gt_dis).float().cuda()
            loss_sdf = mse_loss(dt_preds[:int(args.batch_size / 2), 0, ...], gt_dis)
            loss_seg = ce_loss(
                preds[:int(args.batch_size / 2), ...], labels[:int(args.batch_size / 2), ...].long())
            loss_label_smooth_ce_seg = label_smooth_ce_loss(preds[:int(args.batch_size / 2), ...],
                                                            labels[:int(args.batch_size / 2), ...].long())
            loss_seg_dice = dice_loss(
                preds[:int(args.batch_size / 2), 0, :, :], labels[:int(args.batch_size / 2)] == 0)
            dis_to_mask = torch.sigmoid(-1500 * dt_preds)

            ### cls loss
            cls_loss = ce_loss(cls_logits, cls_label)

            ### consistency loss
            consistency_loss = torch.mean(
                (torch.cat((1 - dis_to_mask, dis_to_mask), 1) - preds) ** 2) 

            consistency_weight = get_current_consistency_weight(epoch)

            ### calculate the total loss function
            if args.train_mode == 'seg_only':
                loss = loss_label_smooth_ce_seg 
            elif args.train_mode == 'cls_only':
                loss = cls_loss
            elif args.train_mode == 'seg+cls':
                if iter_num <= args.cls_late:
                    loss = loss_label_smooth_ce_seg
                else:
                    loss = loss_label_smooth_ce_seg + args.cls_weight * cls_loss
            elif args.train_mode == 'seg+dual':
                loss = loss_label_smooth_ce_seg + consistency_weight * consistency_loss + args.consis_weight * loss_sdf
            elif args.train_mode == 'seg+cls+dual':
                if iter_num <= args.cls_late:
                    loss = loss_label_smooth_ce_seg + consistency_weight * consistency_loss + args.consis_weight * loss_sdf
                else:
                    loss = loss_label_smooth_ce_seg + consistency_weight * consistency_loss + args.consis_weight * loss_sdf + args.cls_weight * cls_loss

            if args.att_loss != 0:
                loss += args.att_loss * attention_loss

            if args.cs_loss != 0:
                loss += args.cs_loss * cs_loss

            if args.ds != 0:
                loss += args.ds * deep_loss_seg

            if args.ac_loss != 0:
                loss += args.ac_loss * ac_loss

            if torch.isnan(loss):
                continue

            iter_num += 1

            loss.backward()
            optimizer.step()

            writer.add_scalar('lr', lr, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss_label_smooth_ce_seg', loss_label_smooth_ce_seg, iter_num)
            writer.add_scalar('loss/loss_hausdorff', loss_sdf, iter_num)
            writer.add_scalar('loss/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('loss/cls_loss', cls_loss, iter_num)
            writer.add_scalar('loss/att_loss', attention_loss, iter_num)
            writer.add_scalar('loss/cs_loss', cs_loss, iter_num)
            writer.add_scalar('loss/ds_loss', deep_loss_seg, iter_num)
            writer.add_scalar('loss/ac_loss', deep_loss_seg, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_consis: %f, loss_seg_ce: %f,  loss_haus: %f, loss_dice: %f, loss_cls: %f, att_loss: %f, cs_loss: %f, ds_loss: %f, ac_loss: %f' %
                (iter_num, loss.item(), consistency_loss.item(), loss_seg.item(), loss_sdf.item(), loss_seg_dice.item(),
                 cls_loss.item(),
                 attention_loss.item(), cs_loss.item(), deep_loss_seg.item(), ac_loss.item()))

            if iter_num % 20 == 0:
                image = images[0, :, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/image', image, iter_num)

                outputs = torch.argmax(torch.softmax(
                    preds, dim=1), dim=1, keepdim=True)
                writer.add_image('train/mask_pred',
                                 outputs[0, ...] * 200, iter_num)

                writer.add_image('train/dt_to_mask',
                                 dis_to_mask[0, ...] * 200, iter_num)

                # print(labels[0,...].shape)
                labs = labels[0, ...].unsqueeze(0) * 200
                writer.add_image('train/mask_gt', labs, iter_num)

                dis = gt_dis.unsqueeze(1)
                dis = dis[0, :, :, :] * 200
                dis = (dis - dis.min()) / (dis.max() - dis.min())
                writer.add_image('train/dt_gt', dis, iter_num)

                dt_pred = dt_preds[0, :, :, :] * 200
                dt_pred = (dt_pred - dt_pred.min()) / (dt_pred.max() - dt_pred.min())
                writer.add_image('train/dt_pred', dt_pred, iter_num)

                ##### Unlabel
                image = images[4, :, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Unlabel_image', image, iter_num)

                outputs = torch.argmax(torch.softmax(
                    preds, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Unlabel_mask_pred',
                                 outputs[4, ...] * 200, iter_num)

                dt_pred = dt_preds[4, :, :, :] * 200
                dt_pred = (dt_pred - dt_pred.min()) / (dt_pred.max() - dt_pred.min())
                writer.add_image('train/Unlabel_dt_pred', dt_pred, iter_num)

        ############# Start the validation
        if not test_flag:
            with torch.no_grad():
                print('start val!')
                [vacc, vdice, vsen, vspe, vjac_score] = val_mode_seg(valloader, model, './history/{}/'.format(
                    get_month_day()) + args.exp_name, epoch)
                logging.info("val%d: vacc=%f, vdice=%f, vsensitivity=%f, vspecifity=%f, vjac=%f \n" % \
                             (epoch, np.nanmean(vacc), np.nanmean(vdice), np.nanmean(vsen), np.nanmean(vspe),
                              np.nanmean(vjac_score)))

                writer.add_scalar('val/vacc', np.nanmean(vacc), epoch)
                writer.add_scalar('val/vdice', np.nanmean(vdice), epoch)
                writer.add_scalar('val/vsen', np.nanmean(vsen), epoch)
                writer.add_scalar('val/vspe', np.nanmean(vspe), epoch)
                writer.add_scalar('val/vjac_score', np.nanmean(vjac_score), epoch)

        with torch.no_grad():
            print('start test!')

            [vacc, vdice, vsen, vspe, vjac_score, total_acc, m_acc, s_acc, dic] = val_mode_seg_multi_scale(args,
                                                                                                           testloader,
                                                                                                           model,
                                                                                                           './history/{}/'.format(
                                                                                                               get_month_day()) + args.exp_name,
                                                                                                           test=True,
                                                                                                           ph2=args.ph2_test,
                                                                                                           logging=logging,
                                                                                                           cls_dic=label_dic)
            logging.info("test%d: tacc=%f, tdice=%f, tsensitivity=%f, tspecifity=%f, tjac=%f \n" % \
                         (epoch, np.nanmean(vacc), np.nanmean(vdice), np.nanmean(vsen), np.nanmean(vspe),
                          np.nanmean(vjac_score)))
            logging.info('cls_acc=%f, m_acc=%f, s_acc=%f' % (total_acc, m_acc, s_acc))

            # ############# Plot val curve
            # val_jac.append(np.nanmean(vjac_score))
            become_best_flag = False
            if best_val_jac < np.nanmean(vjac_score):
                best_val_jac = np.nanmean(vjac_score)
                become_best_flag = True

            become_cls_best = False
            if m_acc + s_acc >= best_cls:
                best_cls = m_acc + s_acc
                become_cls_best = True

            # if epoch % 5 == 0:
            if become_best_flag:
                # torch.save(model.state_dict(), path + 'CoarseSN_e' + str(epoch) + '.pth')
                torch.save(model.state_dict(),
                           './history/{}/'.format(get_month_day()) + args.exp_name + '/best_model.pth')

            if become_cls_best:
                # torch.save(model.state_dict(), path + 'CoarseSN_e' + str(epoch) + '.pth')
                torch.save(model.state_dict(),
                           './history/{}/'.format(get_month_day()) + args.exp_name + '/best_cls_model.pth')

            if (epoch + 1) % 20 == 0:
                torch.save(model.state_dict(),
                           './history/{}/'.format(get_month_day()) + args.exp_name + '/last_epoch.pth')

            writer.add_scalar('test/tacc', np.nanmean(vacc), epoch)
            writer.add_scalar('test/tdice', np.nanmean(vdice), epoch)
            writer.add_scalar('test/tsen', np.nanmean(vsen), epoch)
            writer.add_scalar('test/tspe', np.nanmean(vspe), epoch)
            writer.add_scalar('test/tjac_score', np.nanmean(vjac_score), epoch)

            writer.add_scalar('test/cls_acc', total_acc, epoch)
            writer.add_scalar('test/cls_macc', m_acc, epoch)
            writer.add_scalar('test/cls_sacc', s_acc, epoch)

            writer.add_scalar('test/cls_mauc', dic['mauc'], epoch)
            writer.add_scalar('test/cls_msen', dic['msens'], epoch)
            writer.add_scalar('test/cls_mspec', dic['mspec'], epoch)

            writer.add_scalar('test/cls_sauc', dic['sauc'], epoch)
            writer.add_scalar('test/cls_ssen', dic['ssens'], epoch)
            writer.add_scalar('test/cls_sspec', dic['sspec'], epoch)

            if test_flag:
                print('Finish Testing')
                exit(0)
