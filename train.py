import sys
import torch
import argparse
import warnings
warnings.filterwarnings("ignore")

from engine import trainer

## get config
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--data', type=str, default='isic')
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--random_resize', action='store_true')
parser.add_argument('--w', type=int, default=224)
parser.add_argument('--h', type=int, default=224)
parser.add_argument('--base_lr', type=float, default=0.00001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--power', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--output_class', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--label_data', type=int, default=2000)
parser.add_argument('--unlabel_data', type=int, default=1320)
parser.add_argument('--max_iter', type=int, default=40000)
parser.add_argument('--seg_only', action='store_true')
parser.add_argument('--consis_weight', type=float, default=5.0)
parser.add_argument('--cls_weight', type=float, default=0.25)
parser.add_argument('--swin', action='store_true')
parser.add_argument('--pretrain', type=str, default='')
parser.add_argument('--cls_finetune', action='store_true')
parser.add_argument('--adamw', action='store_true')
parser.add_argument('--warmup', action='store_true')
parser.add_argument('--label_smooth', type=bool, default=True)
parser.add_argument('--optimize_separate', action='store_true')
parser.add_argument('--section_lr', action='store_true')
parser.add_argument('--cls_late', type=int, default=-1)
parser.add_argument('--ph2_test',action='store_true')
parser.add_argument('--blur_ls', type=int, default=0)
parser.add_argument('--mixup', type=float, default=0.0)
parser.add_argument('--att_loss', type=float, default=0.0)
parser.add_argument('--cs_loss', type=float, default=0.0)
parser.add_argument('--ac_loss', type=float, default=0.0)
parser.add_argument('--ds', type=float, default=0.0)
parser.add_argument('--weight_cls', action='store_true')
parser.add_argument('--print_time', action='store_true')
parser.add_argument('--train_mode', default='seg+cls+dual',
                    choices=['seg_only', 'cls_only', 'seg+cls', 'seg+dual', 'seg+cls+dual'])
args = parser.parse_args()

sys.path.append('./model')
sys.path.append('./model/networks')
from model.train import net as model
model = torch.nn.DataParallel(model)

if __name__ == '__main__':
    trainer(args, model)
