import os
from modeling.deeplab import *
from dataloaders.datasets import arbel
import glob
import time
import torch.backends.cudnn as cudnn

from IPython import display
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from dataloaders import make_data_loader
import argparse
from utils.metrics import Evaluator
from torch.autograd import Variable

## Load model in Pytorch

parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus pruning")
parser.add_argument('--backbone', type=str, default='resnet',
                    choices=['resnet', 'xception', 'drn', 'mobilenet'],
                    help='backbone name (default: resnet)')
parser.add_argument('--out-stride', type=int, default=16,
                    help='network output stride (default: 8)')
parser.add_argument('--dataset', type=str, default='pascal',
                    choices=['pascal', 'coco', 'cityscapes', 'arbel'],
                    help='dataset name (default: pascal)')
parser.add_argument('--workers', type=int, default=4,
                    metavar='N', help='dataloader threads')
parser.add_argument('--base-size', type=int, default=513,
                    help='base image size')
parser.add_argument('--crop-size', type=int, default=513,
                    help='crop image size')
parser.add_argument('--sync-bn', type=bool, default=None,
                    help='whether to use sync bn (default: auto)')
parser.add_argument('--freeze-bn', type=bool, default=False,
                    help='whether to freeze bn parameters (default: False)')
parser.add_argument('--loss-type', type=str, default='ce',
                    choices=['ce', 'focal'],
                    help='loss func type (default: ce)')
# training hyper params
parser.add_argument('--epochs', type=int, default=None, metavar='N',
                    help='number of epochs to train (default: auto)')
parser.add_argument('--start_epoch', type=int, default=0,
                    metavar='N', help='start epochs (default:0)')
parser.add_argument('--batch-size', type=int, default=None,
                    metavar='N', help='input batch size for \
                            training (default: auto)')
parser.add_argument('--test-batch-size', type=int, default=None,
                    metavar='N', help='input batch size for \
                            testing (default: auto)')
parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                    help='whether to use balanced weights (default: False)')
# optimizer params

parser.add_argument('--lr', type=float, default=None, metavar='LR',
                    help='learning rate (default: auto)')
parser.add_argument('--lr-scheduler', type=str, default='poly',
                    choices=['poly', 'step', 'cos'],
                    help='lr scheduler mode: (default: poly)')
parser.add_argument('--momentum', type=float, default=0.9,
                    metavar='M', help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4,
                    metavar='M', help='w-decay (default: 5e-4)')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help='whether use nesterov (default: False)')
parser.add_argument('--use_balanced_weights', action='store_true', default=False,
                    help='whether to use balanced weights for training (default: False)')
# cuda, seed and logging
parser.add_argument('--no-cuda', action='store_true', default=
                    False, help='disables CUDA training')
parser.add_argument('--gpu-ids', type=str, default='0',
                    help='use which gpu to train, must be a \
                    comma-separated list of integers only (default=0)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# checking point
parser.add_argument('--resume', type=str, default=None,
                    help='put the path to resuming file if needed')
parser.add_argument('--checkname', type=str, default=None,
                    help='set the checkpoint name')
# finetuning pre-trained models
parser.add_argument('--ft', action='store_true', default=False,
                    help='finetuning on a different dataset')
# evaluation option
parser.add_argument('--eval-interval', type=int, default=1,
                    help='evaluuation interval (default: 1)')
parser.add_argument('--no-val', action='store_true', default=False,
                    help='skip validation during training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Define Dataloader
kwargs = {'num_workers': args.workers, 'pin_memory': True}
train_loader, val_loader, test_data_loader, nclass = make_data_loader(args, **kwargs)

evaluator = Evaluator(nclass)

_CHECKPOINT_PATH = '/home/ido/Deep/Pytorch/pytorch_deeplab-xception/run/arbel/deeplab-xception/experiment_8/checkpoint_fp16_02_os8.pth'
_PARAMS_PATH = '/home/ido/Deep/Pytorch/pytorch_deeplab-xception/run/arbel/deeplab-xception/experiment_8/parameters.txt'
INPUT_SIZE_HEIGHT = 720
INPUT_SIZE_WIDTH = 1280 # was 1280

cudnn.benchmark = True
cudnn.enabled = True

file = open(_PARAMS_PATH, 'r')
parameters = {}
for line in file:

   k, v = line.split(':')
   parameters[k] = v.rstrip()

# models = sorted(glob.glob('/home/ido/Deep/Pytorch/pytorch_deeplab-xception/pruned_models/*.pth'))
# total_infer_times = [0 for _ in models]
# model_acc = [0 for _ in models]
# model_acc_class = [0 for _ in models]
# model_miou = [0 for _ in models]

def list_images(folder, pattern='*', ext='bmp'):
    """List the images in a specified folder by pattern and extension

    Args:
        folder (str): folder containing the images to list
        pattern (str, optional): a bash-like pattern of the files to select
                                 defaults to * (everything)
        ext(str, optional): the image extension (defaults to png)

    Returns:
        str list: list of (filenames) images matching the pattern in the folder
    """
    filenames = sorted(glob.glob(folder + pattern + '.' + ext))
    return filenames


def save_segmentation(image_name, seg_map):
    seg_image = colormap[seg_map.squeeze()].astype(np.uint8)
    segmentation = Image.fromarray(seg_image)
    segmentation = segmentation.resize(size=(1280, 960), resample=Image.NEAREST)
    newfilename = os.path.join(os.path.dirname(image_name),
                               os.path.splitext(os.path.basename(image_name))[0] + '_ID_PYTORCH.png')
    segmentation.save(newfilename, "PNG")

    # seg_image_crf = colormap[seg_map_crf].astype(np.uint8)
    # segmentation_crf = Image.fromarray(seg_image_crf)
    # segmentation_crf = segmentation_crf.resize(size=(1280, 960), resample=Image.NEAREST)
    # newfilename_crf = os.path.join(os.path.dirname(image_name),
    #                            os.path.splitext(os.path.basename(image_name))[0] + '_ID_CRF.png')
    # segmentation_crf.save(newfilename_crf, "PNG")


IMAGE_DIR = '/home/ido/Deep/SegmentationDatasets/elyakim/drive3/'
images = list_images(IMAGE_DIR, ext='jpg')

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

target_size = (int(INPUT_SIZE_WIDTH), int(INPUT_SIZE_HEIGHT))

LABEL_NAMES = np.asarray([
    'Terrain', 'Unpaved route', 'Terrain route - Metal palette', 'Tree - trunk', 'Tree - leaf', 'Rocks', 'Large Shrubs',
    'Low Vegetation', 'Wire Fance', 'Background(sky)', 'Person', 'Vehicle', 'building', 'Paved Road', 'Misc'
    , 'Ignore', 'Ignore', 'Ignore', 'Ignore'
])

labels = []

colormap = np.array(
    [[149, 129, 107],  # Terrain
     [198, 186, 173],  # Terrain route
     [77, 69, 59],  # Terrain route - Metal palette
     [144, 100, 0],  # Tree - trunk
     [103, 128, 88],  # Tree - leaf
     [128, 0, 255],  # Rocks
     [158, 217, 92],  # Large Shrubs
     [200, 217, 92],  # Low Vegetation
     [217, 158, 93],  # Wire Fance
     [0, 0, 128],  # background(sky)
     [160, 5, 5],  # Person
     [53, 133, 193],  # Vehicle
     [0, 0, 0],  # building
     [30, 30, 30],  # paved road
     [255, 255, 255],  # Misc.
     [0, 255, 255],  # empty
     [0, 255, 255],  # empty
     [0, 255, 255],  # empty
     [0, 255, 255],  # empty
     ],
    dtype=np.float32)

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = colormap[FULL_LABEL_MAP]

def list_images(folder, pattern='*', ext='bmp'):
    """List the images in a specified folder by pattern and extension

    Args:
        folder (str): folder containing the images to list
        pattern (str, optional): a bash-like pattern of the files to select
                                 defaults to * (everything)
        ext(str, optional): the image extension (defaults to png)

    Returns:
        str list: list of (filenames) images matching the pattern in the folder
    """
    filenames = sorted(glob.glob(folder + pattern + '.' + ext))
    return filenames


def save_segmentation(image_name, seg_map):
    seg_image = colormap[seg_map.squeeze()].astype(np.uint8)
    segmentation = Image.fromarray(seg_image)
    segmentation = segmentation.resize(size=(1280, 960), resample=Image.NEAREST)
    newfilename = os.path.join(os.path.dirname(image_name),
                               os.path.splitext(os.path.basename(image_name))[0] + '_ID_PYTORCH.png')
    segmentation.save(newfilename, "PNG")

    # seg_image_crf = colormap[seg_map_crf].astype(np.uint8)
    # segmentation_crf = Image.fromarray(seg_image_crf)
    # segmentation_crf = segmentation_crf.resize(size=(1280, 960), resample=Image.NEAREST)
    # newfilename_crf = os.path.join(os.path.dirname(image_name),
    #                            os.path.splitext(os.path.basename(image_name))[0] + '_ID_CRF.png')
    # segmentation_crf.save(newfilename_crf, "PNG")


# for iter, _CHECKPOINT_PATH in enumerate(models):
#
#     if not os.path.isfile(_CHECKPOINT_PATH):
#         raise RuntimeError("=> no checkpoint found at '{}'".format(_CHECKPOINT_PATH))
#     f = _CHECKPOINT_PATH.split('_')
#     f = f[1].split(':')
#
#     curr_iter = int(f[1])
    # checkpoint = torch.load(_CHECKPOINT_PATH)

model = DeepLab(num_classes=arbel.ArbelSegmentation.NUM_CLASSES,
                backbone=parameters['backbone'],
                output_stride=int(parameters['out_stride']),
                sync_bn=True,
                freeze_bn=False)

checkpoint = torch.load(_CHECKPOINT_PATH)

model.load_state_dict(checkpoint['state_dict'])

# model = torch.load(_CHECKPOINT_PATH)

model = model.eval()
model = model.cuda()
torch.set_grad_enabled(False)

print("=> loaded checkpoint '{}')".format(_CHECKPOINT_PATH))
# print("=> curr iteration is {}".format(curr_iter))

## Run on sample images for timings

infer_times = []
for n, image_name in enumerate(images):
    image_path = os.path.join(IMAGE_DIR, image_name)
    if n>100:
        break

    image = Image.open(image_path).convert('RGB').resize(target_size, Image.ANTIALIAS)
    print('running deeplab on image %s...' % image_name)
    start_tot = time.time()
    image_t = trans(image).cuda()
    image_t = image_t.unsqueeze(0)
    image_t_batch = torch.cat([image_t ,image_t ,image_t ,image_t],0)
    with torch.no_grad():
        pred = model(image_t_batch).squeeze().cpu().numpy()
    pred = np.argmax(pred, axis=0)

    # save_segmentation(image_name, pred)

    end_tot = time.time()- start_tot
    infer_times.append(end_tot)
    print("Classification took {} ms".format(end_tot * 1000))

# print("Mean classification time after pruning iteration {} is {} ms".format(iter,1000*np.mean(infer_times)))
# total_infer_times[curr_iter] = 1000*np.mean(infer_times)

## Run on Test set for acc
# print('Testing model:')
evaluator.reset()

for i, sample in enumerate(val_loader):
    batch, target = sample['image'], sample['label']
    if args.cuda:
        batch = batch.cuda()
    input = Variable(batch)
    output = model(input)
    pred = output.data.cpu().numpy()
    target = target.cpu().numpy()
    pred = np.argmax(pred, axis=1)
    # Add batch sample into evaluator
    evaluator.add_batch(target, pred)

Acc = evaluator.Pixel_Accuracy()*100
Acc_class = evaluator.Pixel_Accuracy_Class()*100
mIoU = evaluator.Mean_Intersection_over_Union()*100
FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()*100
print("Acc:{0:.3f}, Acc_class:{1:.3f}, mIoU:{2:.3f}, fwIoU: {3:.3f}".format(Acc, Acc_class, mIoU, FWIoU))

# model_acc[curr_iter] = Acc
# model_acc_class[curr_iter] = Acc_class
# model_miou[curr_iter] = mIoU

title = 'Inference Time of Pruned Model'
plt.plot(total_infer_times, '-b', label='infer_times')
plt.plot(model_acc, '-r', label='accuracy')
plt.plot(model_acc_class, '-g', label='accuracy_class')
plt.plot(model_miou, '-k', label='miou')
plt.legend(loc='upper left')
plt.xlabel("pruning iterations")
plt.title(title)
plt.savefig(title+".png")
plt.show()




