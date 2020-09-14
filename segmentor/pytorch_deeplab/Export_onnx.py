import os
from dataloaders.datasets import arbel
import torch.backends.cudnn as cudnn
from modeling.deeplab import *
from torch.onnx import OperatorExportTypes
import tensorrt

print(tensorrt.__version__)


## Load model in Pytorch

# _CHECKPOINT_PATH = '/home/ido/Deep/Pytorch/pytorch_deeplab-xception/run/arbel/deeplab-xception/experiment_8/checkpoint_fp16_02_os8.pth'
# _PARAMS_PATH = '/home/ido/Deep/Pytorch/pytorch_deeplab-xception/run/arbel/deeplab-xception/experiment_8/parameters.txt'
INPUT_SIZE_HEIGHT = 720
INPUT_SIZE_WIDTH = 1280 # was 1280

cudnn.benchmark = True
cudnn.enabled = True

# file = open(_PARAMS_PATH, 'r')
# parameters = {}
# for line in file:
#
#    k, v = line.split(':')
#    parameters[k] = v.rstrip()
#
#
# # Define network
# model = DeepLab(num_classes=arbel.ArbelSegmentation.NUM_CLASSES,
#                 backbone=parameters['backbone'],
#                 output_stride=int(parameters['out_stride']),
#                 sync_bn=True,
#                 freeze_bn=False)
#
# if not os.path.isfile(_CHECKPOINT_PATH):
#     raise RuntimeError("=> no checkpoint found at '{}'".format(_CHECKPOINT_PATH))
# checkpoint = torch.load(_CHECKPOINT_PATH)

model = torch.load('pruned_models/iter:250_time:49.39_Acc:90.766_Acc_class:78.611_mIoU:69.711_fwIoU:83.831.pth')
# model.load_state_dict(checkpoint['state_dict'])
model.eval()
model = model.cuda()

dummy_input = torch.randn(1, 3, 720, 1280, device='cuda')

# summary(model, dummy_input.size());

input_names = [ "input_1" ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "deeplab_pruned.onnx", verbose=True, input_names=input_names, output_names=output_names)#,operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK)
