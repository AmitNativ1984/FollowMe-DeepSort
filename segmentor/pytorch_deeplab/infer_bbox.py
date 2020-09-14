import os
from .modeling.deeplab import *
import glob
import time
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable


cudnn.benchmark = True
cudnn.enabled = True

class InferBoundingBox(object):
    def __init__(self, cfg):
        self.CHECKPOINT_PATH = cfg.CHECKPOINT_PATH
        self.PARAMS_PATH = cfg.PARAMS_PATH
        self.INPUT_SIZE_HEIGHT = cfg.INPUT_SIZE_HEIGHT
        self.INPUT_SIZE_WIDTH = cfg.INPUT_SIZE_WIDTH

        self.target_size = (int(self.INPUT_SIZE_WIDTH), int(self.INPUT_SIZE_HEIGHT))

        file = open(self.PARAMS_PATH, 'r')
        self.parameters = {}
        for line in file:
           k, v = line.split(':')
           self.parameters[k] = v.rstrip()


    @staticmethod
    def preprocess():
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return trans

    def list_images(self, folder, pattern='*', ext='bmp'):
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


    def save_segmentation(self, image_name, seg_map):
        seg_image = self.colormap[seg_map.squeeze()].astype(np.uint8)
        segmentation = Image.fromarray(seg_image)
        segmentation = segmentation.resize(size=(1280, 960), resample=Image.NEAREST)
        newfilename = os.path.join(os.path.dirname(image_name),
                                   os.path.splitext(os.path.basename(image_name))[0] + '_ID_PYTORCH.png')
        segmentation.save(newfilename, "PNG")



    class labels_decoder:

        def __init__(self):

            self.LABEL_NAMES = np.asarray([
                'Terrain', 'Unpaved route', 'Terrain route - Metal palette', 'Tree - trunk', 'Tree - leaf', 'Rocks', 'Large Shrubs',
                'Low Vegetation', 'Wire Fance', 'Background(sky)', 'Person', 'Vehicle', 'building', 'Paved Road', 'Misc'
                , 'Ignore', 'Ignore', 'Ignore', 'Ignore'
            ])

            self.colormap = np.array(
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

            self.label_names = [k for k in self.LABEL_NAMES]
            self.labels_id = [v[0] for v in self.FULL_LABELS_MAP()]

            self.cls_id_dict = dict(zip(self.label_names, self.labels_id))

        def FULL_LABELS_MAP(self):
            return np.arange(len(self.LABEL_NAMES)).reshape(len(self.LABEL_NAMES), 1)

        def FULL_COLOR_MAP(self):
            return self.colormap[self.FULL_LABEL_MAP]


    def define_model(self):
        self.model = DeepLab(num_classes=len(self.labels_decoder().FULL_LABELS_MAP()),
                        backbone=self.parameters['backbone'],
                        output_stride=int(self.parameters['out_stride']),
                        sync_bn=True,
                        freeze_bn=False)
        return self.model

    def load_model(self):
        checkpoint = torch.load(self.CHECKPOINT_PATH)

        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.model.cuda()
        # torch.set_grad_enabled(False)

        print("=> loaded checkpoint '{}')".format(self.CHECKPOINT_PATH))

        return self.model

    def infer(self, batch):
        with torch.no_grad():
            pred = self.model(batch).cpu().numpy()
            pred = np.argmax(pred, axis=1)

        return pred






