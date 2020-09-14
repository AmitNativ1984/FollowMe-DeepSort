import onnx
import sys, os
sys.path.append('/home/ido/Deep/Pytorch/onnx-tensorrt')
from pathlib import Path
import numpy as np
# import tensorrt as trt
import onnx_tensorrt.backend as backend
import time
import torch
import cv2
import glob
from PIL import Image

def LoadImage(image_path,target_size):

    # Read image
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size, interpolation=cv2.cv2.INTER_CUBIC)
    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0

    return np.expand_dims(img, axis=0)

if __name__ == '__main__':

    dummy_input = torch.randn(4, 3, 720, 1280, device='cuda')
    img_size = dummy_input.size()
    save_txt = False
    save_images = False
    load_engine = False
    precision = 'fp16'  # 'fp16' \ 'fp32' \ 'int8'

    INPUT_SIZE_HEIGHT = 720
    INPUT_SIZE_WIDTH = 1280  # was 1280

    # Set Dataloader
    # dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors

    LABEL_NAMES = np.asarray([
        'Terrain', 'Unpaved route', 'Terrain route - Metal palette', 'Tree - trunk', 'Tree - leaf', 'Rocks',
        'Large Shrubs',
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

    # load model

    model = onnx.load("deeplab_pruned.onnx")
    engine_path = 'deeplab_pruned_' + precision + '.engine'

    engine = backend.prepare(model, device='CUDA:0',serialize_engine=False,precision=precision,max_batch_size=4, load_engine=load_engine,engine_path=engine_path)


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


    ## Run on sample images

    IMAGE_DIR = '/home/ido/Deep/SegmentationDatasets/elyakim/drive3/'
    images = list_images(IMAGE_DIR, ext='jpg')

    target_size = (int(INPUT_SIZE_WIDTH), int(INPUT_SIZE_HEIGHT))

    mtimes = []

    for i,image_name in enumerate(images):
        image_path = os.path.join(IMAGE_DIR, image_name)

        t = time.time()

        # Get detections
        img = LoadImage(image_path,target_size)

        img = np.concatenate((img, img, img, img), axis=0)
        output_data = engine.run(img)[0]

        # post pocessing

        pred = output_data.squeeze()
        pred = np.argmax(pred, axis=0)

        dt = time.time() - t

        mtimes.append(dt)
        print('Done. total time inc. postprocess is (%.3fs)' % dt)

        if i>200:
            break;

    print('Average proccesing time at {} was {}.'.format(precision, np.mean(mtimes)))