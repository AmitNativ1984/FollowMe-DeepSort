import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tqdm
import time
import cv2
from torch.utils.data import Dataset, DataLoader
import utm
from albumentations import (BboxParams,
                            HorizontalFlip,
                            Resize,
                            RandomCrop,
                            RandomScale,
                            PadIfNeeded,
                            ShiftScaleRotate,
                            Blur,
                            MotionBlur,
                            Normalize,
                            RandomBrightnessContrast,
                            OneOf,
                            Compose)

class ProbotSensorsDataset(Dataset):

    """
        Load sensor data from probot recordings.
        This if for inference only, not training!!!!
    """


    def __init__(self, args):
        self.args = args


        all_files = [os.path.join(args.data, f) for f in os.listdir(self.args.data)]
        # retriveing all image files
        self.cam0_files = [imgfile for imgfile in all_files if imgfile.endswith(".bmp") and "drive_cam0" in imgfile]

        # sorting all drivecam0_image_files by name:
        self.cam0_files.sort(key=lambda f: int(os.path.splitext(f)[0].split("_")[-1]))
        self.imts_files = [f.replace(".bmp", ".imts").replace("_cam0", "") for f in self.cam0_files]
        self.ldr0_files = [f.replace(".bmp", ".ldr").replace("_cam0", "_lidar0") for f in self.cam0_files]
        self.ldr1_files = [f.replace(".bmp", ".ldr").replace("_cam0", "_lidar1") for f in self.cam0_files]
        self.mtigv_files = [f.replace(".bmp", ".mtigv").replace("_cam0", "_telemetry0") for f in self.cam0_files]

    def __len__(self):
        return len(self.cam0_files)

    def __getitem__(self, index):
        cam0 = self.cam0_files[index].rstrip()
        imts = self.imts_files[index].rstrip()
        ldr0 = self.ldr0_files[index].rstrip()
        ldr1 = self.ldr1_files[index].rstrip()
        mtigv = self.mtigv_files[index].rstrip()

        # reading image:
        img = np.asarray(Image.open(cam0).convert('RGB')).astype(np.uint8)

        # mtigv (telemetry data)
        telemetry = self.get_telemtry_from_file(mtigv)

        sample = {
            "image": img,
            "telemetry": telemetry,
            "ldr0": ldr0,
            "ldr1": ldr1,
            "imts": imts
        }

        return sample

    def transform_tr(self, sample):
        """ data augmentation for training """
        img = sample["image"]
        bboxes = sample["bboxes"]

        imgH = img.shape[0]
        imgW = img.shape[1]

        if bboxes.size == 0:
            bboxes = np.array([[0.1, 0.1, 0.1, 0.1, 0.0]])  # this is just a dummy - all values must be inside (0,1)

        annotations = {'image': img, 'bboxes': bboxes}

        random_scale = np.random.randint(8, 11)/10

        transforms = ([# padding image in case is too small
                      PadIfNeeded(min_height=self.args.img_size[0], min_width=self.args.img_size[1],
                                  border_mode=cv2.BORDER_REPLICATE,
                                  p=1.0),
                      # changing image size - mainting aspect ratio for later resize
                      OneOf([RandomCrop(height=self.args.img_size[0], width=self.args.img_size[1], p=0.5),
                              RandomCrop(height=int(random_scale * self.args.img_size[0]),
                                         width=int(random_scale * self.args.img_size[1]), p=0.5)], p=1.0),
                      # flipping / rotating
                      OneOf([HorizontalFlip(p=0.5),
                            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=5)], p=0.5),
                      # contrast and brightness
                      RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
                      # making sure resize fits with yolo input size
                      Resize(height=self.args.img_size[0], width=self.args.img_size[1], p=1.0),
                      Normalize(p=1.0)])

        preform_augmentation = Compose(transforms, bbox_params=BboxParams(format='yolo',
                                                                          min_visibility=0.3))
        augmented_sample = preform_augmentation(**annotations)

        augmented_sample["bboxes"] = np.array(augmented_sample["bboxes"])

        return augmented_sample

    def transform_val(self, sample):
        """ data augmentation for training """
        img = sample["image"]
        bboxes = sample["bboxes"]

        imgH = img.shape[0]
        imgW = img.shape[1]

        random_scale = np.random.randint(8, 11) / 10

        if bboxes.size == 0:
            bboxes = np.array([[0.1, 0.1, 0.1, 0.1, 0.0]])  # this is just a dummy - all values must be inside (0,1)

        annotations = {'image': img, 'bboxes': bboxes}

        transforms = ([PadIfNeeded(min_height=self.args.img_size[0], min_width=self.args.img_size[1],
                                   border_mode=cv2.BORDER_REPLICATE,
                                   p=1.0),
                       # changing image size - maintaining aspect ratio for later resize
                       OneOf([RandomCrop(height=self.args.img_size[0], width=self.args.img_size[1], p=0.5),
                              RandomCrop(height=int(random_scale * self.args.img_size[0]),
                                         width=int(random_scale * self.args.img_size[1]), p=0.5)], p=1.0),
                       # making sure resize fits with yolo input size
                       Resize(height=self.args.img_size[0], width=self.args.img_size[1], p=1.0),
                       Normalize(p=1.0)])

        preform_augmentation = Compose(transforms, bbox_params=BboxParams(format='yolo',
                                                                          min_visibility=0.3))
        augmented_sample = preform_augmentation(**annotations)

        augmented_sample["bboxes"] = np.array(augmented_sample["bboxes"])

        return augmented_sample

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """

        valid_image_files = []
        for looproot, _, filenames in os.walk(rootdir):
            for filename in filenames:
                if filename.endswith(suffix):
                    image_path = os.path.join(looproot, filename)
                    label_path = image_path.replace("images", "labels").replace("bmp", "txt")
                    if os.path.isfile(label_path):
                       valid_image_files.append(image_path)

        return valid_image_files

    def get_bbox_from_txt_file(self, txt_file, img):
        """
        return bounding box numpy array in YOLO format:
        [x0, y0, w, h, cls]
        x0, y0, w, h: are normalized by image size and lie between (0,1)
        cls is the the cls index
        """
        bbox = []
        with open(txt_file, 'r') as f:
            for line in f:
                vec = line.split(' ')
                cls = int(vec[0])

                x0 = float(vec[1])
                y0 = float(vec[2])
                w = float(vec[3])
                h = float(vec[4])
                # check valitidy of bbox params:
                rows, cols, _ = img.shape
                # incase there are errors in the txt file bbox width and height
                h = max([h, 1/rows])
                w = max([w, 1 / cols])

                x_min, x_max = x0 - w/2, x0 + w/2
                y_min, y_max = y0 - h/2, y0 + h/2

                delta1, delta2 = 0, 0
                if x_min <= 0 or x_max >=1:
                    if x_min <= 0:
                        delta1 = -x_min

                    if x_max >= 1:
                        delta2 = x_max - 1

                    delta = max([delta1, delta2])

                    w = max([w - 2 * delta - 1/cols, 1/cols])

                delta1, delta2 = 0, 0
                if y_min <= 0 or y_max >= 1:
                    if y_min <= 0:
                        delta1 = -y_min

                    if y_max >= 1:
                        delta2 = y_max - 1

                    delta = max([delta1, delta2])

                    h = max([h - 2 * delta - 1 / rows, 1/rows])

                bbox.append([x0, y0, w, h, cls])

                if not np.all((0 < np.array([x0, y0, w, h])) & (np.array([x0, y0, w, h]) < 1)):
                    print("erros")
                    print([x0, y0, w, h])
                    print("cols {}, rows {}".format(rows, cols))
                    print(txt_file)
                    raise ValueError("In YOLO format all labels must be float and in range (0, 1)")

        bbox = np.array(bbox)
        return bbox

    def class_count(self):
        from ..data.utils.dataset_statistics import count_cls as countCls

        cls_indx = list(self.label_decoding.values())
        cls_indx = [str(i) for i in cls_indx]
        cls_names = list(self.label_decoding.keys())
        cls_dict = dict(zip(cls_indx, cls_names))

        cls_count_dict = countCls(self.args.train_data, cls_dict)

        cls_count = [None] * len(cls_names)
        # turning dict to array in correct order of classes indx:
        for cls_name in cls_names:
            cls_count[self.label_decoding[cls_name]] = cls_count_dict[cls_name]

        print("\ncls count: {}\n".format(cls_count))

        return np.array(cls_count)

    def get_telemtry_from_file(self, mtigv_file):
        """ converts txt in mtigv file to telemtry vector """
        with open(mtigv_file, "r") as f:
            recorded_telemetry_lines = f.readlines()

        # verify only single line in file
        if len(recorded_telemetry_lines) > 1:
            raise("more then 1 line in file: {}".format(mtigv_file))

        # splitting txt in file to dict:
        data = np.array(recorded_telemetry_lines[0].split(" "))[:-1].astype(np.float)

        # convert data
        longtitude, latitude, altitude = data[5:8].astype(np.float)
        X, Y, zone1, zone2  = utm.from_latlon(latitude, longtitude)
        XYZ = np.array([X, Y, altitude])

        Vxyz = np.array([data[12], data[11], data[13]])   # Vnorth, Veast, Vup
        # converting data to dict:
        telemetry_data = {
                            'indx': data[0],
                            'ver': data[1],
                            'timestamp': data[3],
                            'utmpos': XYZ,
                            'yaw_pitch_roll': data[8:11],
                            'velNothEastUp': Vxyz,
                            'vel_yaw_pitch_roll': data[14:17],
                            'dH_FOV': data[17],
                            'dV_FOV': data[18],
                         }

        return telemetry_data

if __name__ == "__main__":
    from DeepTools.argparse_utils.custom_arg_parser import CustomArgumentParser
    from DeepTools.data_augmentations.detection.custom_transforms import visualize
    parser = argparse.ArgumentParser(description="kitti database data loader", fromfile_prefix_chars="@")
    parser.convert_arg_line_to_args = CustomArgumentParser(parser).convert_arg_line_to_args

    parser.add_argument('--data', type=str,
                        required=True,
                        help='txt file containt all train images path')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='train batch size')

    parser.add_argument('--img-size', type=int, default=416,
                        help='image size')

    args, unknow_args = parser.parse_known_args()

    # creating train dataset
    args.batch_size = 1
    train_set = ProbotSensorsDataset(args)

    # creating train dataloader
    data_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # looping over dataloader for sanity check
    pbar = tqdm.tqdm(data_loader)
    pbar.set_description("loaded data")
    for i, sample in enumerate(pbar):
        # annotation = {'image': image[0].permute(1, 2, 0).numpy(), 'bboxes':bbox[0].numpy(), 'category_id': bbox[0][bbox[0,:,-1]>-1][:,-1]}
        # visualize(annotation, category_id_to_name, plot=True)
        cv2.imshow('image', cv2.cvtColor(sample["image"].numpy().squeeze(0), cv2.COLOR_BGR2RGB))

        print(sample["telemetry"]["utmpos"][0][0], sample["telemetry"]["utmpos"][0][1])
        plt.scatter(sample["telemetry"]["utmpos"][0][0], sample["telemetry"]["utmpos"][0][1], color='blue', marker='x')
        ax.set_aspect('equal', 'box')
        # plt.axis('equal', 'box')
        plt.pause(0.005)
        plt.grid(True)
        cv2.waitKey(1)



