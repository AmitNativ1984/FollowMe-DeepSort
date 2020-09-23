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



