import os
from torch.utils.data import Dataset
import cv2
import numpy as np

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif']
pref = "drive_cam0_"

class LoadImages(Dataset):  # for inference
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        imgfiles = [x for x in self.files if os.path.splitext(x)[-1].lower() in img_formats]
        file_stamps = [int(num.split('_')[-1].split('.bmp')[0]) for num in imgfiles]
        inds = np.argsort(file_stamps)
        self.images = [imgfiles[i] for i in inds]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        img = cv2.imread(img_name) # BGR

        return img

