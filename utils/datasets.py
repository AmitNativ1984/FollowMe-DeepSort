import os
from torch.utils.data import Dataset
import cv2

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif']
pref = "cam_front_"

class LoadImages(Dataset):  # for inference
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        imgfiles = [x for x in self.files if os.path.splitext(x)[-1].lower() in img_formats]
        file_stamps = [int(num.split('_')[-1].split('.bmp')[0]) for num in imgfiles]
        file_stamps.sort()
        self.images = [pref + str(c) + '.bmp' for c in file_stamps]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        img = cv2.imread(img_name) # BGR

        return img

