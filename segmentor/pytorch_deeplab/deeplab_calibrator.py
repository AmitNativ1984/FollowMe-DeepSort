import pycuda.driver as cuda
import os
import pycuda.autoinit
import numpy as np
from PIL import Image
import ctypes
import tensorrt as trt

CHANNEL = 3
HEIGHT = 720
WIDTH = 1280


class DeeplabEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, input_layers, stream):
        trt.IInt8EntropyCalibrator2.__init__(self)
        # trt.infer.EntropyCalibrator.__init__(self)
        self.input_layers = input_layers
        self.stream = stream
        self.cache_file = 'calibration_cache.bin'

        self.device_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names):
        try:
            # Get a single batch.
            data = self.stream.next_batch()
            if data is None:
                return None

            # Copy to device, then return a list containing pointers to input device buffers.
            cuda.memcpy_htod(self.device_input, data)

            for i in self.input_layers[0]:
                assert names[0] != i

            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    # def get_batch(self, bindings, names):
    #     batch = self.stream.next_batch()
    #     if not batch.size:
    #         return None
    #
    #     cuda.memcpy_htod(self.d_input, batch)
    #     for i in self.input_layers[0]:
    #         assert names[0] != i
    #
    #     bindings[0] = int(self.d_input)
    #     return bindings

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print('reading cache file from disk')
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        return None


class ImageBatchStream():
    def __init__(self, batch_size, calibration_files):
        self.batch_size = batch_size
        self.max_batches = (len(calibration_files) // batch_size) + (1 if (len(calibration_files) % batch_size) else 0)
        self.files = calibration_files
        self.calibration_data = np.zeros((batch_size, CHANNEL, HEIGHT, WIDTH), dtype=np.float32)
        self.batch = 0

    @staticmethod
    def read_image_chw(path):
        img = Image.open(path).convert('RGB').resize((WIDTH, HEIGHT), Image.ANTIALIAS)
        im = np.array(img, dtype=np.float32, order='C')
        im = im[:, :, ::-1].transpose(2, 0, 1)
        im /= 255.0

        return im

    def reset(self):
        self.batch = 0

    def next_batch(self):
        if self.batch < self.max_batches:
            imgs = []
            print("[ImageBatchStream] Processing batch ", self.batch)
            files_for_batch = self.files[self.batch_size * self.batch: \
                                         self.batch_size * (self.batch + 1)]
            for f in files_for_batch:
                print("[ImageBatchStream] Processing ", f)
                img = ImageBatchStream.read_image_chw(f)
                imgs.append(img)
            for i in range(len(imgs)):
                self.calibration_data[i] = imgs[i]
            self.batch += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return None











