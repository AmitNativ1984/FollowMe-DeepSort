from datetime import datetime
import os
import numpy as np

import logging
from logging import getLogger, Formatter, StreamHandler, FileHandler

PRINT_TO_STDOUT = False

class DeepSortLogger(object):
    def __init__(self):
        self.logger = self.configure_logging()

    def configure_logging(self):
        date, time = (str(datetime.now())).split(' ')
        time = time.split('.')[0].replace(":", "-")
        date = (str(datetime.now())).split(' ')[0]
        full_log_path = os.path.join(os.environ['TERRA_NAVA_LOG_DIR'], 'followme')
        os.makedirs(full_log_path, exist_ok=True)

        log_name = "DeepSort"
        filename = os.path.join(full_log_path, 'tracks_' + date + "_" + time + '.log')

        logger = getLogger(log_name)
        logger.setLevel(logging.DEBUG)

        formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create file output.
        full_log_path = os.path.join(os.environ['TERRA_NAVA_LOG_DIR'], filename)
        file_handler = FileHandler(full_log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if PRINT_TO_STDOUT:
            # Create console output.
            console = StreamHandler()
            console.setLevel(logging.DEBUG)
            console.setFormatter(formatter)
            logger.addHandler(console)

        return logger

    def write(self, frame, tracks, cam2world):

        # log_manager:
        time_sec = datetime.strptime(str(datetime.now()), '%Y-%m-%d %H:%M:%S.%f').timestamp()
        if len(tracks) == 0:
            msg = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(time_sec, frame, None, None, None, None, None, None, None, None, None, None, None, None, None)
            self.logger.info(msg)

        else:
            # logline looks like this:
            # timesec, frame, track_id, cls_id, conf, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, utm_E, utm_N, utm_alt, Vel_E, Vel_N, Vel_alt
            for track in tracks:
                msg = "{},{},{},{},{},{},{}".format(time_sec, frame, track.track_id, int(track.cls_id), track.confidence,
                                                    str(list(track.utm_to_bbox_tlbr(cam2world).squeeze().astype(np.int)))[1:-1].replace(" ", ""),
                                                    str(list(track.mean.squeeze()))[1:-1].replace(" ", ""))
                self.logger.info(msg)

        pass

    @staticmethod
    def read(logline):
        timestamp, data = logline.rsplit(' - INFO - ')
        sec, frame, track_id, cls, conf, bbox_l, bbox_t, bbox_r, bbox_b, x, y, z = data[:-1].split(',')

        return sec, frame, track_id, cls, conf, bbox_l, bbox_t, bbox_r, bbox_b, x, y, z
