from datetime import datetime
import os
from pyterranava.logging import configure_logging
import numpy as np

class DeepSortLogger(object):
    def __init__(self):
        # configuring logger:
        date, time = (str(datetime.now())).split(' ')
        time = time.split('.')[0].replace(":", "-")
        date = (str(datetime.now())).split(' ')[0]
        full_log_path = os.path.join(os.environ['TERRA_NAVA_LOG_DIR'], 'followme', date + "_" + time)
        os.makedirs(full_log_path, exist_ok=True)
        self.logger = configure_logging(log_name="DeepSort", filename=os.path.join(full_log_path, 'tracks.log'))

    def write(self, frame, tracks):

        # logging:
        time_sec = datetime.strptime(str(datetime.now()), '%Y-%m-%d %H:%M:%S.%f').timestamp()
        if len(tracks) == 0:
            msg = "{},{},{},{},{},{},{},{},{},{},{},{}".format(time_sec, frame, None, None, None, None, None, None, None, None, None, None)
            self.logger.info(msg)

        else:
            for track in tracks:
                msg = "{},{},{},{},{},{},{}".format(time_sec, frame, track.track_id, int(track.cls_id), track.confidence,
                                                    str(list(track.to_tlbr().astype(np.int)))[1:-1].replace(" ", ""),
                                                    str(track.xyz_pos)[1:-1].replace(" ", ""))
                self.logger.info(msg)

        pass

    @staticmethod
    def read(logline):
        timestamp, data = logline.rsplit(' - INFO - ')
        sec, frame, track_id, cls, conf, bbox_l, bbox_t, bbox_r, bbox_b, x, y, z = data[:-1].split(',')

        return sec, frame, track_id, cls, conf, bbox_l, bbox_t, bbox_r, bbox_b, x, y, z
