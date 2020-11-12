import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import os

def split_mq_log_file(mqlogfile):
    outfolder = os.path.split(mqlogfile)[0] + r"\pyfollowme_mq_logs"
    os.makedirs(outfolder, exist_ok=True)

    prev_file_open = False
    with open(mqlogfile, 'r') as mqlog:
        while True:
            line = mqlog.readline()
            if not line:
                break

            if "FollowMe MQ" in line:
                if prev_file_open:
                    prev_file.close()
                    prev_file_open = False

                # Beginning of new experiment detected
                # --> open new txt file with name as time stamp
                time = datetime.strptime(' '.join(line.split(' ')[:2]), '%Y-%m-%d %H:%M:%S,%f')
                timestr = [str(c) for c in list(time.timetuple()[0:6])]
                filename = '_'.join(timestr) + ".txt"
                logfile = os.path.join(outfolder, "pyfollowme_mq_expr" + filename)
                log = open(logfile, "+a")
                prev_file = log
                prev_file_open = True

            log.write(line)

    if prev_file_open:
        log.close()

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", type=str, default='')
    args = parser.parse_args()

    split_mq_log_file(args.logfile)
