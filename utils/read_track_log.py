import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

def parse_log_file(logfile):
    frames = {}
    tracks = {}

    track = dict.fromkeys(["sec", "frames", "conf", "XYZ", "cls", "XYZ", "BBOX_tlbr"])

    with open(logfile, 'r') as f:
        for line in f:
            timestamp, data = line.rsplit(' - INFO - ')
            # time = datetime.strptime(timestamp.split(' - DeepSort')[0], '%Y-%m-%d %H:%M:%S,%f')
            # time = datetime.strptime(' '.join(line.split(' ')[:2]), '%Y-%m-%d %H:%M:%S,%f')
            # msec = time.timestamp() * 1E3
            sec, frame, track_id, cls, conf, bbox_l, bbox_t, bbox_r, bbox_b, x, y, z = data[:-1].split(',')

            if frame not in frames:
                frames[str(frame)] = 1
            else:
                frames[str(frame)] += 1

            if "None" not in data:
                track["sec"] = [sec]
                track["frames"] = [float(frame)]
                track["conf"] = [float(conf)]
                track["cls"] = [int(cls)]
                track["XYZ"] = [np.array([float(x), float(y), float(z)])]
                track["BBOX_tlbr"] = [np.array([float(bbox_l), float(bbox_t), float(bbox_r), float(bbox_b)])]

                if track_id not in tracks:
                    tracks[str(track_id)] = track.copy()

                else:
                    tracks[str(track_id)]["sec"].append(track["sec"][0])
                    tracks[str(track_id)]["frames"].append(track["frames"][0])
                    tracks[str(track_id)]["conf"].append(track["conf"][0])
                    tracks[str(track_id)]["cls"].append(track["cls"][0])
                    tracks[str(track_id)]["XYZ"].append(track["XYZ"][0])
                    tracks[str(track_id)]["BBOX_tlbr"].append(track["BBOX_tlbr"][0])


    return frames, tracks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", type=str, default='')
    parser.add_argument("--track-id", type=str, default='1')
    args = parser.parse_args()

    frames, tracks = parse_log_file(args.logfile)
    # frames, x, y, z = parse_log_file(args.logfile)

    frames = np.array(list(frames.keys())).astype(np.int)

    track = tracks[args.track_id]
    x, y, z = np.array(track["XYZ"])[:, 0], np.array(track["XYZ"])[:, 1], np.array(track["XYZ"])[:, 2]
    frames_with_tracking = np.array(track["frames"])
    t = np.array(track["sec"])

    R = np.sqrt(x**2 + z**2)

    dzdt = z[1:] - z[0:-1]
    dRdt = R[1:] - R[0:-1]

    # plotting:
    plt.figure
    plt.subplot(2, 1, 1)
    plt.scatter(frames, np.zeros(np.size(frames)), c='black', label='frames', s=20, alpha=0.5)
    plt.scatter(frames_with_tracking, z, c='blue', label='depth (z)', s=20, alpha=0.5)
    plt.scatter(frames_with_tracking, R, c='red', label='radial', s=20, alpha=0.5)
    plt.title('distance')
    plt.xlabel('# frame')
    plt.ylabel('[m]')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.scatter(frames, np.zeros(np.size(frames)), c='black', label='frames', s=20, alpha=0.5)
    plt.scatter(frames_with_tracking[:-1], np.abs(dzdt), c='blue', label='dz', s=20)
    plt.scatter(frames_with_tracking[:-1], np.abs(dRdt), c='red', label='dR', s=20)
    plt.title('distance')
    plt.xlabel('# frame')
    plt.ylabel('[m/frame]')
    plt.legend()
    plt.tight_layout()
    plt.grid()

    # print(np.average(dRdt[200:600]))
    # print(np.std(dRdt[200:600]))
    # print(np.max(np.abs(dRdt[200:600])))

    plt.show()

