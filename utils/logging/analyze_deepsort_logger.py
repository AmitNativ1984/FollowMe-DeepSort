import matplotlib.pyplot as plt
import argparse
from utils.logging.deepsort_logger import DeepSortLogger
import numpy as np


def analyze_log(logfile):
    frames = {}
    tracks = {}

    track = dict.fromkeys(["sec", "frames", "conf", "XYZ", "cls", "XYZ", "BBOX_tlbr"])

    with open(logfile, 'r') as f:
        for line in f:
            sec, frame, track_id, cls, conf, bbox_l, bbox_t, bbox_r, bbox_b, x, y, z = DeepSortLogger.read(line)

            if frame not in frames:
                frames[str(frame)] = 1
            else:
                frames[str(frame)] += 1

            if track_id != "None":
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

    frames, tracks = analyze_log(args.logfile)
    # frames, x, y, z = parse_log_file(args.logfile)

    frames = np.array(list(frames.keys())).astype(np.int)

    track = tracks[args.track_id]
    x, y, z = np.array(track["XYZ"])[:, 0], np.array(track["XYZ"])[:, 1], np.array(track["XYZ"])[:, 2]
    frames_with_tracking = np.array(track["frames"])
    t = np.array(track["sec"])

    R = np.sqrt(x**2 + z**2)

    dzdt = z[1:] - z[0:-1]
    dRdt = R[1:] - R[0:-1]

    absdzdt = np.abs(dzdt)
    absdRdt = np.abs(dRdt)

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
    plt.scatter(frames_with_tracking[:-1], absdzdt, c='blue', label='dz', s=20)
    plt.scatter(frames_with_tracking[:-1], absdRdt, c='red', label='dR', s=20)
    plt.title('distance')
    plt.xlabel('# frame')
    plt.ylabel('[m/frame]')
    plt.legend()
    plt.tight_layout()
    plt.grid()

    # plotting distance changes as function of distance
    plt.figure()
    plt.scatter(z[:-1], absdzdt, c='blue', label='dz/z', s=20)
    plt.title("dz as function of z")
    plt.xlabel("z [m]")
    plt.ylabel("dz [m]")

    plt.show()

