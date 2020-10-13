import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

class TrackData(object):
    def __init__(self, timestamp, frame, track_id, conf, class_id, bbox, xyz_pos):
        self.timestamps = timestamp
        self.frames = frame
        self.track_id = track_id
        self.confs = conf
        self.class_id = class_id
        self.bbox = bbox
        self.xyz_pos = xyz_pos

    def append_track_data(self, timestamp, frame, conf, class_id, bbox, xyz_pos):
        """ info: list of lines read from log file """
        self.timestamps.append(timestamp)
        self.frames.append(frame)
        self.confs.append(conf)
        self.class_id.append(class_id)
        self.bbox.append(bbox)
        self.xyz_pos.append(xyz_pos)

def parse_track_data(track_info):
    for line in track_info:
        if "Detection ID =" in line:
            track_id = int(line.split("Detection ID = ")[-1])
        if "Confidence =" in line:
            conf = float(line.split("Confidence = ")[-1])
        if "Class ID =" in line:
            class_id = int(line.split("Class ID = ")[-1])
        if "BBox =" in line:
            bbox = line.split("BBox = ")[-1][1:-1].split(',')
            bbox = np.array(bbox, dtype=np.int)
        if "Position =" in line:
            xyz_pos = line.split("Position = ")[-1][1:-1].split(',')
            xyz_pos = np.array(xyz_pos, dtype=np.float32)

    return track_id, conf, class_id, bbox, xyz_pos

def parse_log_file(logfile):
    t0 = -1
    timestamp = []
    newMessage = False
    frames = []
    all_track_ids = []
    tracks = []
    with open(logfile, 'r') as log:
        while True:
            line = log.readline()
            if not line:
                break

            # reading new message from deepsort
            if "Message available" in line:
                newMessage = True
                time = datetime.strptime(' '.join(line.split(' ')[:2]), '%Y-%m-%d %H:%M:%S,%f')
                msec = time.timestamp()*1E3
                if t0 < 0:
                    t0 = msec

                timestamp.append(msec - t0)

            if newMessage and "Detecting on image id =" in line:
                frames.append(int(line.split('Detecting on image id = ')[-1]))

            if newMessage and "detections" in line:
                num_tracks = int(line.split(' ')[-2])
                for track in range(num_tracks):
                    track_info = []
                    for i in range(5):
                        track_info.append(next(log).strip())

                    track_id, conf, class_id, bbox, xyz_pos = parse_track_data(track_info)

                    track_found = False
                    for track in tracks:
                        if track_id == track.track_id:
                            track_found = True
                            track.append_track_data(timestamp[-1], frames[-1], conf, class_id, bbox, xyz_pos)

                    if not track_found:
                        # initiate new track
                        new_track = TrackData([timestamp[-1]], [frames[-1]], track_id, [conf], [class_id], [bbox], [xyz_pos])
                        tracks.append(new_track)


    log.close()
    return timestamp, frames, tracks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", type=str, default='')
    parser.add_argument("--track-id", type=int, default=1)

    args = parser.parse_args()

    timestamp, frames, tracks = parse_log_file(args.logfile)
    timestamp = np.array(timestamp)
    timestamp = timestamp[2:]
    dt = timestamp[1:] - timestamp[:-1]

    # find relevant id:

    ind = -1
    for i, track in enumerate(tracks):
        if track.track_id == args.track_id:
            ind = i

    # only track of intrest
    track = tracks[ind]
    confirmed_frames = np.array(track.frames)
    confirmed_timestamp = np.array(track.timestamps)
    dt = confirmed_timestamp[1:] - confirmed_timestamp[:-1]
    dframes = confirmed_frames[1:] - confirmed_frames[:-1]
    xyz = np.array(track.xyz_pos)
    x = xyz[:, 0]
    z = xyz[:, -1]
    R = np.sqrt(x**2 + z**2)

    with open("./" + args.logfile.split('\\')[-1], 'a+') as f:
        for t, frm, r, depth in zip(list(confirmed_timestamp), list(confirmed_frames), list(R), list(z)):
            f.write("{},{},},{}\n".format(t, frm, r, depth))



    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.scatter(timestamp/1E3, np.zeros(np.shape(timestamp)), c='black', s=10, alpha=0.5, label='')
    plt.scatter(confirmed_timestamp/1E3, z, c='blue', alpha=0.5, s=10, label='depth (z)')
    plt.scatter(confirmed_timestamp/1E3, R, c='red', alpha=0.5, s=10, label='radial')
    plt.title('distance')
    plt.xlabel('t [sec]')
    plt.ylabel('[m]')
    plt.legend()
    plt.grid()

    dzdt = z[1:] - z[0:-1]
    dRdt = R[1:] - R[0:-1]

    plt.subplot(2, 1, 2)
    plt.scatter(timestamp/1E3, np.zeros(np.shape(timestamp)), c='black', s=10, alpha=0.5, label='')
    plt.scatter(confirmed_timestamp[:-1]/1E3, np.abs(dzdt), c='blue', alpha=0.5, s=10, label='dz')
    plt.scatter(confirmed_timestamp[:-1]/1E3, np.abs(dRdt), c='red', alpha=0.5, s=10, label='dR')
    plt.title('distance')
    plt.xlabel('t [msec]')
    plt.ylabel('[m/s]')
    plt.legend()
    plt.tight_layout()
    plt.grid()

    # --- relative to frames ---:

    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.scatter(confirmed_frames, np.zeros(np.shape(confirmed_frames)), c='black', s=25, alpha=0.5, label='')
    plt.scatter(confirmed_frames, z, c='blue', alpha=0.5, s=10, label='depth (z)')
    plt.scatter(confirmed_frames, R, c='red', alpha=0.5, s=10, label='radial')
    plt.title('distance')
    plt.xlabel('# frames')
    plt.ylabel('[m]')
    plt.legend()
    plt.grid()

    dzdt = z[1:] - z[0:-1]
    dRdt = R[1:] - R[0:-1]

    plt.subplot(2, 1, 2)
    plt.scatter(confirmed_frames, np.zeros(np.shape(confirmed_frames)), c='black', s=25, alpha=0.5, label='')
    plt.scatter(confirmed_frames[:-1], np.abs(dzdt), c='blue', alpha=0.5, s=10, label='dz')
    plt.scatter(confirmed_frames[:-1], np.abs(dRdt), c='red', alpha=0.5, s=10, label='dR')
    plt.title('distance')
    plt.xlabel('# frames')
    plt.ylabel('[m/frame]')
    plt.legend()
    plt.tight_layout()
    plt.grid()


    print(np.average(dRdt[250:400]))
    print(np.std(dRdt[250:400]))
    print(np.max(np.abs(dRdt[250:400])))


    fig, ax1 = plt.subplots()
    ax1.set_xlabel('frames')
    ax1.set_ylabel('[m/s]')
    plt.title('distance derivative')
    ax1.scatter(confirmed_frames[:-1], np.abs(dzdt), c='blue', alpha=0.5, s=10, label='dz')
    ax1.scatter(confirmed_frames[:-1], np.abs(dRdt), c='red', alpha=0.5, s=10, label='dR')
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.set_ylabel('dt')
    # ax2.plot(confirmed_timestamp[:-1]/1E3,dt)
    ax2.plot(confirmed_frames[:-1], dframes)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('frames')
    ax1.set_ylabel('[m/s]')
    plt.title('distance derivative')
    ax1.scatter(confirmed_frames[0] + range(len(confirmed_frames[:-1])), np.abs(dzdt), c='blue', alpha=0.5, s=10, label='dz')
    ax1.scatter(confirmed_frames[0] + range(len(confirmed_frames[:-1])), np.abs(dRdt), c='red', alpha=0.5, s=10, label='dR')
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.set_ylabel('dt')
    # ax2.plot(confirmed_timestamp[:-1]/1E3,dt)
    ax2.plot(confirmed_frames[0] + range(len(confirmed_frames[:-1])), dframes)

    plt.show()

