import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_log_file(logfile):
    with open(logfile, 'r') as f:
        lines = f.read().splitlines()

    FRAME = []
    X = []
    Y = []
    Z = []

    for line in lines:
        frame, cls_id, conf, bbox, xyz_pos = line.split(',')
        FRAME.append(float(frame))
        x, y, z = xyz_pos[1:-1].split()
        X.append(float(x))
        Y.append(float(y))
        Z.append(float(z))

    return np.array(FRAME), np.array(X), np.array(Y), np.array(Z)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", type=str, default='')
    args = parser.parse_args()

    frames, x, y, z = parse_log_file(args.logfile)
    dframes = frames[1:] - frames[:-1]

    R = np.sqrt(x**2 + z**2)

    plt.figure
    plt.subplot(2, 1, 1)
    plt.plot(frames, z, c='blue', label='depth (z)')
    plt.plot(frames, R, c='red', label='radial')
    plt.title('distance')
    plt.xlabel('# frame')
    plt.ylabel('[m]')
    plt.legend()
    plt.grid()

    dzdt = z[1:] - z[0:-1]
    dRdt = R[1:] - R[0:-1]

    plt.subplot(2, 1, 2)
    plt.plot(frames[:-1], np.abs(dzdt), c='blue', label='dz')
    plt.plot(frames[:-1], np.abs(dRdt), c='red', label='dR')
    plt.title('distance')
    plt.xlabel('# frame')
    plt.ylabel('[m/frame]')
    plt.legend()
    plt.tight_layout()
    plt.grid()

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('frames')
    ax1.set_ylabel('[m/s]')
    plt.title('distance derivative')
    ax1.scatter(frames[:-1], np.abs(dzdt), c='blue', label='dz', s=10, alpha=0.5)
    ax1.scatter(frames[:-1], np.abs(dRdt), c='red', alpha=0.5, s=10, label='dR')
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.set_ylabel('# dframes [N - (N-1)]')
    # ax2.plot(confirmed_timestamp[:-1]/1E3,dt)
    ax2.plot(frames[:-1], dframes)

    # print(np.average(dRdt[200:600]))
    # print(np.std(dRdt[200:600]))
    # print(np.max(np.abs(dRdt[200:600])))

    plt.show()

