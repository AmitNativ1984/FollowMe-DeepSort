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
        frame, x, y, z = line.split(',')
        FRAME.append(float(frame))
        X.append(float(x))
        Y.append(float(y))
        Z.append(float(z))

    return np.array(FRAME), np.array(X), np.array(Y), np.array(Z)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", type=str, default='')
    args = parser.parse_args()

    frames, x, y, z = parse_log_file(args.logfile)

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
    plt.plot(frames[:-1], dzdt, c='blue', label='dz')
    plt.plot(frames[:-1], dRdt, c='red', label='dR')
    plt.title('distance')
    plt.xlabel('# frame')
    plt.ylabel('[m/frame]')
    plt.legend()
    plt.tight_layout()
    plt.grid()

    print(np.average(dRdt[200:600]))
    print(np.std(dRdt[200:600]))
    print(np.max(np.abs(dRdt[200:600])))

    plt.show()

