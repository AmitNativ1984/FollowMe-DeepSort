import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def create_radar_plot(xlim=(-25, 25), ylim=(-25, 25)):
    # setting up canvas for polar and cartesian plot:

    radar_fig = plt.figure()
    # setting the axis limits in [left, bottom, width, height]
    rect = [0.1, 0.1, 0.8, 0.8]

    # the carthesian axis:
    ax_carthesian = radar_fig.add_axes(rect)
    ax_carthesian.set_xlim(xlim)
    ax_carthesian.set_ylim(ylim)
    ax_carthesian.set_aspect('equal')

    # the polar axis:
    ax_polar = radar_fig.add_axes(rect, polar=True, frameon=False)
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_rlabel_position(90)
    ax_polar.set_ylim(0, max(ylim))


    ticklabelpad = mpl.rcParams['xtick.major.pad']


    ax_carthesian.annotate('[m]', xy=(1, 0), xytext=(5, -ticklabelpad), ha='left', va='top',
                xycoords='axes fraction', textcoords='offset points')

    ticklabelpad = mpl.rcParams['ytick.major.pad']
    ax_carthesian.annotate('[m]', xy=(0, 1), xytext=(-5, ticklabelpad), ha='right', va='top',
                           xycoords='axes fraction', textcoords='offset points')

    ax_polar.set_rmax(max(ylim))
    plt.rgrids((5, 10, 15, 20))
    ax_polar.grid(True)

    # plt.show()

    return radar_fig, ax_polar, ax_carthesian

if __name__ == '__main__':
    fig, ax_polar, ax_carthesian = create_radar_plot()
    line = 50 * np.random.rand(50) - 25.
    # ax_carthesian.plot(range(len(line)), line, 'b')
    # ax_carthesian.scatter([0, 1], [10, 15], marker='x', color='r')?
    plt.show()