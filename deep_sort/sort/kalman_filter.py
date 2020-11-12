import numpy as np
import scipy



def computeCovMatrix(deltaT, sigma_aX, sigma_aY, sigma_aZ):

    G = np.array([[deltaT ** 2 / 2.,                0.,                        0.],
                  [0.,               deltaT ** 2. / 2.,                        0.],
                  [0.,                              0.,         deltaT ** 2. / 2.],
                  [deltaT,                          0.,                        0.],
                  [0.,                          deltaT,                        0.],
                  [0.,                              0.,         deltaT ** 2. / 2.]])

    Q_ni = np.array([[sigma_aX ** 2.,                0.,                    0.],
                     [0.,                      sigma_aY,                    0.],
                     [0.,                            0.,         sigma_aZ **2.]])


    Q = G @ Q_ni @ G.transpose()

    return Q

def computeFmatrix(deltaT):
    """ X = (x,y,z,vx,vy,vz)
        in UTM coordinates, the state_space is:
        x = lat
        y = long,
        z = alt
    """

    F = np.array([[1.,      0.,      0.,    deltaT,           0.,            0.],
                  [0.,      1.,      0.,         0.,      deltaT,            0.],
                  [0.,      0.,      1.,         0.,          0.,        deltaT],
                  [0.,      0.,      0.,         1.,          0.,            0.],
                  [0.,      0.,      0.,         0.,          1.,            0.],
                  [0.,      0.,      0.,         0.,          0.,            1.]])

    return F

class KalmanUTM(object):
    """ kalman filter for linear measurements """

    def __init__(self, MAX_KF_UNCERTAINTY_RADIUS=7):
        """
                Table for the 0.95 quantile of the chi-square distribution with N degrees of
                freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
                function and used as Mahalanobis gating threshold.
                """
        self.chi2inv95 = {
            1: 3.8415,
            2: 5.9915,
            3: 7.8147,
            4: 9.4877,
            5: 11.070,
            6: 12.592,
            7: 14.067,
            8: 15.507,
            9: 16.919}

        self.max_uncertainty_radius = MAX_KF_UNCERTAINTY_RADIUS
        self.maxEigenValue = self.max_uncertainty_radius ** 2 / self.chi2inv95[2]

        sigmaPosX = 1
        sigmaPosY = 1
        sigmaPosZ = 1
        sigmaVelX = 10
        sigmaVelY = 10
        sigmaVelZ = 10

        self.P0 = np.array([[sigmaPosX,      0.,        0.,          0.,        0.,         0.],
                           [0.,      sigmaPosY,        0.,          0.,        0.,         0.],
                           [0.,             0., sigmaPosZ,          0.,        0.,         0.],
                           [0.,             0.,        0.,   sigmaVelX,        0.,         0.],
                           [0.,             0.,        0.,          0.,  sigmaVelY,        0.],
                           [0.,             0.,        0.,          0.,         0., sigmaVelZ]])

        # projection from sensor to state space
        self.H = np.array([[1.,     0.,      0.,      0.,       0.,     0.],
                           [0.,     1.,      0.,      0.,       0.,     0.],
                           [0.,     0.,      1.,      0.,       0.,     0.]])

        # image detection noise (in meters)
        sensor_acc_X = 5
        sensor_acc_Y = 5
        sensor_acc_Z = 0.5
        self.R = np.array([[sensor_acc_X,        0.0,                0.0],
                           [0.0,        sensor_acc_Y,                0.0],
                           [0.0,                 0.0,       sensor_acc_Z]])

    def initiate(self, timestamp, utm_pos0, vel0=np.array([[0], [0], [0]])):
        self.X_state_current = np.vstack((utm_pos0, 0, 0, 0))
        self.timestamp = timestamp
        self.P = self.P0
        return self.X_state_current, self.P0

    def predict(self, new_timestamp, U=None):
        """
        predict state according to the physical model
        """

        if U is None:
            U = np.array([[0],
                          [0],
                          [0],
                          [0],
                          [0],
                          [0]])

        deltaT = (new_timestamp - self.timestamp) / 1E3
        self.timestamp = new_timestamp

        self.F_matrix = computeFmatrix(deltaT)

        Q = computeCovMatrix(deltaT, sigma_aX=2, sigma_aY=2, sigma_aZ=2)

        self.X_state_current = (self.F_matrix @ self.X_state_current) + U
        self.P = self.F_matrix @ self.P @ self.F_matrix.transpose() + Q

        # clip maximum uncertainty (otherwise it may blowup for all space)
        eigenvalues, eigenvectors = np.linalg.eig(self.P[:3, :3])


        for ind, eigenvalue in enumerate(eigenvalues):
            if eigenvalue > self.maxEigenValue:
                a0 = self.maxEigenValue / eigenvalue
                self.P[:, ind] = self.P[:, ind] * np.sqrt(a0)
                self.P[ind, :] = self.P[ind, :] * np.sqrt(a0)

        return self.X_state_current, self.P


    def update_by_measurment(self, currentMeas):
        """
        update state according to new incoming measurement
        :param currentMeas:
        :return:
        """
        # incase of long occlusion, with eignevalues at max, if there is a positive match (based on BBOX IoU):
        # initiate kalman filter to get a tight position around the first detection
        eigenvalues, eigenvectors = np.linalg.eig(self.P[:3, :3])
        if max(eigenvalues) >= self.maxEigenValue:
            self.X_state_current, self.P = self.initiate(self.timestamp, currentMeas)

        else:
            I = np.identity(self.P.shape[0])
            z = currentMeas

            z_pred = self.H @ self.X_state_current
            y = z - z_pred

            S = self.H @ self.P @ self.H.transpose() + self.R
            K = self.P @ self.H.transpose() @ np.linalg.inv(S)

            self.X_state_current = self.X_state_current + (K @ y)
            self.P = (I - K @ self.H) @ self.P

        return self.X_state_current, self.P

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        # mean, covariance = self.project(mean, covariance)
        # TODO: pull from master to get hankl
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:2, :]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha


