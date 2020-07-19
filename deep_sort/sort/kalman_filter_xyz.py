import numpy as np

def computeCovMatrix(deltaT, sigma_aX, sigma_aY):

    G = np.array([[deltaT ** 2 / 2.,                 0.],
                    [0.,                 deltaT ** 2. / 2.],
                    [deltaT,                          0.],
                    [0.,                          deltaT]])

    Q_ni = np.array([[sigma_aX ** 2.,                0.],
                     [0.,                     sigma_aY]])


    Q = G @ Q_ni @ G.transpose()

    return Q

def computeFmatrix(deltaT):

    F = np.array([[1.,      0.,  deltaT,          0.],
                  [0.,      1.,      0.,      deltaT],
                  [0.,      0.,      1.,           0.],
                  [0.,      0.,      0.,           1.]])

    return F

class KalmanXYZ(object):
    """ kalman filter for linear measurements """

    def __init__(self, timestamp, x0):
        sigmaPos = 1
        sigmaVel = 100


        self.P = np.array([[sigmaPos,      0.,      0.,      0.],
                           [0.,       sigmaPos,     0.,      0.],
                           [0.,       0.,      sigmaVel,    0.],
                           [0.,       0.,      0.,    sigmaVel]])

        # projection from sensor to state space
        self.H = np.array([[1.,     0.,      0.,      0.],
                           [0.,     1.,      0.,      0.]])

        # image detection noise (in meters)
        self.R = np.array([[0.5,        0.0],
                           [0.0,        0.5]])

        self.X_state_current = np.vstack((x0[:2], 0, 0))
        self.timestamp = timestamp

    def predict(self, new_timestamp, U):
        """
        predict state according to the physical model
        """
        deltaT = (new_timestamp - self.timestamp) / 1E3
        self.timestamp = new_timestamp

        self.F_matrix = computeFmatrix(deltaT)

        Q = computeCovMatrix(deltaT, sigma_aX=0.5, sigma_aY=0.5)

        self.X_state_current = (self.F_matrix @ self.X_state_current) + U
        self.P = self.F_matrix @ self.P @ self.F_matrix.transpose() + Q

        return self.X_state_current, self.P


    def update_by_measurment(self, currentMeas):
        """
        update state according to new incoming measurement
        :param currentMeas:
        :return:
        """
        I = np.identity(self.P.shape[0])
        z = currentMeas[:2]

        z_pred = self.H @ self.X_state_current
        y = z - z_pred

        S = self.H @ self.P @ self.H.transpose() + self.R
        K = self.P @ self.H.transpose() @ np.linalg.inv(S)

        self.X_state_current = self.X_state_current + (K @ y)
        self.P = (I - K @ self.H) @ self.P

        return self.X_state_current
