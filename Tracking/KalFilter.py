import numpy as np


class KalmanFilter:
    """
    Kalman Filter for tracking objects
    Attributes:
        state: (x, y , velocity_x, velocity_y) state of the object
        P: ndarry (dim_x, dim_x) cov matrix between pos and velocity
    """

    def __init__(self, F, H, P, Q, R, I, position=None, life=5):
        self.F = F  # State transition matrix, the mapping matrix from the previous state to the current state
        self.H = H  # measurement matrix, map internal state to what we observed , aka the masking to get x, y :D
        self.P = P  # state covariance matrix, the uncertainty of the init state
        self.Q = Q  # process noise covariance matrix, the uncertainty in the motion model
        self.R = R  # how noisy the measurement noise is, diagonal matrix
        self.I = I  # Identity matrix, used to update the state
        self.history = []
        self.bbox = None
        self.life = life
        # state info of the object
        self.state = np.zeros((F.shape[0], 1))
        self.state[0, 0] = position[0] if position is not None else 0  # X
        self.state[1, 0] = position[1] if position is not None else 0  # Y

    def predict(self):
        """
        return the predicted position of the object
        example:
        - (x, y) given
        returns: 
        - (x + vx, y + vy) where vx, vy are the velocity in x and y direction
        """
        # state is 4x1 F is 4x4 we making state to have shape (4,1) masking using our F function to predict the next state
        self.state = self.F @ self.state  # state pred
        # cov matrix pred, aware of measure noise, result in a symetric matrix using transpose
        self.P = self.F @ self.P @ self.F.T + self.Q
        predicted_pos = [self.state[0, 0], self.state[1, 0]]  # x, y position
        self.history.append(predicted_pos)
        return predicted_pos

    def update(self, measurement):
        # map H to state to get 2D value, diff between measurement and predicted state res: 2x1
        y = measurement - self.H @ self.state
        # S is the innovation covariance (diff of the measurement and the predicted state) same calculation like self.P but using H and R
        S = self.H @ self.P @ self.H.T + self.R
        # K is the Kalman gain, result is 4x2 that maps to update our state
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # update the state by adding the Kalman gain times the innovation
        self.state = self.state + K @ y
        self.P = (self.I - K @ self.H) @ self.P  # update the state uncertainty

    @staticmethod
    def create_kalman_filter(position=None, life=5):
        # x_new = x + vx, y_new = y + vy hence why F is like this
        F = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        # map 4D to 2D grabing the x and y position
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        # initial state covariance matrix relationship between velocity and position, may not be fully accurate hence we gonna make the value a bit larger, P will be updated as we predict and update the state
        P = np.eye(4) * 1000
        # factor outside of the system, how much noise we expect in the process, so our model won't be confident in the prediction like L1, L2 norms
        Q = np.eye(4) * 1e-1  # model uncertainty
        # measurement noise
        R = np.eye(2) * 1e-1  # measurement uncertainty
        I = np.eye(4)
        return KalmanFilter(F, H, P, Q, R, I, position, life)
