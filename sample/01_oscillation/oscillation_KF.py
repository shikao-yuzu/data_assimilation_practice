import numpy as np
import matplotlib.pyplot as plt
import pyda

# Period of data assimilation
# NT_ASM = 1000
NT_ASM = 400
# Period of prediction
# NT_PRD = 1000
NT_PRD = 400
# Interval of observation
OBS_INTERVAL = 40
# Interval of output monitoring
OUTPUT_INTERVAL = 20
# Time step
DT = 0.01


class Model:
    # Model parameter
    PARAM_M = 1.0
    PARAM_C = 0.3
    PARAM_K = 0.5

    def __init__(self, nt: int, dt: float, x_0: float, v_0: float) -> None:
        self.dt = dt

        self.t = np.arange(0.0, (nt+1)*self.dt, self.dt)
        self.x = np.zeros(nt+1, dtype=np.float64)
        self.v = np.zeros(nt+1, dtype=np.float64)

        self.x[0] = x_0
        self.v[0] = v_0

    def predict(self, it_start: int, it_end: int) -> None:
        for it in range(it_start, it_end+1):
            self.predict_one_step(it)

    def predict_one_step(self, it: int) -> None:
        # Forward euler scheme
        self.x[it] = self.x[it - 1] + self.dt * self.v[it - 1]
        self.v[it] = - (self.PARAM_K * self.dt / self.PARAM_M) * self.x[it - 1] + \
                     (1.0 - self.PARAM_C * self.dt / self.PARAM_M) * self.v[it - 1]

    def output(self, interval: int) -> tuple:
        t = np.copy(self.t[::interval])
        x = np.copy(self.x[::interval])
        v = np.copy(self.v[::interval])
        return t, x, v

    def get_M_matrix(self) -> np.ndarray:
        M = np.zeros((2, 2), dtype=np.float64)
        M[0, 0] = 1.0
        M[0, 1] = self.dt
        M[1, 0] = - self.PARAM_K * self.dt / self.PARAM_M
        M[1, 1] = 1.0 - self.PARAM_C * self.dt / self.PARAM_M
        return M


class DA:
    def __init__(self, mdl: Model) -> None:
        self.mdl = mdl

    def assimilation_KF(self, it_start: int, it_end: int, obs: np.ndarray, R: np.ndarray, H: np.ndarray,
                        obs_interval: int) -> None:
        # Pf initial
        Pf = np.zeros((2, 2), dtype=np.float64)
        Pf[0, 0] = 1.0
        Pf[1, 1] = 1.0

        for it in range(it_start, it_end+1):
            # Time integration
            self.mdl.predict_one_step(it)

            # State Transient Matrix
            M = self.mdl.get_M_matrix()

            # Lyapunov equation: Obtain Pf
            Pf = M * Pf * M.transpose()

            if it % obs_interval == 0:
                # Kalman gain: Weighting of model result and obs.
                #   Obsevation only in x --> component 1(x) only
                #   In this case, inverse matrix --> scalar inverse
                Kg = np.zeros((2, 2), dtype=np.float64)
                Kg[0, 0] = Pf[0, 0] / (R[0, 0] + Pf[0, 0])
                Kg[1, 0] = Pf[1, 0] / (R[0, 0] + Pf[0, 0])

                # Calculate innovation and correction
                it_obs = int(it / obs_interval)
                innov = obs[it_obs] - self.mdl.x[it]
                self.mdl.x[it] = self.mdl.x[it] + Kg[0, 0] * innov
                self.mdl.v[it] = self.mdl.v[it] + Kg[1, 0] * innov

                # Analysis error covariance matrix
                Pa = Pf - Kg * H * Pf
                Pf = Pa


if __name__ == '__main__':
    # True Field
    mdl_t = Model(nt=NT_ASM+NT_PRD, dt=DT, x_0=5.0, v_0=0.0)
    mdl_t.predict(1, NT_ASM+NT_PRD)
    t_t, x_t, v_t = mdl_t.output(OUTPUT_INTERVAL)

    # Observation error covariance matrix
    R = np.zeros((1, 1), dtype=np.float64)
    R[0, 0] = 0.1

    # H Matrix
    H = np.zeros((2, 2), dtype=np.float64)
    H[0, 0] = 1.0

    # Observations
    #   generate observation by adding Gaussian noise (uniform random number) to true value
    np.random.seed(0)
    t_obs, x_obs, _ = mdl_t.output(OBS_INTERVAL)
    gnoise = np.sqrt(R[0, 0]) * np.random.randn(len(t_obs))
    x_obs += gnoise

    # Simulation run without DA (wrong initial value)
    mdl_s = Model(nt=NT_ASM+NT_PRD, dt=DT, x_0=4.0, v_0=1.0)
    mdl_s.predict(1, NT_ASM+NT_PRD)
    t_s, x_s, v_s = mdl_s.output(OUTPUT_INTERVAL)

    # Simulation run with DA (wrong initial value)
    mdl_da = Model(nt=NT_ASM+NT_PRD, dt=DT, x_0=4.0, v_0=1.0)
    da = DA(mdl_da)
    da.assimilation_KF(1, NT_ASM, x_obs, R, H, OBS_INTERVAL)
    mdl_da.predict(NT_ASM+1, NT_ASM+NT_PRD)
    t_da, x_da, v_da = mdl_da.output(OUTPUT_INTERVAL)

    # File output
    with open('result.txt', mode='wt', encoding='cp932') as f:
        f.write('******* x ********')
        f.write(' [Time]   [True]  [No Assim]  [Assim]')
        for i in range(len(t_t)):
            f.write('{0:7.2f}{1:10.3f}{2:10.3f}{3:10.3f}'.format(t_t[i], x_t[i], x_s[i], x_da[i]))
        f.write('')

        # Echo: v
        f.write('******* v ********')
        f.write(' [Time]   [True]  [No Assim]  [Assim]')
        for i in range(len(t_t)):
            f.write('{0:7.2f}{1:10.3f}{2:10.3f}{3:10.3f}'.format(t_t[i], v_t[i], v_s[i], v_da[i]))
        f.write('')

    # Plot: x
    plt.plot(t_t  , x_t  , color='k', marker='.', label='True')
    plt.plot(t_obs, x_obs, color='b', marker='.', label='Observation')
    plt.plot(t_s  , x_s  , color='g', marker='.', label='No Assimilation')
    plt.plot(t_da , x_da , color='r', marker='.', label='Assimilation')
    plt.legend(loc='best')
    plt.show()

    # Plot: v
    plt.plot(t_t , v_t , color='k', marker='.', label='True')
    plt.plot(t_s , v_s , color='g', marker='.', label='No Assimilation')
    plt.plot(t_da, v_da, color='r', marker='.', label='Assimilation')
    plt.legend(loc='best')
    plt.show()
