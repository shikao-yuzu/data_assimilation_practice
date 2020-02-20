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

    def predict(self, nt_start: int, nt_end: int) -> None:
        # Forward euler scheme
        for it in range(nt_start, nt_end+1):
            self.x[it] = self.x[it-1] + self.dt * self.v[it-1]
            self.v[it] = - (self.PARAM_K * self.dt / self.PARAM_M) * self.x[it - 1] + \
                         (1.0 - self.PARAM_C * self.dt / self.PARAM_M) * self.v[it - 1]

    def output(self, interval: int) -> tuple:
        t = np.copy(self.t[::interval])
        x = np.copy(self.x[::interval])
        v = np.copy(self.v[::interval])
        return t, x, v


class DA:
    def __init__(self) -> None:
        # Observation error covariance matrix
        self.R = np.zeros((1, 1), dtype=np.float64)
        self.R[0, 0] = 0.1


if __name__ == '__main__':
    # Pre procedure
    da = DA()
    np.random.seed(0)

    # True Field
    mdl_t = Model(nt=NT_ASM+NT_PRD, dt=DT, x_0=5.0, v_0=0.0)
    mdl_t.predict(1, NT_ASM+NT_PRD)
    t_t, x_t, v_t = mdl_t.output(OUTPUT_INTERVAL)

    # Observations
    #   generate observation by adding Gaussian noise (uniform random number) to true value
    t_obs, x_obs, _ = mdl_t.output(OBS_INTERVAL)
    gnoise = np.sqrt(da.R[0, 0]) * np.random.randn(len(t_obs))
    x_obs += gnoise

    # Simulation run without DA (wrong initial value)
    mdl_s = Model(nt=NT_ASM+NT_PRD, dt=DT, x_0=4.0, v_0=1.0)
    mdl_s.predict(1, NT_ASM+NT_PRD)
    t_s, x_s, v_s = mdl_s.output(OUTPUT_INTERVAL)

    # Simulation run with DA (wrong initial value)
    mdl_da = Model(nt=NT_ASM+NT_PRD, dt=DT, x_0=4.0, v_0=1.0)
    mdl_da.predict(1, NT_ASM+NT_PRD)
    t_da, x_da, v_da = mdl_da.output(OUTPUT_INTERVAL)

    # Echo: x
    print('******* x ********')
    print(' [Time]   [True]  [No Assim]  [Assim]')
    for i in range(len(t_t)):
        print('{0:7.2f}{1:10.3f}{2:10.3f}{3:10.3f}'.format(t_t[i], x_t[i], x_s[i], x_da[i]))
    print()

    # Echo: v
    print('******* v ********')
    print(' [Time]   [True]  [No Assim]  [Assim]')
    for i in range(len(t_t)):
        print('{0:7.2f}{1:10.3f}{2:10.3f}{3:10.3f}'.format(t_t[i], v_t[i], v_s[i], v_da[i]))
    print()

    # Plot: x
    plt.plot(t_t  , x_t  , color='k', label='True')
    plt.plot(t_obs, x_obs, color='b', label='Observation')
    plt.plot(t_s  , x_s  , color='g', label='No Assimilation')
    plt.plot(t_da , x_da , color='r', label='Assimilation')
    plt.legend(loc='best')
    plt.show()

    # Plot: v
    plt.plot(t_t , v_t , color='k', label='True')
    plt.plot(t_s , v_s , color='g', label='No Assimilation')
    plt.plot(t_da, v_da, color='r', label='Assimilation')
    plt.legend(loc='best')
    plt.show()
