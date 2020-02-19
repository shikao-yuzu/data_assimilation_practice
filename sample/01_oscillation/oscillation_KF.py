import numpy as np
import matplotlib.pyplot as plt
import pyda

# Period of data assimilation
NT_ASM = 400
# Period of prediction
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
        self.nt = nt
        self.dt = dt

        self.t = np.arange(0.0, (self.nt+1)*self.dt, self.dt)
        self.x = np.zeros(self.nt+1, dtype=np.float64)
        self.v = np.zeros(self.nt+1, dtype=np.float64)

        self.x[0] = x_0
        self.v[0] = v_0

    def predict(self) -> None:
        # Forward euler scheme
        for it in range(1, self.nt+1):
            self.x[it] = self.x[it-1] + self.dt * self.v[it-1]
            self.v[it] = - (self.PARAM_K * self.dt / self.PARAM_M) * self.x[it - 1] + \
                         (1.0 - self.PARAM_C * self.dt / self.PARAM_M) * self.v[it - 1]

    def output(self, interval: int) -> tuple:
        return self.t[::interval], self.x[::interval], self.v[::interval]


class DA:
    def __init__(self) -> None:
        pass


if __name__ == '__main__':
    mdl_t = Model(nt=NT_ASM+NT_PRD, dt=DT, x_0=5.0, v_0=0.0)
    mdl_t.predict()
    t, x_t, v_t = mdl_t.output(OUTPUT_INTERVAL)

    print('******* x ********')
    for i in range(len(t)):
        print('{0:7.2f}{1:10.3f}'.format(t[i], x_t[i]))

    print()
    print('******* v ********')
    for i in range(len(t)):
        print('{0:7.2f}{1:10.3f}'.format(t[i], v_t[i]))

    plt.plot(x_t, v_t)
    plt.show()
