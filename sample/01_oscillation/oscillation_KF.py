import numpy as np
import matplotlib.pyplot as plt
import pyda

# 同化ステップ数
# NT_ASM = 1000
NT_ASM = 400
# 予報ステップ数
# NT_PRD = 1000
NT_PRD = 400
# 観測値のステップ間隔
OBS_INTERVAL = 40
# 出力のステップ間隔
OUTPUT_INTERVAL = 20
# 時間ステップ
DT = 0.01


class Model:
    # 1自由度減衰自由振動系
    #   m*ddx + c*dx + k*x = 0

    def __init__(self, nt: int, dt: float, x_0: float, v_0: float) -> None:
        # モデルパラメータ
        self.PARAM_M = 1.0  # 質量m
        self.PARAM_C = 0.3  # 減衰係数c
        self.PARAM_K = 0.5  # ばね剛性k

        self.dt = dt                                      # 時間刻みdt
        self.t = np.arange(0.0, (nt+1)*self.dt, self.dt)  # 時刻t
        self.x = np.zeros(nt+1, dtype=np.float64)         # 変位x
        self.v = np.zeros(nt+1, dtype=np.float64)         # 速度dx

        self.x[0] = x_0  # 変位の初期値x0
        self.v[0] = v_0  # 速度の初期値dx0

    def predict(self, it_start: int, it_end: int) -> None:
        '''
        it_startステップからit_endステップまで数値積分を行う
        '''
        for it in range(it_start, it_end+1):
            self.predict_one_step(it)

    def predict_one_step(self, it: int) -> None:
        '''
        1ステップ間(it-1 -> it)の数値積分を行う
        <オイラー法(陽解法)>
        '''
        self.x[it] = self.x[it - 1] + self.dt * self.v[it - 1]
        self.v[it] = - (self.PARAM_K * self.dt / self.PARAM_M) * self.x[it - 1] + \
                     (1.0 - self.PARAM_C * self.dt / self.PARAM_M) * self.v[it - 1]

    def output(self, interval: int) -> tuple:
        """
        ステップ間隔intervalについて結果の配列をdeep copyして返す
        """
        t = np.copy(self.t[::interval])
        x = np.copy(self.x[::interval])
        v = np.copy(self.v[::interval])
        return t, x, v

    def get_m_matrix(self) -> np.ndarray:
        '''
        状態遷移行列[M] <Nx x Nx>を作成する
        '''
        M = np.zeros((2, 2), dtype=np.float64)
        M[0, 0] = 1.0
        M[0, 1] = self.dt
        M[1, 0] = - self.PARAM_K * self.dt / self.PARAM_M
        M[1, 1] = 1.0 - self.PARAM_C * self.dt / self.PARAM_M
        return M


class Observation:
    def __init__(self, R: np.ndarray, H: np.ndarray, t: np.ndarray, x: np.ndarray) -> None:
        self.R = np.copy(R)  # 観測誤差共分散行列[R] <No x No>
        self.H = np.copy(H)  # 観測行列[H] <No x Nx>
        self.t = np.copy(t)  # 時刻t
        self.x = np.copy(x)  # 変位x

        # 真値に正規乱数による誤差を加えて観測値を生成
        self.x += np.sqrt(self.R[0, 0]) * np.random.randn(len(self.t))


class DA:
    def __init__(self, mdl: Model) -> None:
        self.mdl = mdl

    def assimilation_KF(self, it_start: int, it_end: int, obs: Observation) -> None:
        '''
        カルマンフィルタによるデータ同化
        '''
        # 予報誤差共分散行列[Pf] <Nx x Nx>の初期値
        Pf = np.zeros((2, 2), dtype=np.float64)
        Pf[0, 0] = 1.0
        Pf[1, 1] = 1.0

        # it_startステップからit_endステップまで同化を行う
        for it in range(it_start, it_end+1):
            # 数値積分(it-1 -> it)
            self.mdl.predict_one_step(it)

            # 状態遷移行列[M] <Nx x Nx>
            M = self.mdl.get_m_matrix()

            # リアプノフ方程式: 予報誤差共分散行列[Pf]の時間発展
            Pf = M * Pf * M.transpose()

            # 解析値と同一時刻の観測値を検索して同化
            for it_obs, t_obs in enumerate(obs.t):
                if self.mdl.t[it] == t_obs:
                    # カルマンゲイン[K] <Nx x No>
                    Kg = np.zeros((2, 2), dtype=np.float64)
                    Kg[0, 0] = Pf[0, 0] / (obs.R[0, 0] + Pf[0, 0])
                    Kg[1, 0] = Pf[1, 0] / (obs.R[0, 0] + Pf[0, 0])

                    # カルマンゲインに応じた値の修正
                    #   {xa} = {xf} + [K]({y} - [H]{xf})
                    #     {xa}: 状態ベクトル(同化後の解析値) <Nx x 1>
                    #     {xf}: 状態ベクトル(同化前の予報値) <Nx x 1>
                    #     {y} : 観測値ベクトル               <No x 1>
                    self.mdl.x[it] = self.mdl.x[it] + Kg[0, 0] * (obs.x[it_obs] - self.mdl.x[it])
                    self.mdl.v[it] = self.mdl.v[it] + Kg[1, 0] * (obs.x[it_obs] - self.mdl.x[it])

                    # 解析誤差共分散行列[Pa] <Nx x Nx>
                    #   [Pa] = [Pf] - [K][H][Pf]
                    Pa = Pf - Kg * obs.H * Pf

                    # 予報誤差共分散行列[Pf]の更新
                    Pf = Pa


if __name__ == '__main__':
    # 乱数のシード設定
    np.random.seed(0)

    # 真値
    mdl_t = Model(nt=NT_ASM+NT_PRD, dt=DT, x_0=5.0, v_0=0.0)
    mdl_t.predict(1, NT_ASM+NT_PRD)
    t_t, x_t, v_t = mdl_t.output(OUTPUT_INTERVAL)

    # 観測誤差共分散行列[R] <No x No>
    R = np.zeros((1, 1), dtype=np.float64)
    R[0, 0] = 0.1

    # 観測行列[H] <No x Nx>
    H = np.zeros((1, 2), dtype=np.float64)
    H[0, 0] = 1.0

    # 観測値: 真値に正規乱数による誤差を加えたもの
    t_obs, x_obs, _ = mdl_t.output(OBS_INTERVAL)
    obs = Observation(R, H, t_obs, x_obs)

    # 誤った初期値による数値解析解(データ同化なし)
    mdl_s = Model(nt=NT_ASM+NT_PRD, dt=DT, x_0=4.0, v_0=1.0)
    mdl_s.predict(1, NT_ASM+NT_PRD)
    t_s, x_s, v_s = mdl_s.output(OUTPUT_INTERVAL)

    # 誤った初期値による数値解析解(データ同化あり)
    mdl_da = Model(nt=NT_ASM+NT_PRD, dt=DT, x_0=4.0, v_0=1.0)
    da = DA(mdl_da)
    da.assimilation_KF(1, NT_ASM, obs)
    mdl_da.predict(NT_ASM+1, NT_ASM+NT_PRD)
    t_da, x_da, v_da = mdl_da.output(OUTPUT_INTERVAL)

    # ファイル出力
    with open('result.txt', mode='wt', encoding='cp932') as f:
        f.writelines('******* x ********\n')
        f.write(' [Time]   [True]  [No Assim]  [Assim]\n')
        for i in range(len(t_t)):
            f.write('{0:7.2f}{1:10.3f}{2:10.3f}{3:10.3f}\n'.format(t_t[i], x_t[i], x_s[i], x_da[i]))
        f.write('\n')

        f.write('******* v ********\n')
        f.write(' [Time]   [True]  [No Assim]  [Assim]\n')
        for i in range(len(t_t)):
            f.write('{0:7.2f}{1:10.3f}{2:10.3f}{3:10.3f}\n'.format(t_t[i], v_t[i], v_s[i], v_da[i]))
        f.write('\n')

    # プロット: x
    plt.plot(t_t  , x_t  , color='k', marker='.', label='True')
    plt.plot(t_obs, x_obs, color='b', marker='.', label='Observation')
    plt.plot(t_s  , x_s  , color='g', marker='.', label='No Assimilation')
    plt.plot(t_da , x_da , color='r', marker='.', label='Assimilation')
    plt.legend(loc='best')
    plt.show()

    # プロット: v
    plt.plot(t_t , v_t , color='k', marker='.', label='True')
    plt.plot(t_s , v_s , color='g', marker='.', label='No Assimilation')
    plt.plot(t_da, v_da, color='r', marker='.', label='Assimilation')
    plt.legend(loc='best')
    plt.show()
