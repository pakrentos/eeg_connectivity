import numpy as np

def get_rossler(z0=np.zeros(6), epsylon=[0., 0.03], a=0.15, p=0.2, c=10, omega=[0.99, 0.95], fps=250, time=(0, 200)):
    def rossler(t, vec: np.array):
        res = np.zeros(6)
        x = vec[0:4:3]
        y = vec[1:5:3]
        z = vec[2:6:3]
        res[0:4:3] = -omega*y - z + epsylon*(x[::-1] - x)
        res[1:5:3] = omega*x + a*y
        res[2:6:3] = p + z*(x-c)
        return res
    omega = np.array(omega)
    epsylon = np.array(epsylon)
    start_time = time[0]
    end_time = time[1]
    X1 = 0
    Y1 = 1
    Z1 = 2
    X2 = 3
    Y2 = 4
    Z2 = 5
    t=np.linspace(start_time, end_time, num=int(fps*end_time))
    # z0 = np.array([ 4.2930454 ,  2.56096433,  0.03710937, -4.2591115 , -3.92001009, 0.01380057])
    # z0 = np.zeros(6)
    res1 = ivp(rossler, (start_time,end_time), z0, t_eval=t)
    y = res1['y']
    t_span = res1['t']
    return t_span, y