import numpy as np
from python.Fisica_I import oscilador_armonico

if __name__ == '__main__':
    t = np.linspace(0, 10, 1001)
    q, v = oscilador_armonico(m=1.0, k=4.0, q0=1.0, v0=0.0, t=t, method='rk4')
    E = 0.5 * 1.0 * v**2 + 0.5 * 4.0 * q**2
    print(f"Energia (min,max) = ({E.min():.6f}, {E.max():.6f}), variacion={E.max()-E.min():.6g}")
