import numpy as np
import matplotlib.pyplot as plt

g = 9.81
v0 = 20
angulo = np.radians(45)

t = np.linspace(0, 2.8, 200)

x = v0 * np.cos(angulo) * t
y = v0 * np.sin(angulo) * t - 0.5 * g * t**2

plt.plot(x, y)
plt.xlabel("Distancia (m)")
plt.ylabel("Altura (m)")
plt.title("Movimiento parabólico")
plt.grid()
plt.show()