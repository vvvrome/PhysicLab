import numpy as np
from scipy.integrate import quad

f = lambda x: x**2

resultado, error = quad(f, 0, 3)

print("Integral:", resultado)
print("Error estimado:", error)