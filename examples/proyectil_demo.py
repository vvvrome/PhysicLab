from python.Fisica_I import simulacion_proyectil

if __name__ == '__main__':
    datos = simulacion_proyectil(v0=30.0, angulo_deg=40.0, tiempo_max=5.0)
    print(f"Alcance teórico: {datos['alcance']:.3f} m")
    print(f"Altura maxima: {datos['altura_maxima']:.3f} m")
    print(f"Puntos de trayectoria: {len(datos['trayectoria'])}")
