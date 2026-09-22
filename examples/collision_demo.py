from python.Fisica_I import colision_unidimensional

if __name__ == '__main__':
    v1p, v2p = colision_unidimensional(1.0, 1.0, 1.0, 0.0, elastic=True)
    print(f"Velocidades despues de colision (elastica): v1'={v1p:.3f}, v2'={v2p:.3f}")
