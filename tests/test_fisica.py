import unittest
import numpy as np

from python.Fisica_I import (
    simulacion_proyectil,
    movimiento_mru,
    movimiento_mrua,
    cinematica_nd,
    caida_libre,
    fuerza_lorentz,
    colision_unidimensional,
    oscilador_armonico,
    verlet,
    euler_cromer_second_order,
)


class TestFisica(unittest.TestCase):
    def test_movimiento_mru(self):
        x = movimiento_mru(2.0, 3.0, 4.0)
        self.assertAlmostEqual(x, 2.0 + 3.0 * 4.0)

    def test_movimiento_mrua(self):
        r = movimiento_mrua(0.0, 2.0, -1.0, 3.0)
        self.assertAlmostEqual(r['velocidad'], 2.0 + -1.0 * 3.0)
        self.assertAlmostEqual(r['posicion'], 0.0 + 2.0 * 3.0 + 0.5 * -1.0 * 9.0)

    def test_cinematica_nd(self):
        pos, vel = cinematica_nd(np.array([0.0, 0.0]), np.array([1.0, 2.0]), np.array([0.0, -9.81]), 2.0)
        self.assertEqual(pos.shape, (2,))
        self.assertEqual(vel.shape, (2,))

    def test_caida_libre(self):
        res = caida_libre(4.905, g=9.81)
        # tiempo aproximadamente 1.0
        self.assertAlmostEqual(res['tiempo'], 1.0, places=3)

    def test_simulacion_proyectil_alcance(self):
        datos = simulacion_proyectil(v0=10.0, angulo_deg=45.0, g=9.81, tiempo_max=2.0)
        # alcance aproximado formula v0^2/g for sin(90)=1
        self.assertAlmostEqual(datos['alcance'], (10.0 ** 2) * np.sin(np.radians(90)) / 9.81, places=6)

    def test_fuerza_lorentz(self):
        F = fuerza_lorentz(1.0, [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0])
        # E component plus v x B = [1,0,0] x [0,0,3] = [0,-3,0]
        np.testing.assert_allclose(F, np.array([0.0, -3.0 + 2.0, 0.0]))

    def test_colision_elastica(self):
        v1p, v2p = colision_unidimensional(1.0, 1.0, 1.0, 0.0, elastic=True)
        self.assertAlmostEqual(v1p, 0.0)
        self.assertAlmostEqual(v2p, 1.0)

    def test_oscilador_energy_conservation_rk4(self):
        t = np.linspace(0, 2 * np.pi, 201)
        q, v = oscilador_armonico(1.0, 1.0, 1.0, 0.0, t, method='rk4')
        E = 0.5 * v ** 2 + 0.5 * q ** 2
        self.assertLess(E.max() - E.min(), 1e-6)

    def test_verlet_and_euler_cromer(self):
        # harmonic oscillator as acceleration
        k = 1.0
        m = 1.0
        def a(ti, q):
            return -k / m * q

        t = np.linspace(0, 2 * np.pi, 201)
        qv, vv = verlet(a, np.array([1.0]), np.array([0.0]), t)
        qec, vec = euler_cromer_second_order(a, np.array([1.0]), np.array([0.0]), t)
        # both should produce arrays of correct shape
        self.assertEqual(qv.shape[0], t.size)
        self.assertEqual(qec.shape[0], t.size)


if __name__ == '__main__':
    unittest.main()
