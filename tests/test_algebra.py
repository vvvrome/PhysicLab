import unittest

import numpy as np

from app import app, evaluar_expresion
from python.Algebra import analizar_sistema, calcular_determinante, inversa_matriz


class PhysicsLabTests(unittest.TestCase):
    def test_calcular_determinante_2x2(self):
        matriz = np.array([[4.0, 3.0], [2.0, 1.0]])
        determinante, pasos = calcular_determinante(matriz)

        self.assertTrue(np.isclose(determinante, -2.0))
        self.assertTrue(pasos)
        self.assertTrue(any("determinante" in paso.lower() for paso in pasos))

    def test_inversa_matriz_2x2(self):
        matriz = np.array([[4.0, 7.0], [2.0, 6.0]])
        inversa, pasos = inversa_matriz(matriz)

        self.assertEqual(inversa.shape, (2, 2))
        self.assertTrue(np.allclose(matriz @ inversa, np.eye(2), atol=1e-8))
        self.assertTrue(np.allclose(inversa @ matriz, np.eye(2), atol=1e-8))
        self.assertTrue(pasos)

    def test_analizar_sistema_lineal(self):
        resultado = analizar_sistema([[2, 1], [1, 1]], [5, 3], "reduccion")
        self.assertIn("Sistema compatible determinado", resultado["tipo"])
        self.assertIn("x = 2", resultado["solucion_general"])
        self.assertIn("y = 1", resultado["solucion_general"])

    def test_evaluar_expresion_segura(self):
        self.assertAlmostEqual(evaluar_expresion("x**2 + sin(x)"), 0.0, places=9)
        self.assertAlmostEqual(evaluar_expresion("sqrt(9)"), 3.0, places=9)

    def test_formulario_sistema_web(self):
        client = app.test_client()
        respuesta = client.post(
            "/",
            data={
                "section": "sistema",
                "filas": "2",
                "columnas": "2",
                "metodo": "reduccion",
                "accion": "resolver",
                "a_0_0": "2",
                "a_0_1": "1",
                "b_0": "5",
                "a_1_0": "1",
                "a_1_1": "1",
                "b_1": "3",
            },
        )
        self.assertEqual(respuesta.status_code, 200)
        self.assertIn("Sistema compatible determinado", respuesta.get_data(as_text=True))

    def test_formulario_invalido_web(self):
        client = app.test_client()
        respuesta = client.post(
            "/",
            data={
                "section": "sistema",
                "filas": "9",
                "columnas": "2",
                "metodo": "reduccion",
                "accion": "generar",
            },
        )
        self.assertEqual(respuesta.status_code, 200)
        self.assertIn("entre 1 y 8", respuesta.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
