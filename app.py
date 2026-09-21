from __future__ import annotations

from typing import Any, Dict

import numpy as np
from flask import Flask, render_template, request
from sympy import E, cos, exp, log, pi, sin, sqrt, symbols, tan
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from python.Algebra import (
    analizar_sistema,
    autovalores_autovectores,
    calcular_determinante,
    convertir_formulario,
    derivada_numerica,
    fuerza_lorentz,
    integrar_numerica,
    inversa_matriz,
    matriz_rotacion_2d,
    producto_escalar,
    producto_vectorial,
    simulacion_proyectil,
    transformar_vector,
)

app = Flask(__name__)


def parse_matrix_2x2(form: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            [float(form.get("a", 0)), float(form.get("b", 0))],
            [float(form.get("c", 0)), float(form.get("d", 0))],
        ],
        dtype=float,
    )


def parse_vector_3d(form: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            float(form.get("vx", 0)),
            float(form.get("vy", 0)),
            float(form.get("vz", 0)),
        ],
        dtype=float,
    )


def evaluar_expresion(expr: str) -> float:
    if not expr or not expr.strip():
        raise ValueError("Escribe una funcion valida, por ejemplo x**2 + sin(x)")

    x = symbols("x")
    allowed = {
        "x": x,
        "sin": sin,
        "cos": cos,
        "tan": tan,
        "exp": exp,
        "log": log,
        "sqrt": sqrt,
        "pi": pi,
        "e": E,
        "abs": abs,
    }

    try:
        parsed = parse_expr(
            expr,
            local_dict=allowed,
            transformations=(standard_transformations + (implicit_multiplication_application,)),
            evaluate=True,
        )
    except Exception as exc:
        raise ValueError(f"Expresion no valida: {exc}") from exc

    free_symbols = getattr(parsed, "free_symbols", set())
    if free_symbols - {x}:
        raise ValueError("Solo se permite la variable x y funciones matematicas seguras.")

    try:
        return float(parsed.subs(x, 0).evalf())
    except Exception as exc:
        raise ValueError(f"No se pudo evaluar la expresion: {exc}") from exc




@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    sistema_filas = 2
    sistema_columnas = 2
    sistema_metodo = "reduccion"
    sistema_mostrar = False
    sistema_valores = {}
    sistema_resultado = None

    if request.method == "POST":
        section = request.form.get("section")

        if section == "sistema":
            try:
                sistema_filas = int(request.form.get("filas", 2))
                sistema_columnas = int(request.form.get("columnas", 2))
                if not 1 <= sistema_filas <= 8 or not 1 <= sistema_columnas <= 8:
                    raise ValueError("Las dimensiones deben estar entre 1 y 8.")
                sistema_metodo = request.form.get("metodo", "reduccion")
                sistema_mostrar = True
                sistema_valores = request.form.to_dict(flat=True)
                if request.form.get("accion") == "resolver":
                    coeficientes, independientes = convertir_formulario(request.form, sistema_filas, sistema_columnas)
                    resultado = analizar_sistema(coeficientes, independientes, sistema_metodo)
                    sistema_resultado = (
                        f"Tipo: {resultado['tipo']}\n"
                        f"Rango de A: {resultado['rango_matriz']}\n"
                        f"Rango de [A|b]: {resultado['rango_ampliada']}\n\n"
                        f"Solucion general: {resultado['solucion_general']}\n\n"
                        f"Forma reducida:\n{resultado['reducida']}\n\n"
                        f"Pasos:\n{'\n\n'.join(resultado['pasos'])}"
                    )
            except Exception as exc:
                error = str(exc)
                sistema_resultado = str(exc)

        else:
            try:
                if section == "determinante":
                    matriz = parse_matrix_2x2(request.form)
                    det, pasos = calcular_determinante(matriz)
                    result = (
                        f"Determinante: {det}\n\n"
                        + "Matriz:\n" + np.array2string(matriz, precision=4, suppress_small=False) + "\n\n"
                        + "Pasos:\n" + "\n".join(pasos)
                    )
                elif section == "inversa":
                    matriz = parse_matrix_2x2(request.form)
                    inv, pasos = inversa_matriz(matriz)
                    result = (
                        f"Inversa:\n{np.array2string(inv, precision=4)}\n\n"
                        + "Comprobacion: A·A⁻¹ = I\n"
                        + np.array2string(matriz @ inv, precision=4) + "\n\n"
                        + "Pasos:\n" + "\n".join(pasos)
                    )
                elif section == "eigen":
                    matriz = parse_matrix_2x2(request.form)
                    vals, vecs, checks = autovalores_autovectores(matriz)
                    lines = [f"Autovalores: {vals}"]
                    for item in checks:
                        lines.append(
                            f"λ{item['indice']}: {item['autovalor']} | "
                            f"cumple Av = λv: {item['cumple']}"
                        )
                    result = "\n".join(lines)
                elif section == "vector":
                    u = parse_vector_3d(request.form)
                    ang = float(request.form.get("ang", 0.0))
                    rot = matriz_rotacion_2d(ang)
                    v = transformar_vector(np.array([u[0], u[1]], dtype=float), rot)
                    dot = producto_escalar(u[:2], v)
                    cross = producto_vectorial(np.array([1.0, 0.0, 0.0]), u)
                    result = (
                        f"Vector original: {u}\n"
                        f"Vector transformado (rotacion 2D): {v}\n"
                        f"Producto escalar: {dot}\n"
                        f"Producto vectorial e1 x u: {cross}"
                    )
                elif section == "calculo":
                    expr = request.form.get("expr", "x**2")
                    a = float(request.form.get("a", 0.0))
                    b = float(request.form.get("b", 1.0))
                    if not (a < b):
                        raise ValueError("El intervalo debe cumplir a < b.")

                    x_sym = symbols("x")
                    parsed = parse_expr(
                        expr,
                        local_dict={
                            "x": x_sym,
                            "sin": sin,
                            "cos": cos,
                            "tan": tan,
                            "exp": exp,
                            "log": log,
                            "sqrt": sqrt,
                            "pi": pi,
                            "e": E,
                            "abs": abs,
                        },
                        transformations=(standard_transformations + (implicit_multiplication_application,)),
                        evaluate=True,
                    )
                    if parsed.free_symbols - {x_sym}:
                        raise ValueError("Solo se permite la variable x y funciones matematicas seguras.")

                    f = lambda value: float(parsed.subs(x_sym, value).evalf())
                    deriv = derivada_numerica(f, (a + b) / 2)
                    integral = integrar_numerica(f, a, b)
                    result = f"Funcion: {expr}\nDerivada en x={(a + b)/2}: {deriv}\nIntegral en [{a},{b}]: {integral}"
                elif section == "fisica":
                    v0 = float(request.form.get("v0", 20.0))
                    angulo = float(request.form.get("angulo", 45.0))
                    g = float(request.form.get("g", 9.81))
                    tmax = float(request.form.get("tmax", 5.0))
                    datos = simulacion_proyectil(v0=v0, angulo_deg=angulo, g=g, tiempo_max=tmax)
                    result = (
                        f"Alcance: {datos['alcance']}\n"
                        f"Altura maxima: {datos['altura_maxima']}\n"
                        f"Punto final: {datos['trayectoria'][-1]}"
                    )
                else:
                    raise ValueError("Seccion no valida.")
            except Exception as exc:
                error = str(exc)
                result = None

    return render_template(
        "index.html",
        result=result,
        error=error,
        sistema_filas=sistema_filas,
        sistema_columnas=sistema_columnas,
        sistema_metodo=sistema_metodo,
        sistema_mostrar=sistema_mostrar,
        sistema_valores=sistema_valores,
        sistema_resultado=sistema_resultado,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
