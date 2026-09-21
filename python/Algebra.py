import argparse
from typing import Callable, List, Tuple

import numpy as np
from scipy.integrate import quad


TOLERANCIA = 1e-10
NOMBRES_VARIABLES = ["x", "y", "z", "w", "v", "u", "p", "q"]


def formatear_matriz(matriz: np.ndarray) -> str:
    filas = []
    for fila in matriz:
        valores = [0.0 if abs(valor) < TOLERANCIA else valor for valor in fila]
        filas.append("[ " + "  ".join(f"{valor:8.4f}" for valor in valores) + " ]")
    return "\n".join(filas)


def gauss_jordan(matriz_ampliada: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    matriz = matriz_ampliada.astype(float).copy()
    filas, columnas = matriz.shape
    pasos = ["Matriz inicial:\n" + formatear_matriz(matriz)]
    fila_pivote = 0

    for columna in range(columnas - 1):
        if fila_pivote >= filas:
            break
        posicion = fila_pivote + int(np.argmax(np.abs(matriz[fila_pivote:, columna])))
        if abs(matriz[posicion, columna]) < TOLERANCIA:
            continue
        if posicion != fila_pivote:
            matriz[[fila_pivote, posicion]] = matriz[[posicion, fila_pivote]]
            pasos.append(f"Intercambio F{fila_pivote + 1} <-> F{posicion + 1}:\n" + formatear_matriz(matriz))

        pivote = matriz[fila_pivote, columna]
        if not np.isclose(pivote, 1.0):
            matriz[fila_pivote] /= pivote
            pasos.append(f"F{fila_pivote + 1} <- F{fila_pivote + 1} / {pivote:.4g}:\n" + formatear_matriz(matriz))

        for fila in range(filas):
            if fila == fila_pivote:
                continue
            factor = matriz[fila, columna]
            if abs(factor) >= TOLERANCIA:
                matriz[fila] -= factor * matriz[fila_pivote]
                pasos.append(f"F{fila + 1} <- F{fila + 1} - ({factor:.4g}) F{fila_pivote + 1}:\n" + formatear_matriz(matriz))
        fila_pivote += 1

    matriz[np.abs(matriz) < TOLERANCIA] = 0.0
    pasos.append("Matriz reducida por Gauss-Jordan:\n" + formatear_matriz(matriz))
    return matriz, pasos


def calcular_determinante(matriz: np.ndarray) -> Tuple[float, List[str]]:
    """Calcula el determinante por eliminacion de Gauss-Jordan con control de cambios de fila."""
    if matriz.ndim != 2 or matriz.shape[0] != matriz.shape[1]:
        raise ValueError("La matriz debe ser cuadrada para calcular su determinante.")

    n = matriz.shape[0]
    matriz_reducida = matriz.astype(float).copy()
    pasos = ["Determinante: matriz inicial\n" + formatear_matriz(matriz_reducida)]
    signo = 1.0
    determinante = 1.0

    for columna in range(n):
        fila_pivote = columna + int(np.argmax(np.abs(matriz_reducida[columna:, columna])))
        if abs(matriz_reducida[fila_pivote, columna]) < TOLERANCIA:
            return 0.0, pasos + ["El determinante es cero porque aparece un pivote nulo."]

        if fila_pivote != columna:
            matriz_reducida[[columna, fila_pivote]] = matriz_reducida[[fila_pivote, columna]]
            signo *= -1
            pasos.append(f"Intercambio F{columna + 1} <-> F{fila_pivote + 1}:\n" + formatear_matriz(matriz_reducida))

        pivote = matriz_reducida[columna, columna]
        determinante *= pivote
        pasos.append(f"Pivote en la columna {columna + 1}: {pivote:.6g}. Determinante parcial = {determinante:.6g}")

        if not np.isclose(pivote, 1.0):
            matriz_reducida[columna] /= pivote
            pasos.append(f"F{columna + 1} <- F{columna + 1} / {pivote:.6g}:\n" + formatear_matriz(matriz_reducida))

        for fila in range(columna + 1, n):
            factor = matriz_reducida[fila, columna]
            if abs(factor) >= TOLERANCIA:
                matriz_reducida[fila] -= factor * matriz_reducida[columna]
                pasos.append(f"F{fila + 1} <- F{fila + 1} - ({factor:.6g}) F{columna + 1}:\n" + formatear_matriz(matriz_reducida))

    determinante_final = signo * determinante
    pasos.append(f"Determinante final: det(A) = {determinante_final:.6g}")
    return determinante_final, pasos


def inversa_matriz(matriz: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Calcula la matriz inversa usando la ampliada [A|I] y Gauss-Jordan."""
    if matriz.ndim != 2 or matriz.shape[0] != matriz.shape[1]:
        raise ValueError("La matriz debe ser cuadrada para calcular su inversa.")

    n = matriz.shape[0]
    matriz_original = matriz.astype(float).copy()
    identidad = np.eye(n)
    ampliada = np.hstack((matriz_original, identidad))
    pasos = ["Matriz aumentada inicial [A|I]:\n" + formatear_matriz(ampliada)]

    if abs(np.linalg.det(matriz_original)) < TOLERANCIA:
        return np.zeros_like(matriz_original), pasos + ["La matriz no es invertible porque su determinante es cero."]

    fila_pivote = 0
    for columna in range(n):
        if fila_pivote >= n:
            break

        posicion = fila_pivote + int(np.argmax(np.abs(ampliada[fila_pivote:, columna])))
        if abs(ampliada[posicion, columna]) < TOLERANCIA:
            continue

        if posicion != fila_pivote:
            ampliada[[fila_pivote, posicion]] = ampliada[[posicion, fila_pivote]]
            pasos.append(f"Intercambio F{fila_pivote + 1} <-> F{posicion + 1}:\n" + formatear_matriz(ampliada))

        pivote = ampliada[fila_pivote, columna]
        ampliada[fila_pivote] /= pivote
        pasos.append(f"F{fila_pivote + 1} <- F{fila_pivote + 1} / {pivote:.6g}:\n" + formatear_matriz(ampliada))

        for fila in range(n):
            if fila == fila_pivote:
                continue
            factor = ampliada[fila, columna]
            if abs(factor) >= TOLERANCIA:
                ampliada[fila] -= factor * ampliada[fila_pivote]
                pasos.append(f"F{fila + 1} <- F{fila + 1} - ({factor:.6g}) F{fila_pivote + 1}:\n" + formatear_matriz(ampliada))

        fila_pivote += 1

    inversa = ampliada[:, n:]
    inversa[np.abs(inversa) < TOLERANCIA] = 0.0
    pasos.append("Matriz inversa final:\n" + formatear_matriz(inversa))
    return inversa, pasos


def autovalores_autovectores(matriz: np.ndarray):
    """Calcula autovalores y autovectores usando NumPy y verifica Av = λv."""
    if matriz.ndim != 2 or matriz.shape[0] != matriz.shape[1]:
        raise ValueError("La matriz debe ser cuadrada para calcular autovalores y autovectores.")

    autovalores, autovectores = np.linalg.eig(matriz)
    verificaciones = []
    for indice, valor in enumerate(autovalores):
        vector = autovectores[:, indice]
        residuo = matriz @ vector - valor * vector
        verificaciones.append({
            "indice": indice,
            "autovalor": complex(valor),
            "autovector": np.asarray(vector, dtype=complex),
            "residuo": np.asarray(residuo, dtype=complex),
            "cumple": bool(np.allclose(residuo, 0.0 + 0.0j, atol=1e-8))
        })
    return autovalores, autovectores, verificaciones


def producto_escalar(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != v.shape:
        raise ValueError("Los vectores deben tener la misma dimension.")
    return float(np.dot(u, v))


def norma_vector(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    return float(np.linalg.norm(v))


def producto_vectorial(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != (3,) or v.shape != (3,):
        raise ValueError("El producto vectorial solo esta definido para vectores 3D.")
    return np.cross(u, v)


def transformar_vector(vector: np.ndarray, matriz: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    matriz = np.asarray(matriz, dtype=float)
    if vector.size != matriz.shape[0]:
        raise ValueError("La dimension del vector debe coincidir con las filas de la matriz.")
    return matriz @ vector


def matriz_rotacion_2d(angulo_rad: float) -> np.ndarray:
    c = np.cos(angulo_rad)
    s = np.sin(angulo_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def matriz_escala_2d(fx: float, fy: float) -> np.ndarray:
    return np.array([[fx, 0.0], [0.0, fy]], dtype=float)


def derivada_numerica(funcion: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    return (funcion(x + h) - funcion(x - h)) / (2 * h)


def integrar_numerica(funcion: Callable[[float], float], a: float, b: float, pasos: int = 1000) -> float:
    xs = np.linspace(a, b, pasos)
    ys = np.array([funcion(x) for x in xs], dtype=float)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(ys, xs))
    return float(np.trapz(ys, xs))


def representar_funcion(funcion: Callable[[float], float], a: float, b: float, puntos: int = 400):
    xs = np.linspace(a, b, puntos)
    ys = np.array([funcion(x) for x in xs], dtype=float)
    return xs, ys


def simulacion_proyectil(v0: float = 20.0, angulo_deg: float = 45.0, g: float = 9.81, tiempo_max: float = 5.0):
    angulo = np.radians(angulo_deg)
    t = np.linspace(0, tiempo_max, 400)
    x = v0 * np.cos(angulo) * t
    y = v0 * np.sin(angulo) * t - 0.5 * g * t ** 2
    alcance = (v0 ** 2 * np.sin(2 * angulo)) / g
    altura_maxima = (v0 ** 2 * np.sin(angulo) ** 2) / (2 * g)
    return {
        "tiempo": t,
        "x": x,
        "y": y,
        "alcance": alcance,
        "altura_maxima": altura_maxima,
        "trayectoria": np.column_stack((x, y))
    }


def fuerza_lorentz(carga: float, velocidad: np.ndarray, campo_electrico: np.ndarray, campo_magnetico: np.ndarray):
    velocidad = np.asarray(velocidad, dtype=float)
    campo_electrico = np.asarray(campo_electrico, dtype=float)
    campo_magnetico = np.asarray(campo_magnetico, dtype=float)
    if velocidad.shape != (3,) or campo_electrico.shape != (3,) or campo_magnetico.shape != (3,):
        raise ValueError("La velocidad y los campos deben ser vectores tridimensionales.")
    return carga * (campo_electrico + np.cross(velocidad, campo_magnetico))


def resolucion_integral_definida(funcion: Callable[[float], float], a: float, b: float):
    valor, error = quad(funcion, a, b)
    return {"valor": valor, "error": error}


def resolver_por_igualacion(coeficientes: np.ndarray, independientes: np.ndarray) -> List[str]:
    if coeficientes.shape != (2, 2):
        return ["El metodo de igualacion solo esta disponible para sistemas 2x2."]

    a1, b1 = coeficientes[0]
    a2, b2 = coeficientes[1]
    c1, c2 = independientes
    pasos = ["Metodo de igualacion para el sistema 2x2:"]

    if abs(a1) >= TOLERANCIA and abs(a2) >= TOLERANCIA:
        variable = "x"
        paso1 = f"x = ({c1:g} - ({b1:g})y) / ({a1:g})"
        paso2 = f"x = ({c2:g} - ({b2:g})y) / ({a2:g})"
        pasos.append(f"Despejamos x en la ecuacion 1: {paso1}")
        pasos.append(f"Despejamos x en la ecuacion 2: {paso2}")
        pasos.append(f"Igualamos las expresiones: ({c1:g} - ({b1:g})y) / ({a1:g}) = ({c2:g} - ({b2:g})y) / ({a2:g})")
        determinante = a1 * b2 - a2 * b1
        if abs(determinante) < TOLERANCIA:
            return pasos + [
                "El determinante es cero, por lo que el sistema no es compatible determinado y no puede resolverse por igualacion."
            ]
        y = (a1 * c2 - a2 * c1) / determinante
        x = (c1 - b1 * y) / a1
        pasos.append(f"Resolvemos para y: y = ({c1:g}*{a2:g} - {c2:g}*{a1:g}) / ({a1:g}*{b2:g} - {a2:g}*{b1:g}) = {y:.6g}")
        pasos.append(f"Sustituimos en la ecuacion 1: x = ({c1:g} - ({b1:g})*{y:.6g}) / ({a1:g}) = {x:.6g}")
        pasos.append(f"Solucion: x = {x:.6g}, y = {y:.6g}")
        return pasos

    if abs(b1) >= TOLERANCIA and abs(b2) >= TOLERANCIA:
        variable = "y"
        paso1 = f"y = ({c1:g} - ({a1:g})x) / ({b1:g})"
        paso2 = f"y = ({c2:g} - ({a2:g})x) / ({b2:g})"
        pasos.append(f"Despejamos y en la ecuacion 1: {paso1}")
        pasos.append(f"Despejamos y en la ecuacion 2: {paso2}")
        pasos.append(f"Igualamos las expresiones: ({c1:g} - ({a1:g})x) / ({b1:g}) = ({c2:g} - ({a2:g})x) / ({b2:g})")
        determinante = a1 * b2 - a2 * b1
        if abs(determinante) < TOLERANCIA:
            return pasos + [
                "El determinante es cero, por lo que el sistema no es compatible determinado y no puede resolverse por igualacion."
            ]
        x = (c1 * b2 - c2 * b1) / determinante
        y = (c1 - a1 * x) / b1
        pasos.append(f"Resolvemos para x: x = ({c1:g}*{b2:g} - {c2:g}*{b1:g}) / ({a1:g}*{b2:g} - {a2:g}*{b1:g}) = {x:.6g}")
        pasos.append(f"Sustituimos en la ecuacion 1: y = ({c1:g} - ({a1:g})*{x:.6g}) / ({b1:g}) = {y:.6g}")
        pasos.append(f"Solucion: x = {x:.6g}, y = {y:.6g}")
        return pasos

    return ["No se puede aplicar igualacion: no hay una variable despejable en ambas ecuaciones."]


def analizar_sistema(coeficientes: List[List[float]], independientes: List[float], metodo: str = "reduccion"):
    matriz = np.array(coeficientes, dtype=float)
    vector_independientes = np.array(independientes, dtype=float)
    ampliada = np.column_stack((matriz, vector_independientes))
    reducida, pasos = gauss_jordan(ampliada)
    filas, variables = matriz.shape
    rango_matriz = sum(any(abs(valor) >= TOLERANCIA for valor in reducida[fila, :variables]) for fila in range(filas))
    rango_ampliada = sum(any(abs(valor) >= TOLERANCIA for valor in reducida[fila]) for fila in range(filas))

    if metodo == "igualacion" and rango_matriz == rango_ampliada == 2 and matriz.shape == (2, 2):
        pasos = resolver_por_igualacion(matriz, vector_independientes)
    elif metodo == "igualacion":
        pasos = ["El metodo de igualacion requiere un sistema 2x2 compatible determinado."]

    if rango_matriz < rango_ampliada:
        tipo = "Sistema incompatible: no tiene solucion."
        solucion_general = "No existe solucion general."
    else:
        pivotes = []
        for fila in range(filas):
            posiciones = np.where(np.abs(reducida[fila, :variables]) >= TOLERANCIA)[0]
            if len(posiciones):
                pivotes.append(int(posiciones[0]))
        libres = [columna for columna in range(variables) if columna not in pivotes]
        if rango_matriz == variables:
            tipo = "Sistema compatible determinado: tiene una solucion unica."
            solucion_general = ", ".join(
                f"{NOMBRES_VARIABLES[columna]} = {reducida[fila, -1]:.6g}"
                for fila, columna in enumerate(pivotes)
            )
        else:
            tipo = "Sistema compatible indeterminado: tiene infinitas soluciones."
            parametros = {columna: f"t{indice + 1}" for indice, columna in enumerate(libres)}
            expresiones = []
            for columna in range(variables):
                nombre = NOMBRES_VARIABLES[columna]
                if columna in parametros:
                    expresiones.append(f"{nombre} = {parametros[columna]}")
                    continue
                fila = pivotes.index(columna)
                expresion = f"{reducida[fila, -1]:.6g}"
                for libre in libres:
                    coeficiente = reducida[fila, libre]
                    if abs(coeficiente) >= TOLERANCIA:
                        signo = " - " if coeficiente > 0 else " + "
                        expresion += f"{signo}{abs(coeficiente):.6g}*{parametros[libre]}"
                expresiones.append(f"{nombre} = {expresion}")
            solucion_general = ", ".join(expresiones)

    return {"reducida": reducida, "rango_matriz": rango_matriz, "rango_ampliada": rango_ampliada,
        "tipo": tipo, "solucion_general": solucion_general, "pasos": pasos, "metodo": metodo}


def convertir_formulario(formulario, filas: int, columnas: int):
    coeficientes = []
    independientes = []
    for fila in range(filas):
        coeficientes.append([float(formulario[f"a_{fila}_{columna}"]) for columna in range(columnas)])
        independientes.append(float(formulario[f"b_{fila}"]))
    return coeficientes, independientes


def leer_entero_validado(prompt: str, minimo: int = 1, maximo: int = 8) -> int:
    while True:
        try:
            valor = int(input(prompt))
            if not minimo <= valor <= maximo:
                raise ValueError(f"el valor debe estar entre {minimo} y {maximo}.")
            return valor
        except ValueError as exc:
            print(f"Entrada invalida: {exc}")
            print(f"Intentalo de nuevo.")

PLANTILLA = None


def resolver_consola():
    print("=== RESOLVEDOR DE SISTEMAS LINEALES ===")
    try:
        filas = leer_entero_validado("Numero de ecuaciones: ", 1, 8)
        columnas = leer_entero_validado("Numero de incognitas: ", 1, 8)
    except KeyboardInterrupt:
        print("\nOperacion cancelada por el usuario.")
        return

    metodo = input("Metodo (igualacion/reduccion): ").strip().lower()
    if metodo not in {"igualacion", "reduccion"}:
        raise ValueError("El metodo debe ser 'igualacion' o 'reduccion'.")

    coeficientes, independientes = [], []
    for fila in range(filas):
        print(f"\nEcuacion {fila + 1}")
        fila_coeficientes = []
        for columna in range(columnas):
            try:
                fila_coeficientes.append(float(input(f"Coeficiente de {NOMBRES_VARIABLES[columna]}: ")))
            except ValueError:
                raise ValueError(f"El coeficiente de {NOMBRES_VARIABLES[columna]} debe ser numerico.")
        coeficientes.append(fila_coeficientes)
        try:
            independientes.append(float(input("Termino independiente: ")))
        except ValueError:
            raise ValueError(f"El termino independiente de la ecuacion {fila + 1} debe ser numerico.")

    resultado = analizar_sistema(coeficientes, independientes, metodo)
    print(f"\n{resultado['tipo']}\nRango de A: {resultado['rango_matriz']}\nRango de [A|b]: {resultado['rango_ampliada']}")
    print(f"Solucion general: {resultado['solucion_general']}\n\nPasos de Gauss-Jordan:\n" + "\n\n".join(resultado["pasos"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resuelve sistemas lineales por rangos.")
    parser.add_argument("--cli", action="store_true", help="Usar la version de consola.")
    args = parser.parse_args()
    if args.cli:
        resolver_consola()
    else:
        print("Este modulo contiene utilidades de algebra; ejecuta 'app.py' para la interfaz web.")