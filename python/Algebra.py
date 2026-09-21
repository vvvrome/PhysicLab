import argparse
from typing import List, Tuple

import numpy as np
from flask import Flask, render_template_string, request


TOLERANCIA = 1e-10
NOMBRES_VARIABLES = ["x", "y", "z", "w", "v", "u", "p", "q"]
app = Flask(__name__)


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
            valor = float(input(prompt))
            if not minimo <= valor <= maximo:
                raise ValueError(f"el valor debe estar entre {minimo} y {maximo}.")
            return valor
        except ValueError as exc:
            print(f"Entrada invalida: {exc}")
            print(f"Intentalo de nuevo.")

PLANTILLA = """<!doctype html>
<html lang="es">

<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <title>Algebra I | Sistemas lineales</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            font-family: system-ui, sans-serif;
            background: #101820;
            color: #f4f1de;
            margin: 0;
            padding: 32px;
        }

        main {
            max-width: 1100px;
            margin: auto;
        }

        h1 {
            color: #f2c14e;
        }

        h2 {
            color: #3fa7d6;
            margin-top: 24px;
        }

        section {
            background: #182832;
            border: 1px solid #31505c;
            border-radius: 10px;
            padding: 22px;
            margin: 18px 0;
        }

        .config {
            display: flex;
            gap: 12px;
            align-items: end;
            flex-wrap: wrap;
        }

        .campo {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        input,
        select {
            padding: 9px;
            border: 1px solid #55727d;
            border-radius: 5px;
            background: #0f1b21;
            color: white;
            font-size: 14px;
        }

        input[type="number"] {
            width: 88px;
        }

        button {
            padding: 10px 16px;
            border: 0;
            border-radius: 5px;
            background: #f2c14e;
            color: #101820;
            font-weight: 700;
            cursor: pointer;
        }

        button:hover {
            background: #ffd166;
        }

        table {
            border-collapse: collapse;
            margin-top: 14px;
        }

        td {
            padding: 4px;
        }

        td input {
            width: 72px;
        }

        pre {
            overflow: auto;
            background: #0b1318;
            border-left: 3px solid #f2c14e;
            padding: 14px;
            line-height: 1.45;
            white-space: pre-wrap;
        }

        .resultado {
            border-left: 4px solid #3fa7d6;
        }

        .error {
            color: #ff8b8b;
        }

        .acciones {
            margin-top: 16px;
        }

        details {
            margin-top: 20px;
        }

        summary {
            cursor: pointer;
            color: #3fa7d6;
            font-weight: bold;
        }

        @media (max-width: 600px) {
            body {
                padding: 12px;
            }

            section {
                padding: 14px;
            }

            input[type="number"] {
                width: 65px;
            }

            td {
                padding: 2px;
            }
        }
    </style>
</head>

<body>
<main>

    <h1>Resolucion de sistemas lineales</h1>

    <p>
        Rouché-Frobenius, eliminacion de Gauss-Jordan
        y solucion general.
    </p>

    <!-- CONFIGURACION DE LA MATRIZ -->

    <section>

        <h2>1. Define la matriz</h2>

        <form method="post">

            <div class="config">

                <label class="campo">
                    Ecuaciones

                    <input
                        type="number"
                        name="filas"
                        min="1"
                        max="8"
                        value="{{ filas }}"
                        required
                    >
                </label>

                <label class="campo">
                    Incognitas

                    <input
                        type="number"
                        name="columnas"
                        min="1"
                        max="8"
                        value="{{ columnas }}"
                        required
                    >
                </label>

                <label class="campo">
                    Metodo

                    <select name="metodo">
                        <option
                            value="reduccion"
                            {% if metodo == "reduccion" %}selected{% endif %}
                        >
                            Reduccion (Gauss-Jordan)
                        </option>

                        <option
                            value="igualacion"
                            {% if metodo == "igualacion" %}selected{% endif %}
                        >
                            Igualacion (2x2)
                        </option>
                    </select>
                </label>

                <button name="accion" value="generar">
                    Generar matriz
                </button>

            </div>

            <!-- MATRIZ DE COEFICIENTES -->

            {% if mostrar_matriz %}

                <h2>2. Introduce los coeficientes</h2>

                <p>
                    Introduce los coeficientes de cada incognita
                    y el termino independiente.
                </p>

                <!-- Mantener dimensiones y metodo al resolver -->

                <input
                    type="hidden"
                    name="filas"
                    value="{{ filas }}"
                >

                <input
                    type="hidden"
                    name="columnas"
                    value="{{ columnas }}"
                >

                <input
                    type="hidden"
                    name="metodo"
                    value="{{ metodo }}"
                >

                <table>
                    <tbody>

                    {% for i in range(filas) %}

                        <tr>

                            {% for j in range(columnas) %}

                                <td>
                                    <input
                                        required
                                        type="number"
                                        step="any"
                                        name="a_{{ i }}_{{ j }}"
                                        value="{{ valores_formulario.get('a_%d_%d' % (i, j), '') }}"
                                        placeholder="{{ nombres[j] }}"
                                        aria-label="Ecuacion {{ i+1 }}, coeficiente de {{ nombres[j] }}"
                                    >
                                </td>

                            {% endfor %}

                            <td>=</td>

                            <!-- Termino independiente -->

                            <td>
                                <input
                                    required
                                    type="number"
                                    step="any"
                                    name="b_{{ i }}"
                                    value="{{ valores_formulario.get('b_%d' % i, '') }}"
                                    placeholder="b{{ i+1 }}"
                                    aria-label="Termino independiente de la ecuacion {{ i+1 }}"
                                >
                            </td>

                        </tr>

                    {% endfor %}

                    </tbody>
                </table>

                <div class="acciones">
                    <button name="accion" value="resolver">
                        Resolver sistema
                    </button>
                </div>

            {% endif %}

        </form>

    </section>

    <!-- MENSAJES DE ERROR -->

    {% if error %}

        <section class="error">
            <strong>Error:</strong> {{ error }}
        </section>

    {% endif %}

    <!-- RESULTADOS -->

    {% if resultado %}

        <section class="resultado">

            <h2>Resultado</h2>

            <p>
                <strong>{{ resultado.tipo }}</strong>
            </p>

            <p>
                Rango de A:
                <strong>{{ resultado.rango_matriz }}</strong>

                |

                Rango de [A|b]:
                <strong>{{ resultado.rango_ampliada }}</strong>
            </p>

            <h3>Solucion general</h3>

            <p>{{ resultado.solucion_general }}</p>

            <h3>Forma reducida</h3>

            <pre>{{ formatear(resultado.reducida) }}</pre>

            <!-- PASOS DEL PROCEDIMIENTO -->

            <details>

                <summary>
                    Mostrar pasos del procedimiento
                </summary>

                {% for paso in resultado.pasos %}

                    <pre>{{ paso }}</pre>

                {% endfor %}

            </details>

        </section>

    {% endif %}

</main>
</body>
</html>"""


@app.route("/", methods=["GET", "POST"])
def inicio():
    filas, columnas, metodo = 2, 2, "reduccion"
    mostrar_matriz, resultado, error = False, None, None
    valores_formulario = {}
    if request.method == "POST":
        valores_formulario = request.form.to_dict(flat=True)
        try:
            filas, columnas = int(request.form["filas"]), int(request.form["columnas"])
            metodo = request.form.get("metodo", "reduccion")
            if metodo not in {"reduccion", "igualacion"}:
                raise ValueError("Metodo no valido.")
            if not 1 <= filas <= 8 or not 1 <= columnas <= 8:
                raise ValueError("Las dimensiones deben estar entre 1 y 8.")
            mostrar_matriz = True
            if request.form.get("accion") == "resolver":
                resultado = analizar_sistema(*convertir_formulario(request.form, filas, columnas), metodo=metodo)
        except (KeyError, ValueError) as exc:
            error = str(exc)
    return render_template_string(PLANTILLA, filas=filas, columnas=columnas, nombres=NOMBRES_VARIABLES,
                                  mostrar_matriz=mostrar_matriz, resultado=resultado, error=error,
                                  formatear=formatear_matriz, metodo=metodo, valores_formulario=valores_formulario)


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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    if args.cli:
        resolver_consola()
    else:
        app.run(host=args.host, port=args.port, debug=True)