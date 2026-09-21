# PhysicsLab

Herramientas matematicas y simulaciones fisicas desarrolladas con Python.

## Modulos
- Algebra Lineal
- Analisis Matematico
- Fisica
- Simulacion y visualizacion

## Tecnologias
Python, NumPy, SymPy, SciPy, Matplotlib y Flask.

## Instalacion
1. Clona el repositorio.
2. Entra en la carpeta del proyecto.
3. Crea un entorno virtual si lo deseas:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecucion de la aplicacion web
Desde la raiz del proyecto ejecuta:
```bash
python app.py
```
Luego abre en el navegador:
```text
http://127.0.0.1:5000
```

## Ejecutar pruebas automatizadas
Desde la raiz del proyecto:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Estructura principal
- `app.py`: aplicacion web con la mini interfaz
- `python/Algebra.py`: funciones matematicas y fisicas
- `tests/`: pruebas automatizadas

## Notas
- El proyecto valida dimensiones y entradas en el servidor.
- La evaluacion de expresiones matematicas usa SymPy con un conjunto seguro de funciones permitidas.
- La parte de sistemas de ecuaciones incluye resolucion por reduccion e igualacion.
