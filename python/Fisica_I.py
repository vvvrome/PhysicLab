import numpy as np
from typing import Dict, Any, Callable, Tuple, List


def simulacion_proyectil(v0: float = 20.0, angulo_deg: float = 45.0, g: float = 9.81, tiempo_max: float = 5.0) -> Dict[str, Any]:
    """Simula un tiro parabólico sin resistencia al aire.

    Retorna un diccionario con tiempos, x, y, alcance, altura máxima y trayectoria.
    """
    angulo = np.radians(angulo_deg)
    t = np.linspace(0, tiempo_max, 400)
    x = v0 * np.cos(angulo) * t
    y = v0 * np.sin(angulo) * t - 0.5 * g * t ** 2
    alcance = (v0 ** 2 * np.sin(2 * angulo)) / g
    altura_maxima = (v0 ** 2 * np.sin(angulo) ** 2) / (2 * g)
    trayectoria = np.column_stack((x, y))
    return {
        "tiempo": t,
        "x": x,
        "y": y,
        "alcance": alcance,
        "altura_maxima": altura_maxima,
        "trayectoria": trayectoria,
    }


def caida_libre(h0: float, g: float = 9.81) -> Dict[str, Any]:
    """Devuelve tiempo de caída y velocidad al impacto para caída desde altura h0."""
    if h0 < 0:
        raise ValueError("La altura inicial debe ser no negativa.")
    t = np.sqrt(2 * h0 / g) if h0 > 0 else 0.0
    v = g * t
    return {"tiempo": t, "velocidad_final": v}


def fuerza_lorentz(carga: float, velocidad, campo_electrico, campo_magnetico):
    """Calcula la fuerza de Lorentz F = q (E + v x B)."""
    v = np.asarray(velocidad, dtype=float)
    E = np.asarray(campo_electrico, dtype=float)
    B = np.asarray(campo_magnetico, dtype=float)
    if v.shape != (3,) or E.shape != (3,) or B.shape != (3,):
        raise ValueError("La velocidad y los campos deben ser vectores 3D.")
    return carga * (E + np.cross(v, B))


def movimiento_mru(x0: float, v0: float, t: float) -> float:
    """Posición en MRU: x = x0 + v0 * t."""
    return x0 + v0 * t


def movimiento_mrua(x0: float, v0: float, a: float, t: float) -> Dict[str, float]:
    """Devuelve velocidad y posición en MRUA.

    Parameters
    - x0: posición inicial
    - v0: velocidad inicial
    - a: aceleración constante
    - t: tiempo

    Returns diccionario con 'velocidad' y 'posicion'.
    """
    v = float(v0 + a * t)
    x = float(x0 + v0 * t + 0.5 * a * t ** 2)
    return {"velocidad": v, "posicion": x}


def cinematica_nd(pos0: np.ndarray, vel0: np.ndarray, a: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Cinemática para sistemas N-d: devuelve (pos, vel) en tiempo t."""
    pos0 = np.asarray(pos0, dtype=float)
    vel0 = np.asarray(vel0, dtype=float)
    a = np.asarray(a, dtype=float)
    pos = pos0 + vel0 * t + 0.5 * a * t ** 2
    vel = vel0 + a * t
    return pos, vel


def transformar_frame(pos: np.ndarray, vel: np.ndarray, r_rel: np.ndarray, v_rel: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Transforma posición y velocidad a un marco con desplazamiento r_rel y velocidad v_rel.

    pos, vel: en marco A; retorna (pos_B, vel_B) en marco B donde B se mueve respecto de A.
    """
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    r_rel = np.asarray(r_rel, dtype=float)
    v_rel = np.asarray(v_rel, dtype=float)
    if pos.shape != r_rel.shape or vel.shape != v_rel.shape:
        raise ValueError("Las dimensiones de pos/vel y r_rel/v_rel deben coincidir.")
    pos_b = pos - (r_rel + v_rel * float(t))
    vel_b = vel - v_rel
    return pos_b, vel_b


def fuerza_resultante(fuerzas: np.ndarray) -> np.ndarray:
    """Suma de fuerzas aplicadas (vectorial)."""
    return np.sum(np.asarray(fuerzas, dtype=float), axis=0)


def aceleracion_por_fuerza( F: np.ndarray, m: float) -> np.ndarray:
    """a = F / m"""
    if m == 0:
        raise ValueError("La masa debe ser distinta de cero.")
    return np.asarray(F, dtype=float) / float(m)


def fuerza_rozamiento(mu_k: float, normal: float, vel_vec: np.ndarray) -> np.ndarray:
    """Calcula fuerza de rozamiento cinético: direccion opuesta a la velocidad.

    Si la velocidad es cero devuelve 0 (no modela rozamiento estático complejo aquí).
    """
    vel = np.asarray(vel_vec, dtype=float)
    vnorm = np.linalg.norm(vel)
    if vnorm <= 0:
        return np.zeros_like(vel)
    direction = -vel / vnorm
    return direction * (mu_k * abs(normal))


def aceleracion_plano_inclinado(theta_rad: float, mu_k: float = 0.0, g: float = 9.81) -> float:
    """Aceleración a lo largo del plano inclinado (positivo hacia abajo)."""
    return g * np.sin(theta_rad) - mu_k * g * np.cos(theta_rad)


def trabajo(fuerza: np.ndarray, desplazamiento: np.ndarray) -> float:
    f = np.asarray(fuerza, dtype=float)
    d = np.asarray(desplazamiento, dtype=float)
    if f.shape != d.shape:
        raise ValueError("fuerza y desplazamiento deben tener la misma forma.")
    return float(np.dot(f, d))


def energia_cinetica(m: float, v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    return 0.5 * float(m) * float(np.dot(v, v))


def energia_potencial_gravitatoria(m: float, h: float, g: float = 9.81) -> float:
    return m * g * h


def impulso(m: float, v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    return m * (np.asarray(v1, dtype=float) - np.asarray(v0, dtype=float))


def colision_unidimensional(m1: float, v1: float, m2: float, v2: float, elastic: bool = True) -> Tuple[float, float]:
    """Resuelve colisión en 1D entre dos partículas; devuelve (v1', v2')."""
    if m1 + m2 == 0:
        raise ValueError("Masas invalidas para colision.")
    if elastic:
        v1p = (v1 * (m1 - m2) + 2 * m2 * v2) / (m1 + m2)
        v2p = (v2 * (m2 - m1) + 2 * m1 * v1) / (m1 + m2)
        return float(v1p), float(v2p)
    else:
        # colisión perfectamente inelástica: se unen
        v = (m1 * v1 + m2 * v2) / (m1 + m2)
        return v, v


def fuerza_centripeta(m: float, v: float, r: float) -> float:
    return m * v * v / r


def momento_inercia_roda(m: float, r: float, tipo: str = "cilindro_central") -> float:
    """Momento de inercia para formas comunes.

    tipos soportados: 'cilindro_central', 'disco', 'esfera_solida', 'barra_centro' (longitud r es usado como longitud)
    """
    if tipo == "cilindro_central" or tipo == "disco":
        return 0.5 * float(m) * float(r) * float(r)
    if tipo == "esfera_solida":
        return 2.0 / 5.0 * m * r * r
    if tipo == "barra_centro":
        L = r
        return 1.0 / 12.0 * m * L * L
    raise ValueError("Tipo no soportado para momento de inercia.")


def torque(I: float, alpha: float) -> float:
    return I * alpha


def impulso_rotacional(torque_func: Callable[[float], float], t0: float, tf: float) -> float:
    if tf <= t0:
        return 0.0
    ts = np.linspace(t0, tf, 200)
    vals = np.array([float(torque_func(float(ti))) for ti in ts], dtype=float)
    return float(np.trapz(vals, ts))


def potencial_gradiente(pot_func: Callable[[np.ndarray], float], q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Calcula gradiente numérico de un potencial V(q) y devuelve -grad(V) (fuerza conservativa)."""
    q = np.asarray(q, dtype=float)
    grad = np.zeros_like(q, dtype=float)
    for i in range(q.size):
        dq = np.zeros_like(q)
        dq[i] = eps
        grad[i] = (pot_func(q + dq) - pot_func(q - dq)) / (2 * eps)
    return -grad


def euler(f: Callable[[float, np.ndarray], np.ndarray], y0: np.ndarray, t: np.ndarray) -> np.ndarray:
    ys = np.zeros((t.size, y0.size), dtype=float)
    ys[0] = y0
    for i in range(1, t.size):
        dt = t[i] - t[i - 1]
        ys[i] = ys[i - 1] + dt * f(t[i - 1], ys[i - 1])
    return ys


def euler_cromer_second_order(f_acc: Callable[[float, np.ndarray], np.ndarray], q0: np.ndarray, v0: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Integrador Euler-Cromer para sistemas de segundo orden.

    f_acc(t, q) -> aceleracion (array)
    q0, v0: vectores iniciales
    t: array de tiempos
    Returns (q_array, v_array)
    """
    q0 = np.asarray(q0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    q = np.zeros((t.size, q0.size), dtype=float)
    v = np.zeros((t.size, v0.size), dtype=float)
    q[0] = q0
    v[0] = v0
    for i in range(1, t.size):
        dt = t[i] - t[i - 1]
        a = np.asarray(f_acc(t[i - 1], q[i - 1]), dtype=float)
        v[i] = v[i - 1] + a * dt
        q[i] = q[i - 1] + v[i] * dt
    return q, v


def rk4(f: Callable[[float, np.ndarray], np.ndarray], y0: np.ndarray, t: np.ndarray) -> np.ndarray:
    ys = np.zeros((t.size, y0.size), dtype=float)
    ys[0] = y0
    for i in range(1, t.size):
        dt = t[i] - t[i - 1]
        ti = t[i - 1]
        yi = ys[i - 1]
        k1 = f(ti, yi)
        k2 = f(ti + dt / 2.0, yi + dt * k1 / 2.0)
        k3 = f(ti + dt / 2.0, yi + dt * k2 / 2.0)
        k4 = f(ti + dt, yi + dt * k3)
        ys[i] = yi + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return ys


def verlet(f_acc: Callable[[float, np.ndarray], np.ndarray], q0: np.ndarray, v0: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Verlet (velocity Verlet) integrator for second-order ODEs.

    f_acc(t, q) -> acceleration vector
    Returns (q_array, v_array)
    """
    q0 = np.asarray(q0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    q = np.zeros((t.size, q0.size), dtype=float)
    v = np.zeros((t.size, v0.size), dtype=float)
    q[0] = q0
    v[0] = v0
    a = np.asarray(f_acc(t[0], q0), dtype=float)
    for i in range(1, t.size):
        dt = t[i] - t[i - 1]
        q[i] = q[i - 1] + v[i - 1] * dt + 0.5 * a * dt * dt
        a_new = np.asarray(f_acc(t[i], q[i]), dtype=float)
        v[i] = v[i - 1] + 0.5 * (a + a_new) * dt
        a = a_new
    return q, v


def leapfrog(f_acc: Callable[[float, np.ndarray], np.ndarray], q0: np.ndarray, v0: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Leapfrog integrator.

    Uses staggered velocities v_{n+1/2}. Returns (q_array, v_array) with velocities at integer times.
    """
    q0 = np.asarray(q0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    q = np.zeros((t.size, q0.size), dtype=float)
    v = np.zeros((t.size, v0.size), dtype=float)
    q[0] = q0
    if t.size < 2:
        return q, v
    dt0 = t[1] - t[0]
    a0 = np.asarray(f_acc(t[0], q0), dtype=float)
    v_half = v0 + 0.5 * dt0 * a0
    for i in range(1, t.size):
        dt = t[i] - t[i - 1]
        q[i] = q[i - 1] + v_half * dt
        a = np.asarray(f_acc(t[i], q[i]), dtype=float)
        v_half = v_half + a * dt
        v[i] = v_half - 0.5 * a * dt
    return q, v


def oscilador_armonico(m: float, k: float, q0: float, v0: float, t: np.ndarray, method: str = "rk4") -> Tuple[np.ndarray, np.ndarray]:
    def f(ti, y):
        q, v = y[0], y[1]
        return np.array([v, -k / m * q], dtype=float)

    y0 = np.array([q0, v0], dtype=float)
    if method == "rk4":
        ys = rk4(f, y0, t)
    else:
        ys = euler(f, y0, t)
    return ys[:, 0], ys[:, 1]


def sistema_acoplado_dos_masas(m1: float, m2: float, k1: float, k2: float, kc: float, q0: np.ndarray, v0: np.ndarray, t: np.ndarray, method: str = "rk4") -> Tuple[np.ndarray, np.ndarray]:
    """Simula dos masas acopladas en linea (dos grados de libertad).

    Ecuaciones: m1 q1'' = -k1 q1 - kc (q1 - q2)
                  m2 q2'' = -k2 q2 - kc (q2 - q1)
    """
    def f(ti, y):
        q1, q2, v1, v2 = y
        a1 = (-k1 * q1 - kc * (q1 - q2)) / m1
        a2 = (-k2 * q2 - kc * (q2 - q1)) / m2
        return np.array([v1, v2, a1, a2], dtype=float)

    y0 = np.array([q0[0], q0[1], v0[0], v0[1]], dtype=float)
    if method == "rk4":
        ys = rk4(f, y0, t)
    else:
        ys = euler(f, y0, t)
    q = ys[:, :2]
    v = ys[:, 2:4]
    return q, v


def doble_pendulo(theta0: Tuple[float, float], omega0: Tuple[float, float], L1: float, L2: float, m1: float, m2: float, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simula el doble péndulo (ecuaciones no lineales) usando RK4."""
    def f(ti, y):
        th1, th2, w1, w2 = y
        d = m1 + m2 * np.sin(th1 - th2) ** 2
        g = 9.81
        a1 = (
            m2 * g * np.sin(th2) * np.cos(th1 - th2)
            - m2 * np.sin(th1 - th2) * (L1 * w1 ** 2 * np.cos(th1 - th2) + L2 * w2 ** 2)
            - (m1 + m2) * g * np.sin(th1)
        ) / (L1 * d)
        a2 = (
            (m1 + m2) * (L1 * w1 ** 2 * np.sin(th1 - th2) - g * np.sin(th2) + g * np.sin(th1) * np.cos(th1 - th2))
            + m2 * L2 * w2 ** 2 * np.sin(th1 - th2) * np.cos(th1 - th2)
        ) / (L2 * d)
        return np.array([w1, w2, a1, a2], dtype=float)

    y0 = np.array([theta0[0], theta0[1], omega0[0], omega0[1]], dtype=float)
    ys = rk4(f, y0, t)
    thetas = ys[:, :2]
    omegas = ys[:, 2:4]
    return thetas, omegas


if __name__ == "__main__":
    print("Módulo de física: contiene funciones de cinemática, dinámica y métodos numéricos.")

# Public API
__all__ = [
    "simulacion_proyectil",
    "caida_libre",
    "fuerza_lorentz",
    "movimiento_mru",
    "movimiento_mrua",
    "cinematica_nd",
    "transformar_frame",
    "fuerza_resultante",
    "aceleracion_por_fuerza",
    "fuerza_rozamiento",
    "aceleracion_plano_inclinado",
    "trabajo",
    "energia_cinetica",
    "energia_potencial_gravitatoria",
    "impulso",
    "colision_unidimensional",
    "fuerza_centripeta",
    "momento_inercia_roda",
    "torque",
    "impulso_rotacional",
    "potencial_gradiente",
    "euler",
    "euler_cromer_second_order",
    "rk4",
    "verlet",
    "leapfrog",
    "oscilador_armonico",
    "sistema_acoplado_dos_masas",
    "doble_pendulo",
]
