#!/usr/bin/env python3
"""Replica los 15 casos de la Tabla 2 de Eriksen (2022), sin calibrar."""
from pathlib import Path
import csv
import sys
import numpy as np
from scipy.integrate import solve_ivp

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from modelo import ParametrosEriksen, estado_inicial_desde_dw, tasas_larva

# nombre, XDW0, edad0, Bmax, amax, YB, m, alpha, beta,
# XDWmax_ref, tp_ref, lipidos_ref, deltaDW_ref
CASOS = [
    ("Fig3A", .03, 1, 65, 1.4, .44, .08, .92, 2.02, 68, 16, 27, 31),
    ("Fig3B", 3, 5, 65, 1.4, .44, .08, 1.02, 1.01, 82, 13, 43, 37),
    ("Fig4_1", 2.46, 5.8, 90, 1.21, .44, .08, 1, 1, 105, 20, 41, 30),
    ("Fig4_2", 2.46, 5.8, 90, 1.06, .44, .08, 1, 1, 97, 21, 39, 29),
    ("Fig4_3", 2.46, 5.8, 90, .60, .44, .08, 1, 1, 68, 27, 32, 27),
    ("Fig4_4", 2.46, 5.8, 90, .34, .44, .08, 1, 1, 39, 34, 24, 25),
    ("Fig4_5", 2.46, 5.8, 90, .25, .44, .08, 1, 1, 27, 37, 20, 24),
    ("Fig5A", .6, 5, 90, 1.2, .44, .08, .81, 1.17, 91, 21, 35, 34),
    ("Fig5B", .6, 5, 90, 1.2, .59, .09, .48, .94, 74, 25, 35, 34),
    ("Fig5C", .6, 5, 90, 1.3, .63, .15, .26, .90, 45, 26, 26, 31),
    ("Fig5D", .6, 5, 90, 1.4, .76, .23, .28, .88, 39, 25, 25, 31),
    ("Fig6A", .6, 5, 90, 1.1, .44, .08, 1.19, 3.19, 82, 18, 18, 29),
    ("Fig6B", .6, 5, 90, 1.1, .44, .08, .92, 1.66, 84, 20, 27, 31),
    ("Fig6C", .6, 5, 90, .9, .44, .08, 3.27, 1.38, 108, 20, 39, 35),
    ("Fig6D", .6, 5, 90, .8, .44, .08, 2.59, 1.13, 101, 22, 41, 36),
]

# Tolerancias operativas declaradas antes de ejecutar: no provienen de Eriksen.
TOL_REL = .10       # XDWmax y tp
TOL_LIP = 5.0       # puntos porcentuales
TOL_DDW = 3.0       # puntos porcentuales


def rhs(t, y, p):
    r = tasas_larva(t, y[0], y[1], p)
    return [r.dB, r.dLtot]


def cruce(t, y, p):
    bmax = max(p.Bmax0 - p.rho * max(t - p.tp_min, 0.0), 1e-12)
    return bmax - y[0]


cruce.terminal = True
cruce.direction = -1


def main():
    filas = []
    for c in CASOS:
        (nombre, xdw0, edad0, bmax, amax, yb, mant, alpha, beta,
         xdw_ref, tp_ref, lip_ref, ddw_ref) = c

        # delta_I=5: Tabla 1 y texto de Eriksen; Tabla 2 imprime su recíproco, 0.2.
        p = ParametrosEriksen(
            amax=amax, alpha=alpha, beta=beta, YB=yb, m=mant,
            Bmax0=bmax * .49 * .9, tp_min=13.0, rho=1.0, delta_I=5.0,
            f_mant_prepupa=1.0, k_cat_prepupa=0.0,
        )
        y0 = estado_inicial_desde_dw(xdw0, 0.0, 0.0, p)[:2]
        sol = solve_ivp(
            rhs, (edad0, 60.0), y0, args=(p,), method="LSODA",
            t_eval=np.linspace(edad0, 60.0, 3001), events=cruce,
            max_step=.02, rtol=1e-8, atol=1e-10,
        )
        if not sol.success:
            raise RuntimeError(f"{nombre}: {sol.message}")

        B, Ltot = np.maximum(sol.y, 0.0)
        xdw_b = (1 - p.delta_LB) * B / (p.delta_CB * p.delta_O)
        ldw = Ltot / p.delta_CL
        xdw = xdw_b + ldw
        xww = xdw_b / .25 + ldw

        xdw_mod = float(np.max(xdw))
        lip_mod = float(np.max(100 * ldw / xdw))
        ddw_mod = float(np.max(100 * xdw / xww))
        # Definicion del Excel oficial: edad en XDWmax.
        tp_mod = float(sol.t[int(np.argmax(xdw))])

        fila = {
            "caso": nombre,
            "xdw_ref": xdw_ref, "xdw_mod": xdw_mod,
            "tp_ref": tp_ref, "tp_mod": tp_mod,
            "lip_ref": lip_ref, "lip_mod": lip_mod,
            "ddw_ref": ddw_ref, "ddw_mod": ddw_mod,
            "ok_xdw": int(abs(xdw_mod - xdw_ref) <= TOL_REL * xdw_ref),
            "ok_tp": int(np.isfinite(tp_mod)
                         and abs(tp_mod - tp_ref) <= TOL_REL * tp_ref),
            "ok_lip": int(abs(lip_mod - lip_ref) <= TOL_LIP),
            "ok_ddw": int(abs(ddw_mod - ddw_ref) <= TOL_DDW),
        }
        filas.append(fila)
        print(f"{nombre:7s} XDW {xdw_mod:6.1f}/{xdw_ref:3d}  "
              f"tp {tp_mod:5.1f}/{tp_ref:2d}  "
              f"lip {lip_mod:5.1f}/{lip_ref:2d}  "
              f"ddW {ddw_mod:5.1f}/{ddw_ref:2d}")

    salida = RAIZ / "resultados" / "validacion_eriksen.csv"
    salida.parent.mkdir(exist_ok=True)
    with salida.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=filas[0].keys())
        w.writeheader()
        w.writerows(filas)

    print("Valores: modelo/Eriksen")
    for k in ("xdw", "tp", "lip", "ddw"):
        print(f"{k}: {sum(f['ok_' + k] for f in filas)}/{len(filas)}")
    print(f"CSV: {salida}")


if __name__ == "__main__":
    main()
