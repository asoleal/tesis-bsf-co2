#!/usr/bin/env python3
"""Figuras del articulo. No calibra parametros ni sustituye las simulaciones canonicas."""
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modelo import (ParametrosEriksen, ParametrosLecho, estado_inicial_desde_dw,
                    rhs, tasas_larva, tasas_lecho, carbono_a_ppm_min)

FIG = ROOT / "figuras"
FIG.mkdir(exist_ok=True)
T_EVAL = np.arange(0.0, 14.0001, 0.05)
PH0 = {"D1": 4.9, "D4": 3.3}
MS = {"D1": 0.863, "D4": 0.803}
MICRO = {"D1": (0.6086237218572979, 0.95),
         "D4": (1.0, 0.2483341741138431)}
OBS = {
    "D1": [(6, 670.558, True, 1), (6, 474.229, True, 2),
           (8, 311.822, True, 1), (10, 285.572, False, 1)],
    "D4": [(6, 228.516, False, 1), (8, 383.674, True, 1),
           (10, 104.114, False, 1), (14, 79.107, False, 1)]}


def original(dieta):
    p = ParametrosEriksen(tp_min=8.5)
    y0 = estado_inicial_desde_dw(PH0[dieta] * 0.325, 0, 0, p)[:2]

    def fun(t, y):
        r = tasas_larva(t, max(y[0], 0), max(y[1], 0), p, 1.0)
        return [r.dB, r.dLtot]

    sol = solve_ivp(fun, (0, 14), y0, method="LSODA", t_eval=T_EVAL,
                    rtol=1e-7, atol=1e-10)
    assert sol.success
    ppm = []
    for t, y in zip(sol.t, sol.y.T):
        r = tasas_larva(t, max(y[0], 0), max(y[1], 0), p, 1.0)
        ppm.append(carbono_a_ppm_min(700 * r.rCO2_C / 1000, 13.08, 300))
    return np.asarray(ppm)


def propuesto(dieta):
    kref, dcs = MICRO[dieta]
    pl = ParametrosEriksen(amax=1.4995, rho=4.395, tp_min=8.5)
    pm = ParametrosLecho(f_dieta=1.0 if dieta == "D1" else 0.8047,
                         K_D=2.0, k_ref=kref, K_SM=50.0,
                         Y_X=0.4, Y_Cmic=0.4)
    y = estado_inicial_desde_dw(PH0[dieta] * 0.325, 0, 1, pl)
    ts, ys = [], []
    for a, b in [(0, 5), (5, 14)]:
        y[2] += 250 * MS[dieta] * 0.47 * dcs
        te = T_EVAL[(T_EVAL >= a) & (T_EVAL <= b)]
        if a > 0:
            te = te[1:]
        sol = solve_ivp(rhs, (a, b), y, args=(pl, pm, 700, 26.85),
                        method="LSODA", t_eval=te,
                        rtol=1e-6, atol=1e-9)
        assert sol.success
        ts.append(sol.t)
        ys.append(sol.y)
        y = sol.y[:, -1]
    t = np.concatenate(ts)
    Y = np.concatenate(ys, axis=1)
    lar, mic = [], []
    for tk, yk in zip(t, Y.T):
        B, L, D, X, _, _ = np.maximum(yk, 0)
        mm = tasas_lecho(D, X, 26.85, pm)
        ll = tasas_larva(tk, B, L, pl, pm.f_dieta * mm.phi_D)
        lar.append(carbono_a_ppm_min(700 * ll.rCO2_C / 1000, 13.08, 300))
        mic.append(carbono_a_ppm_min(mm.rCmic, 13.08, 300))
    return np.asarray(lar), np.asarray(mic)


def residuos_censurados(modelo):
    res = []
    for dieta in ("D1", "D4"):
        for dia, obs, sat, _ in OBS[dieta]:
            r = float(np.interp(dia, T_EVAL, modelo[dieta])) - obs
            res.append(min(r, 0.0) if sat else r)
    return np.asarray(res)


E0 = {d: original(d) for d in ("D1", "D4")}
COMP = {d: propuesto(d) for d in ("D1", "D4")}
E1 = {d: COMP[d][0] + COMP[d][1] for d in ("D1", "D4")}
r0 = residuos_censurados(E0)
r1 = residuos_censurados(E1)
rmse0 = float(np.sqrt(np.mean(r0 ** 2)))
rmse1 = float(np.sqrt(np.mean(r1 ** 2)))
assert abs(rmse0 - 183.6127) < 0.01, rmse0
assert abs(rmse1 - 85.1604) < 0.01, rmse1
print(f"RMSE propio: Eriksen={rmse0:.2f}; propuesto={rmse1:.2f} ppm/min")

# Figura 1: datos propios y fuentes de CO2.
fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
for j, dieta in enumerate(("D1", "D4")):
    a = ax[0, j]
    a.plot(T_EVAL, E0[dieta], "--", color="black",
           label="Eriksen: solo larvas")
    a.plot(T_EVAL, E1[dieta], color="tab:red",
           label="Propuesto: larvas + lecho")
    for dia, obs, sat, rep in OBS[dieta]:
        a.plot(dia, obs, "^" if sat else "o", color="tab:blue",
               markerfacecolor="none" if sat else "tab:blue", ms=6)
    a.set_title(f"{dieta}: CO2 total")
    a.set_ylabel("ppm/min")
    a.grid(alpha=0.2)

    b = ax[1, j]
    b.plot(T_EVAL, COMP[dieta][0], label="Componente larvario",
           color="tab:blue")
    b.plot(T_EVAL, COMP[dieta][1], label="Componente microbiano",
           color="tab:green")
    b.plot(T_EVAL, E1[dieta], label="Total", color="tab:red", lw=1)
    b.set_title(f"{dieta}: particion predicha")
    b.set_xlabel("Dia desde el montaje")
    b.set_ylabel("ppm/min")
    b.grid(alpha=0.2)

ax[0, 0].legend(fontsize=8)
ax[1, 0].legend(fontsize=8)
fig.suptitle("Triangulos vacios: cotas inferiores por saturacion")
fig.tight_layout()
fig.savefig(FIG / "datos_propios_modelos.pdf")
fig.savefig(FIG / "datos_propios_modelos.png", dpi=220)
plt.close(fig)

# Figura 2: transferencia externa. Valores canonicos ya calculados.
fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
dias = np.array([2, 7, 11])
ax[0, 0].plot(dias, [2.53, 8.04, 5.58], "ko-", label="Observado")
ax[0, 0].plot(dias, [0.600, 5.877, 6.847], "--s", label="Eriksen")
ax[0, 0].plot(dias, [4.688, 9.040, 4.979], "--^", label="Propuesto")
ax[0, 0].set_title("Ermolaev: flujo total")
ax[0, 0].set_xlabel("Dia")
ax[0, 0].set_ylabel("mg CO2/min")
ax[0, 0].legend(fontsize=8)

T = np.array([27.0, 35.3, 39.4, 41.9])
ax[0, 1].fill_between(T, 26, 39, color="gray", alpha=0.25,
                      label="Eriksen: rango")
ax[0, 1].plot(T, [50, 54.5, 53, 24], "ko-", label="Observado")
ax[0, 1].plot(T, [81.57, 90.32, 98.19, 104.79], "--^",
              color="tab:red", label="Propuesto")
ax[0, 1].set_title("Schon: efecto de temperatura")
ax[0, 1].set_xlabel("Temperatura (C)")
ax[0, 1].set_ylabel("L CO2/kg")
ax[0, 1].legend(fontsize=8)

x = np.arange(3)
w = 0.25
ax[1, 0].bar(x-w, [5.55, 32.38, 37.30], w, label="Observado")
ax[1, 0].bar(x, [6.455]*3, w, label="Eriksen")
ax[1, 0].bar(x+w, [20.0178]*3, w, label="Propuesto")
ax[1, 0].set_xticks(x, ["RUN-1", "RUN-2", "RUN-3"])
ax[1, 0].set_title("Rossi: CO2 neto acumulado")
ax[1, 0].set_ylabel("g CO2")
ax[1, 0].legend(fontsize=8)

h = np.array([45, 55, 65, 75])
ax[1, 1].plot(h, [78.08, 62.60, 43.52, 28.38], "ko-",
              label="Observado")
ax[1, 1].plot(h, [67.34, 67.32, 67.25, 66.11], "--^",
              color="tab:red", label="Propuesto")
ax[1, 1].set_title("Bekker: fraccion microbiana")
ax[1, 1].set_xlabel("Humedad del sustrato (%)")
ax[1, 1].set_ylabel("CO2 microbiano (%)")
ax[1, 1].legend(fontsize=8)

for a in ax.flat:
    a.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(FIG / "comparacion_externa.pdf")
fig.savefig(FIG / "comparacion_externa.png", dpi=220)
plt.close(fig)

# Figura 3: detalle de Bekker.
fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
ax[0, 0].plot(h, [28.15697, 20.35742, 13.34650, 6.98028],
              "ko-", label="Observado")
ax[0, 0].plot(h, [20.77, 21.35, 19.81, 13.55], "--^",
              label="Propuesto")
ax[0, 0].set_title("CO2 total acumulado")
ax[0, 0].set_ylabel("g C")
ax[0, 0].legend(fontsize=8)

ax[0, 1].plot(h, [78.08, 62.60, 43.52, 28.38], "ko-",
              label="Observado")
ax[0, 1].plot(h, [67.34, 67.32, 67.25, 66.11], "--^",
              label="Propuesto")
ax[0, 1].set_title("Fraccion microbiana")
ax[0, 1].set_ylabel("%")
ax[0, 1].legend(fontsize=8)

ax[1, 0].plot(h, [3.9646, 2.2916, 4.5802, 3.1692], "--s",
              label="Eriksen")
ax[1, 0].plot(h, [1.19, 1.31, 1.67, 1.76], "--^",
              label="Propuesto")
ax[1, 0].set_title("RMSE del CO2 individual")
ax[1, 0].set_ylabel("mg CO2/d/larva")
ax[1, 0].legend(fontsize=8)

ax[1, 1].plot(h, [2.56, 3.56, 6.47, 7.50], "--^",
              color="tab:red", label="Propuesto")
ax[1, 1].set_title("Anticipacion del pico individual")
ax[1, 1].set_ylabel("dias")
ax[1, 1].legend(fontsize=8)

for a in ax.flat:
    a.set_xlabel("Humedad del sustrato (%)")
    a.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(FIG / "bekker_comparacion.pdf")
fig.savefig(FIG / "bekker_comparacion.png", dpi=220)
plt.close(fig)
print("Figuras escritas en", FIG)
