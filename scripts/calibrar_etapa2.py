#!/usr/bin/env python3
"""Etapa 2 de calibracion: nucleo larvario contra crecimiento y tratamientos.

Calibrados (3 grados de libertad, multinicio, region de confianza):
  a_max  en [1.0, 1.5] d^-1   (escala log; rango de Eriksen para dietas)
  rho    en [0.5, 15] mg C/d  (escala log; cierre de ciclo mas rapido)
  f_D4   en [0.4, 1.0]        (calidad de D4; D1 queda fija en 1.0)

Congelados:
  - Nucleo de Eriksen: tp_min = 8.5 d (13 d desde eclosion - 4.5 d de edad
    al montaje), Bmax0, m, YB, YL, alpha, beta, delta_I y conversiones.
  - Modulo microbiano: k_ref y delta_CS por dieta, leidos de
    resultados/calibracion_etapa1.csv (etapa 1).
  - f_MS = 0.325 (conversion a PH, decision A con banda declarada).

Datos y ponderacion:
  - Crecimiento: peso = max(DE, 10 % del valor). SOLO entran al ajuste
    los dias <= 8 (fase de crecimiento/pico); los dias 10 y 14 se
    reportan como diagnostico de fase tardia (opcion A, ver abajo).
  - Gas de tratamientos: 8 cierres, peso = 25 ppm/min declarado; los
    saturados son cotas inferiores: solo penalizan si el modelo queda
    por debajo de lo observado. Los cierres tardios (dia > 8) se marcan
    como fase post-pico en la salida.
  - El ayuno NO entra al ajuste: se reporta al final como validacion.

Decisiones documentadas:
  - Primer pulso en t = 0: las larvas llegaron de la crianza con
    contenido intestinal; la alimentacion efectiva arranca con el
    montaje (no en el dia larvario 1).
  - Opcion A (fase tardia): el nucleo de Eriksen no cataboliza B ni
    lipidos y mantiene el mantenimiento pleno en prepupa, mientras el
    descenso de PH observado post-pico es mayormente agua. La fase
    tardia se excluye del costo y se reporta como limitacion
    estructural documentada del nucleo.

Las fronteras alcanzadas se reportan, no se corrigen.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from modelo import (
    ParametrosEriksen,
    ParametrosLecho,
    carbono_a_ppm_min,
    estado_inicial_desde_dw,
    rhs,
    tasas_larva,
    tasas_lecho,
)

# Condiciones del montaje
# Primer pulso en t = 0: las larvas llegaron con contenido intestinal
# de la crianza (decision documentada); segundo pulso dia larvario 5.
PULSOS = [(0.0, 250.0), (5.0, 250.0)]
MS_DIETA = {"D1": 0.863, "D4": 0.803}
N_LARVAS = 700
T_K = 300.0
T_C = T_K - 273.15
V_AIRE_L = 13.08
DELTA_C_ALIM = 0.47
F_MS = 0.325
T_FIN = 14.0
DIA_MONTAJE_SEP = 3
TP_MIN = 8.5
# Ventana de ajuste (opcion A): solo fase de crecimiento/pico.
# Dias > DIA_AJUSTE_MAX se reportan como diagnostico de fase tardia.
DIA_AJUSTE_MAX = 8.0

# Fijos del lecho (mismos que etapa 1)
X0_G_C = 1.0
Y_X = 0.4
Y_CMIC = 0.4
K_SM = 50.0
K_D = 2.0

# Ponderacion declarada
PESO_GAS = 25.0  # ppm/min

# Cotas fisicamente plausibles
LO = np.array([np.log10(1.0), np.log10(0.5), 0.4])
HI = np.array([np.log10(1.5), np.log10(15.0), 1.0])
ARRANQUES = [
    (np.log10(1.05), np.log10(1.0), 0.8),
    (np.log10(1.4), np.log10(1.0), 0.8),
    (np.log10(1.05), np.log10(8.0), 0.6),
    (np.log10(1.4), np.log10(8.0), 0.95),
]

AYUNO_OBS = 28.51  # ppm/min, dia larvario 14 (validacion, no entra al ajuste)
AYUNO_DIA = 14.0


def cargar_microbiano(ruta):
    df = pd.read_csv(ruta)
    falta = {"k_ref", "delta_CS", "dieta"} - set(df.columns)
    if falta:
        raise AssertionError(f"calibracion_etapa1.csv sin columnas {falta}")
    salida = {}
    for dieta, g in df.groupby("dieta"):
        salida[dieta] = (float(g["k_ref"].iloc[0]), float(g["delta_CS"].iloc[0]))
    if set(salida) != {"D1", "D4"}:
        raise AssertionError(f"dietas inesperadas en etapa 1: {sorted(salida)}")
    return salida


def simular_dieta(dieta, amax, rho, f_d4, micro, t_eval):
    """Sistema completo N=700 con pulsos; devuelve t, ppm y PH."""
    k_ref, dcs = micro[dieta]
    p_larva = ParametrosEriksen(amax=amax, rho=rho, tp_min=TP_MIN)
    p_lecho = ParametrosLecho(
        f_dieta=(1.0 if dieta == "D1" else f_d4),
        K_D=K_D, k_ref=k_ref, K_SM=K_SM, Y_X=Y_X, Y_Cmic=Y_CMIC,
    )
    y = estado_inicial_desde_dw(PH0[dieta] * F_MS, 0.0, X0_G_C, p_larva)
    cortes = sorted(set([0.0] + [d for d, _ in PULSOS] + [T_FIN]))
    pulsos = dict(PULSOS)
    ts, ys = [], []
    for a, b in zip(cortes[:-1], cortes[1:]):
        if a in pulsos:
            y[2] += pulsos[a] * MS_DIETA[dieta] * DELTA_C_ALIM * dcs
        t_seg = t_eval[(t_eval >= a) & (t_eval <= b)]
        if len(t_seg) < 2:
            t_seg = np.linspace(a, b, 3)
        sol = solve_ivp(
            rhs, (a, b), y, args=(p_larva, p_lecho, N_LARVAS, T_C),
            method="LSODA", t_eval=t_seg, rtol=1e-6, atol=1e-9,
        )
        if not sol.success:
            raise RuntimeError(f"fallo la integracion en [{a}, {b}]: {sol.message}")
        ts.append(sol.t)
        ys.append(sol.y)
        y = sol.y[:, -1]
    t = np.concatenate(ts)
    Y = np.concatenate(ys, axis=1)

    R_C = np.empty_like(t)
    for k, (tk, yk) in enumerate(zip(t, Y.T)):
        B, Ltot, D_C, X, _, _ = [max(float(v), 0.0) for v in yk]
        mic = tasas_lecho(D_C, X, T_C, p_lecho)
        lar = tasas_larva(tk, B, Ltot, p_larva, p_lecho.f_dieta * mic.phi_D)
        R_C[k] = N_LARVAS * lar.rCO2_C / 1000.0 + mic.rCmic
    ppm = carbono_a_ppm_min(R_C, V_AIRE_L, T_K)
    B = np.maximum(Y[0], 0.0)
    Ltot = np.maximum(Y[1], 0.0)
    ph = (B / (p_larva.delta_CB * p_larva.delta_O) + Ltot / p_larva.delta_CL) / F_MS
    return t, ppm, ph, Y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cierres", default="datos/procesados/cierres_etiquetados.csv")
    parser.add_argument("--crecimiento", default="datos/procesados/crecimiento_canonico.csv")
    parser.add_argument("--etapa1", default="resultados/calibracion_etapa1.csv")
    parser.add_argument("--salida", default="resultados/calibracion_etapa2.csv")
    parser.add_argument("--figura", default="figuras/calibracion_etapa2.png")
    args = parser.parse_args()

    micro = cargar_microbiano(args.etapa1)
    print(f"Microbiano congelado de etapa 1: {micro}")

    crec = pd.read_csv(args.crecimiento)
    cierres = pd.read_csv(args.cierres)
    gas = cierres[
        (~cierres["excluir_promedio"])
        & (cierres["Gas"] == "co2")
        & (cierres["condicion"] == "tratamiento")
    ].copy()
    if gas.empty:
        raise AssertionError("no hay cierres de tratamiento validos")
    gas["dia_larvario"] = gas["Dia_Experimento"] - DIA_MONTAJE_SEP

    global PH0
    PH0 = {
        d: float(g.loc[g["dia_larvario"] == 0, "ph_mg"].iloc[0])
        for d, g in crec.groupby("dieta")
    }

    t_eval = np.arange(0.0, T_FIN + 0.1, 0.1)
    obs_crec = [
        (d, float(r["dia_larvario"]), float(r["ph_mg"]), max(float(r["de_mg"]), 0.1 * float(r["ph_mg"])))
        for d, g in crec.groupby("dieta") for _, r in g.iterrows()
    ]
    obs_gas = [
        (d, float(r["dia_larvario"]), float(r["tasa_ppm_min_tiempo"]), bool(r["Saturado"]))
        for d, g in gas.groupby("dieta") for _, r in g.iterrows()
    ]

    def residuos(theta):
        amax = 10.0 ** theta[0]
        rho = 10.0 ** theta[1]
        f_d4 = theta[2]
        res = []
        curvas = {}
        for dieta in ("D1", "D4"):
            t, ppm, ph, _ = simular_dieta(dieta, amax, rho, f_d4, micro, t_eval)
            curvas[dieta] = (t, ppm, ph)
        for dieta, dia, obs, w in obs_crec:
            if dia > DIA_AJUSTE_MAX:
                continue  # opcion A: fase tardia fuera del costo
            t, _, ph = curvas[dieta]
            res.append((np.interp(dia, t, ph) - obs) / w)
        for dieta, dia, obs, sat in obs_gas:
            t, ppm, _ = curvas[dieta]
            r = np.interp(dia, t, ppm) - obs
            if sat:
                r = min(r, 0.0)  # cota inferior: solo penaliza por debajo
            res.append(r / PESO_GAS)
        return np.array(res)

    mejor = None
    for arranque in ARRANQUES:
        x0 = np.clip(np.array(arranque, dtype=float), LO + 1e-9, HI - 1e-9)
        sol = least_squares(residuos, x0, bounds=(LO, HI), method="trf")
        if mejor is None or sol.cost < mejor.cost:
            mejor = sol

    amax = 10.0 ** mejor.x[0]
    rho = 10.0 ** mejor.x[1]
    f_d4 = mejor.x[2]
    frontera = []
    for nombre, val, lo, hi in (
        ("a_max", mejor.x[0], LO[0], HI[0]),
        ("rho", mejor.x[1], LO[1], HI[1]),
        ("f_D4", mejor.x[2], LO[2], HI[2]),
    ):
        if abs(val - lo) < 1e-3 or abs(val - hi) < 1e-3:
            frontera.append(nombre)
    marca = f" [frontera: {','.join(frontera)}]" if frontera else ""
    print(f"\nCalibrados: a_max={amax:.4f} d^-1, rho={rho:.3f} mg C/d, f_D4={f_d4:.4f}{marca}")

    # Persistir parametros para reproducibilidad (los lee validar_externos.py)
    pdest = Path(args.salida).with_name("parametros_etapa2.csv")
    pd.DataFrame([
        dict(parametro="a_max", valor=amax, frontera=("a_max" in frontera)),
        dict(parametro="rho", valor=rho, frontera=("rho" in frontera)),
        dict(parametro="f_D4", valor=f_d4, frontera=("f_D4" in frontera)),
    ]).to_csv(pdest, index=False)
    print(f"Parametros: {pdest}")

    # Salida detallada y validacion de ayuno
    filas = []
    t_fino = np.arange(0.0, T_FIN + 0.05, 0.05)
    for dieta in ("D1", "D4"):
        t, ppm, ph, Y = simular_dieta(dieta, amax, rho, f_d4, micro, t_fino)
        for d, dia, obs, w in obs_crec:
            if d == dieta:
                mod = float(np.interp(dia, t, ph))
                filas.append(dict(bloque="crecimiento", dieta=dieta, dia_larvario=dia,
                                  obs=obs, mod=mod, residuo=mod - obs, saturado=False,
                                  en_ajuste=(dia <= DIA_AJUSTE_MAX)))
        for d, dia, obs, sat in obs_gas:
            if d == dieta:
                mod = float(np.interp(dia, t, ppm))
                filas.append(dict(bloque="gas", dieta=dieta, dia_larvario=dia,
                                  obs=obs, mod=mod, residuo=mod - obs, saturado=sat,
                                  en_ajuste=True))
        if dieta == "D1":
            p_larva = ParametrosEriksen(amax=amax, rho=rho, tp_min=TP_MIN)
            B14 = max(float(Y[0, int(np.argmin(np.abs(t - AYUNO_DIA)))]), 0.0)
            L14 = max(float(Y[1, int(np.argmin(np.abs(t - AYUNO_DIA)))]), 0.0)
            lar = tasas_larva(AYUNO_DIA, B14, L14, p_larva, factor_alimento=0.0)
            ayuno_mod = float(carbono_a_ppm_min(N_LARVAS * lar.rCO2_C / 1000.0, V_AIRE_L, T_K))

    comp = pd.DataFrame(filas)
    destino = Path(args.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(destino, index=False)
    print(f"Archivo: {destino}")

    for bloque, g in comp.groupby("bloque"):
        if bloque == "gas":
            ns = g[~g["saturado"]]
            nota = f"saturados excluidos: {int(g['saturado'].sum())}"
        else:
            ns = g[g["en_ajuste"]]
            tardios = g[~g["en_ajuste"]]
            nota = f"fase tardia solo diagnostico: {len(tardios)}"
        rmse = float(np.sqrt(np.mean(ns["residuo"] ** 2)))
        print(f"RMSE {bloque}: {rmse:.2f} (n={len(ns)}; {nota})")

    diag = comp[(comp["bloque"] == "crecimiento") & (~comp["en_ajuste"])]
    if not diag.empty:
        print("\nDiagnostico de fase tardia (NO entro al ajuste, opcion A):")
        print(diag[["dieta", "dia_larvario", "obs", "mod", "residuo"]].round(2).to_string(index=False))
    tardio_gas = comp[(comp["bloque"] == "gas") & (comp["dia_larvario"] > DIA_AJUSTE_MAX)]
    if not tardio_gas.empty:
        print("\nCierres de gas en fase post-pico (dia > 8, interpretar con la limitacion documentada):")
        print(tardio_gas[["dieta", "dia_larvario", "obs", "mod", "residuo", "saturado"]].round(2).to_string(index=False))

    print(f"\nValidacion independiente (ayuno, NO usado en el ajuste):")
    print(f"  obs={AYUNO_OBS:.2f} ppm/min, mod={ayuno_mod:.2f} ppm/min, ratio={ayuno_mod / AYUNO_OBS:.2f}")
    print("\nTabla:")
    print(comp.round(2).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib no disponible; se omite la figura")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    colores = {"D1": "tab:blue", "D4": "tab:orange"}
    for j, dieta in enumerate(("D1", "D4")):
        t, ppm, ph, _ = simular_dieta(dieta, amax, rho, f_d4, micro, t_fino)
        gc = crec[crec["dieta"] == dieta]
        axes[0][j].plot(t, ph, color=colores[dieta], label="modelo")
        axes[0][j].errorbar(gc["dia_larvario"], gc["ph_mg"], yerr=gc["de_mg"],
                            fmt="o", color=colores[dieta], capsize=3, label="obs")
        axes[0][j].axvspan(DIA_AJUSTE_MAX, T_FIN, color="gray", alpha=0.15)
        axes[0][j].set_title(f"Crecimiento {dieta} (gris = fase tardia, fuera del ajuste)")
        axes[0][j].legend(fontsize=8)
        gg = comp[(comp["bloque"] == "gas") & (comp["dieta"] == dieta)]
        axes[1][j].plot(t, ppm, color=colores[dieta], label="modelo")
        for sat, cara in ((False, colores[dieta]), (True, "none")):
            p = gg[gg["saturado"] == sat]
            if len(p):
                axes[1][j].plot(p["dia_larvario"], p["obs"], "^" if sat else "o",
                                color=colores[dieta], markerfacecolor=cara, linestyle="none")
        axes[1][j].axvspan(DIA_AJUSTE_MAX, T_FIN, color="gray", alpha=0.15)
        axes[1][j].set_title(f"Gas tratamiento {dieta} (gris = post-pico)")
        axes[1][j].set_xlabel("dia larvario")
    axes[0][0].set_ylabel("PH (mg/larva)")
    axes[1][0].set_ylabel("CO2 (ppm/min)")
    fig.suptitle(f"Etapa 2: a_max={amax:.3f}, rho={rho:.2f}, f_D4={f_d4:.3f} (triangulos = saturados)", fontsize=9)
    fig.tight_layout()
    ruta_fig = Path(args.figura)
    ruta_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_fig, dpi=200)
    print(f"\nFigura: {ruta_fig}")


if __name__ == "__main__":
    main()
