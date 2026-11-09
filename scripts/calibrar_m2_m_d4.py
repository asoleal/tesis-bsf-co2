#!/usr/bin/env python3
"""M2: a_max y rho comunes; D4 cambia m, con asimilacion f_dieta=1."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calibrar_etapa2 as e2
from modelo import (
    ParametrosEriksen, ParametrosLecho, carbono_a_ppm_min,
    estado_inicial_desde_dw, rhs, tasas_larva, tasas_lecho,
)

M_D1 = ParametrosEriksen().m
LO = np.array([np.log10(1.0), np.log10(0.5), np.log10(M_D1)])
HI = np.array([np.log10(1.5), np.log10(15.0), np.log10(0.36)])
ARRANQUES = [
    (1.05, 1.0, 0.12), (1.40, 1.0, 0.24),
    (1.05, 8.0, 0.16), (1.40, 8.0, 0.32),
]


def simular(dieta, amax, rho, m_d4, micro, t_eval):
    k_ref, dcs = micro[dieta]
    m = M_D1 if dieta == "D1" else m_d4
    p_larva = ParametrosEriksen(
        amax=amax, rho=rho, tp_min=e2.TP_MIN, m=m,
    )
    p_lecho = ParametrosLecho(
        f_dieta=1.0, K_D=e2.K_D, k_ref=k_ref, K_SM=e2.K_SM,
        Y_X=e2.Y_X, Y_Cmic=e2.Y_CMIC,
    )
    y = estado_inicial_desde_dw(
        e2.PH0[dieta] * e2.F_MS, 0.0, e2.X0_G_C, p_larva,
    )
    cortes = sorted(set([0.0] + [d for d, _ in e2.PULSOS] + [e2.T_FIN]))
    pulsos = dict(e2.PULSOS)
    ts, ys = [], []

    for a, b in zip(cortes[:-1], cortes[1:]):
        if a in pulsos:
            y[2] += (
                pulsos[a] * e2.MS_DIETA[dieta] * e2.DELTA_C_ALIM * dcs
            )
        t_seg = t_eval[(t_eval >= a) & (t_eval <= b)]
        if len(t_seg) < 2:
            t_seg = np.linspace(a, b, 3)
        sol = solve_ivp(
            rhs, (a, b), y,
            args=(p_larva, p_lecho, e2.N_LARVAS, e2.T_C),
            method="LSODA", t_eval=t_seg, rtol=1e-6, atol=1e-9,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        ts.append(sol.t)
        ys.append(sol.y)
        y = sol.y[:, -1]

    t = np.concatenate(ts)
    Y = np.concatenate(ys, axis=1)
    R_C = np.empty_like(t)
    for k, (tk, yk) in enumerate(zip(t, Y.T)):
        B, Ltot, D_C, X, _, _ = [max(float(v), 0.0) for v in yk]
        mic = tasas_lecho(D_C, X, e2.T_C, p_lecho)
        lar = tasas_larva(tk, B, Ltot, p_larva, mic.phi_D)
        R_C[k] = e2.N_LARVAS * lar.rCO2_C / 1000.0 + mic.rCmic

    ppm = carbono_a_ppm_min(R_C, e2.V_AIRE_L, e2.T_K)
    B = np.maximum(Y[0], 0.0)
    Ltot = np.maximum(Y[1], 0.0)
    ph = (
        (1.0 - p_larva.delta_LB) * B
        / (p_larva.delta_CB * p_larva.delta_O)
        + Ltot / p_larva.delta_CL
    ) / e2.F_MS
    return t, ppm, ph, Y


def main():
    micro = e2.cargar_microbiano(
        "resultados/dcs_fija/calibracion_etapa1.csv"
    )
    crec = pd.read_csv("datos/procesados/crecimiento_canonico.csv")
    cierres = pd.read_csv("datos/procesados/cierres_etiquetados.csv")
    gas = cierres[
        (~cierres["excluir_promedio"])
        & (cierres["Gas"] == "co2")
        & (cierres["condicion"] == "tratamiento")
    ].copy()
    gas["dia_larvario"] = gas["Dia_Experimento"] - e2.DIA_MONTAJE_SEP

    e2.PH0 = {
        d: float(g.loc[g["dia_larvario"] == 0, "ph_mg"].iloc[0])
        for d, g in crec.groupby("dieta")
    }
    obs_crec = [
        (d, float(r["dia_larvario"]), float(r["ph_mg"]),
         max(float(r["de_mg"]), 0.1 * float(r["ph_mg"])))
        for d, g in crec.groupby("dieta") for _, r in g.iterrows()
    ]
    obs_gas = [
        (d, float(r["dia_larvario"]),
         float(r["tasa_ppm_min_tiempo"]), bool(r["Saturado"]))
        for d, g in gas.groupby("dieta") for _, r in g.iterrows()
    ]
    t_eval = np.arange(0.0, e2.T_FIN + 0.1, 0.1)

    def residuos(theta):
        amax, rho, m_d4 = 10 ** theta
        curvas = {
            d: simular(d, amax, rho, m_d4, micro, t_eval)[:3]
            for d in ("D1", "D4")
        }
        res = []
        for d, dia, obs, w in obs_crec:
            if dia <= e2.DIA_AJUSTE_MAX:
                t, _, ph = curvas[d]
                res.append((np.interp(dia, t, ph) - obs) / w)
        for d, dia, obs, sat in obs_gas:
            t, ppm, _ = curvas[d]
            r = np.interp(dia, t, ppm) - obs
            if sat:
                r = min(r, 0.0)
            res.append(r / e2.PESO_GAS)
        return np.asarray(res)

    mejor = None
    for a0, r0, y0 in ARRANQUES:
        x0 = np.clip(np.log10([a0, r0, y0]), LO + 1e-9, HI - 1e-9)
        sol = least_squares(residuos, x0, bounds=(LO, HI), method="trf")
        if mejor is None or sol.cost < mejor.cost:
            mejor = sol

    amax, rho, m_d4 = 10 ** mejor.x
    frontera = [
        n for n, x, lo, hi in zip(
            ("a_max", "rho", "m_D4"), mejor.x, LO, HI
        ) if abs(x - lo) < 1e-3 or abs(x - hi) < 1e-3
    ]
    print(
        f"M2: a_max={amax:.4f}, rho={rho:.3f}, "
        f"m_D4={m_d4:.4f}; frontera={frontera}"
    )

    filas = []
    t_fino = np.arange(0.0, e2.T_FIN + 0.05, 0.05)
    for dieta in ("D1", "D4"):
        t, ppm, ph, Y = simular(
            dieta, amax, rho, m_d4, micro, t_fino
        )
        for d, dia, obs, _ in obs_crec:
            if d == dieta:
                mod = float(np.interp(dia, t, ph))
                filas.append({
                    "bloque": "crecimiento", "dieta": d,
                    "dia_larvario": dia, "obs": obs, "mod": mod,
                    "residuo": mod - obs, "saturado": False,
                    "en_ajuste": dia <= e2.DIA_AJUSTE_MAX,
                })
        for d, dia, obs, sat in obs_gas:
            if d == dieta:
                mod = float(np.interp(dia, t, ppm))
                filas.append({
                    "bloque": "gas", "dieta": d,
                    "dia_larvario": dia, "obs": obs, "mod": mod,
                    "residuo": mod - obs, "saturado": sat,
                    "en_ajuste": True,
                })
        if dieta == "D1":
            p_d1 = ParametrosEriksen(
                amax=amax, rho=rho, tp_min=e2.TP_MIN, YB=M_D1,
            )
            j = int(np.argmin(np.abs(t - e2.AYUNO_DIA)))
            lar = tasas_larva(
                e2.AYUNO_DIA, max(Y[0, j], 0.0), max(Y[1, j], 0.0),
                p_d1, factor_alimento=0.0,
            )
            ayuno = carbono_a_ppm_min(
                e2.N_LARVAS * lar.rCO2_C / 1000.0,
                e2.V_AIRE_L, e2.T_K,
            )

    comp = pd.DataFrame(filas)
    destino = Path("resultados/m2_m_d4")
    destino.mkdir(parents=True, exist_ok=True)
    comp.to_csv(destino / "calibracion_etapa2.csv", index=False)
    pd.DataFrame([
        {"parametro": n, "valor": v, "frontera": n in frontera}
        for n, v in zip(("a_max", "rho", "m_D4"), (amax, rho, m_d4))
    ]).to_csv(destino / "parametros_etapa2.csv", index=False)

    cg = comp[
        (comp["bloque"] == "crecimiento") & comp["en_ajuste"]
    ]
    gg = comp[(comp["bloque"] == "gas") & ~comp["saturado"]]
    rmse_c = np.sqrt(np.mean(cg["residuo"] ** 2))
    rmse_g = np.sqrt(np.mean(gg["residuo"] ** 2))
    print(f"RMSE crecimiento={rmse_c:.2f} mg PH")
    print(f"RMSE gas={rmse_g:.2f} ppm/min")
    print(
        f"Ayuno: mod={ayuno:.2f}, obs={e2.AYUNO_OBS:.2f}, "
        f"ratio={ayuno / e2.AYUNO_OBS:.2f}"
    )
    print(f"Archivo: {destino / 'calibracion_etapa2.csv'}")


if __name__ == "__main__":
    main()
