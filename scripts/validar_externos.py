#!/usr/bin/env python3
"""Validacion externa del modelo calibrado con datos independientes.

Estudios:
  - Ermolaev et al. (2019) Waste Management 96:65-74: 700 larvas de 5 d,
    food waste (MS 24.9 %), 28 C, 14 d, pulsos de 42/105/84/63 g MS/caja en
    los dias 0/2/7/11. Datos: 3 tasas puntuales + emision acumulada
    (tratamiento L = solo larvas).
  - Schon et al. (2026) J. Insect Physiol. 170:104961: 600 larvas
    (~1.3 mg MS al inicio), 80 g MS en lote unico, 27 C, 7 d, flujo abierto
    sin control de alimento solo. Datos: curvas horarias de VCO2
    (29 puntos/dieta) en 4 dietas de proteina decreciente.

Dos niveles, ambos con el nucleo y el microbiano de las etapas 1 y 2:
  Nivel A (transferencia pura): f = 1 en todas las dietas, cero reajuste.
    Es la prueba dura de transferibilidad.
  Nivel B (ajuste minimo, comparable con la version anterior del articulo):
    Ermolaev reajusta (f, tp_min, rho, Bmax0) contra las 3 tasas + acumulado;
    Schon reajusta solo f por dieta contra la curva horaria.

Transferencias declaradas (sin reajuste):
  - Estado inicial transplantado desde la trayectoria D1 propia calibrada:
    Ermolaev toma (B, Ltot) en t* = 0.5 d (sus larvas tienen 5 d; las
    nuestras 4.5 d al montaje); Schon toma (B, Ltot) donde la masa seca
    alcanza 1.3 mg (su masa inicial), con tp_min desplazado en -t*.
  - Modulo microbiano de D1 (k_ref y delta_CS de etapa 1), X0 = 1.0 g C,
    delta_C_alim = 0.47, humedad y Q10 como en el montaje propio; el Q10
    corrige a 28 C (Ermolaev) y 27 C (Schon). El nucleo larvario no tiene
    dependencia termica (igual que la version anterior).
  - El CO2 se compara en flujo masico/volumetrico (mg CO2/min por caja;
    mL CO2/min por camara a 27 C con 0.5596 mL/mg), sin volumen de camara.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import least_squares

from modelo import (
    ParametrosEriksen,
    ParametrosLecho,
    estado_inicial_desde_dw,
    rhs,
    tasas_larva,
    tasas_lecho,
)

# --- Transferencias declaradas (etapas 1 y 2 del experimento propio) ---
F_MS = 0.325
DELTA_C_ALIM = 0.47
X0_G_C = 1.0
K_D = 2.0
K_SM = 50.0
Y_X = 0.4
Y_CMIC = 0.4
TP_MIN_PROPIO = 8.5
T_C_PROPIO = 300.0 - 273.15
N_PROPIO = 700
PULSOS_PROPIOS = [(0.0, 250.0), (5.0, 250.0)]
MS_D1 = 0.863

# --- Ermolaev et al. (2019), tratamiento L ---
ERM_N = 700
ERM_T_C = 28.0
ERM_T_FIN = 14.0
ERM_PULSOS = {0.0: 42.0, 2.0: 105.0, 7.0: 84.0, 11.0: 63.0}  # g MS/caja
ERM_KG_FW = 1.171
ERM_T_STAR = 0.5            # sus larvas: 5 d; las nuestras: 4.5 d al montaje
ERM_MASA_FINAL_DM = 101.9   # mg MS/larva al dia 14 (su Table 1)

# --- Schon et al. (2026) ---
SCH_N = 600
SCH_T_C = 27.0
SCH_T_FIN = 7.0
SCH_DM_CAMA = 80.0          # g MS, lote unico
SCH_MASA_INI_MS = 1.3       # mg MS/larva al inicio
SCH_DIETAS = ["REF", "P:C 1:3", "P:C 1:5", "P:C 1:9"]
SCH_VCO2_TABLA = {"REF": 19.6, "P:C 1:3": 16.9, "P:C 1:5": 16.2, "P:C 1:9": 10.8}  # L

# Conversiones de CO2
MGCO2_MIN_POR_GC_D = (44.01 / 12.0) * 1000.0 / 1440.0  # mg CO2/min por (g C/d)
ML_POR_MG_CO2 = 0.5596  # 27 C, 1 atm (Vmolar = 24617 mL/mol, M = 44.01 g/mol)


def cargar_calibracion(ruta_e1, ruta_e2):
    """Lee k_ref y delta_CS de D1 (etapa 1) y a_max, rho, f_D4 (etapa 2)."""
    e1 = pd.read_csv(ruta_e1)
    falta = {"k_ref", "delta_CS", "dieta"} - set(e1.columns)
    if falta:
        raise AssertionError(f"calibracion_etapa1.csv sin columnas {falta}")
    d1 = e1[e1["dieta"] == "D1"]
    if d1.empty:
        raise AssertionError("etapa 1 sin fila D1")
    k_ref = float(d1["k_ref"].iloc[0])
    dcs = float(d1["delta_CS"].iloc[0])

    e2 = pd.read_csv(ruta_e2)
    falta = {"parametro", "valor"} - set(e2.columns)
    if falta:
        raise AssertionError(
            f"parametros_etapa2.csv sin columnas {falta}; "
            "ejecute primero scripts/calibrar_etapa2.py (version que lo genera)")
    pars = dict(zip(e2["parametro"], e2["valor"]))
    falta2 = {"a_max", "rho", "f_D4"} - set(pars)
    if falta2:
        raise AssertionError(f"parametros_etapa2.csv sin {falta2}")
    return k_ref, dcs, float(pars["a_max"]), float(pars["rho"]), float(pars["f_D4"])


def simular(p_larva, p_lecho, N, T_C, y0, pulsos_ms, dcs, t_fin, t_eval):
    """Integra por tramos entre pulsos (g MS -> g C degradable).

    El punto del dia de un pulso queda PRE-pulso (convencion de Ermolaev:
    midieron antes de alimentar).
    """
    y = np.array(y0, dtype=float)
    y[2] += pulsos_ms.get(0.0, 0.0) * DELTA_C_ALIM * dcs
    cortes = sorted(set([0.0] + [d for d in pulsos_ms if 0.0 < d < t_fin] + [t_fin]))
    ts, ys = [], []
    for j, (a, b) in enumerate(zip(cortes[:-1], cortes[1:])):
        t_seg = t_eval[(t_eval >= a) & (t_eval <= b)]
        if len(t_seg) < 2:
            t_seg = np.linspace(a, b, 3)
        sol = solve_ivp(rhs, (a, b), y, args=(p_larva, p_lecho, N, T_C),
                        method="LSODA", t_eval=t_seg, rtol=1e-6, atol=1e-9)
        if not sol.success:
            raise RuntimeError(f"fallo la integracion en [{a}, {b}]: {sol.message}")
        if j == 0:
            ts.append(sol.t)
            ys.append(sol.y)
        else:
            ts.append(sol.t[1:])
            ys.append(sol.y[:, 1:])
        y = sol.y[:, -1]
        if b in pulsos_ms:
            y[2] += pulsos_ms[b] * DELTA_C_ALIM * dcs
    t = np.concatenate(ts)
    Y = np.concatenate(ys, axis=1)
    R_C = np.empty_like(t)
    for k, (tk, yk) in enumerate(zip(t, Y.T)):
        B, Ltot, D_C, X = [max(float(v), 0.0) for v in yk[:4]]
        mic = tasas_lecho(D_C, X, T_C, p_lecho)
        lar = tasas_larva(tk, B, Ltot, p_larva, p_lecho.f_dieta * mic.phi_D)
        R_C[k] = N * lar.rCO2_C / 1000.0 + mic.rCmic
    m_dw = (np.maximum(Y[0], 0.0) / (p_larva.delta_CB * p_larva.delta_O)
            + np.maximum(Y[1], 0.0) / p_larva.delta_CL)
    return t, Y, R_C, m_dw


def trayectoria_d1_propia(a_max, rho, k_ref, dcs, crec):
    """Trayectoria D1 del montaje propio con los parametros calibrados."""
    sel = crec.loc[(crec["dieta"] == "D1") & (crec["dia_larvario"] == 0), "ph_mg"]
    if sel.empty:
        raise AssertionError("crecimiento_canonico.csv sin D1 dia 0")
    ph0 = float(sel.iloc[0])
    p_larva = ParametrosEriksen(amax=a_max, rho=rho, tp_min=TP_MIN_PROPIO)
    p_lecho = ParametrosLecho(f_dieta=1.0, K_D=K_D, k_ref=k_ref,
                              K_SM=K_SM, Y_X=Y_X, Y_Cmic=Y_CMIC)
    y0 = estado_inicial_desde_dw(ph0 * F_MS, 0.0, X0_G_C, p_larva)
    pulsos_ms = {d: g * MS_D1 for d, g in PULSOS_PROPIOS}
    t_eval = np.arange(0.0, 14.0 + 0.02, 0.02)
    return simular(p_larva, p_lecho, N_PROPIO, T_C_PROPIO, y0, pulsos_ms, dcs, 14.0, t_eval)


def trasplante(t, Y, m_dw, criterio, valor):
    """(B, Ltot, t*) para inicializar el escenario externo."""
    if criterio == "tiempo":
        t_star = float(valor)
    elif criterio == "masa":
        alcanza = np.where(m_dw >= valor)[0]
        t_star = float(t[alcanza[0]]) if len(alcanza) else 0.0
    else:
        raise AssertionError(f"criterio desconocido: {criterio}")
    i = int(np.argmin(np.abs(t - t_star)))
    return float(Y[0, i]), float(Y[1, i]), float(t[i])


def correr_ermolaev(obs, a_max, rho, k_ref, dcs, B5, L5):
    """Niveles A y B contra el tratamiento L; devuelve tabla y metricas."""
    Lobs = obs[obs["tratamiento"] == "L"].dropna(subset=["co2_mg_min"])
    dias_obs = Lobs["dia"].to_numpy(float)
    fl_obs = Lobs["co2_mg_min"].to_numpy(float)
    sd_obs = Lobs["sd_mg_min"].to_numpy(float)
    t_eval = np.arange(0.0, ERM_T_FIN + 0.02, 0.02)
    bmax0_def = ParametrosEriksen().Bmax0

    def sim(f, tp, rho_e, bmax0_e):
        p_larva = ParametrosEriksen(amax=a_max, rho=rho_e, tp_min=tp, Bmax0=bmax0_e)
        p_lecho = ParametrosLecho(f_dieta=f, K_D=K_D, k_ref=k_ref,
                                  K_SM=K_SM, Y_X=Y_X, Y_Cmic=Y_CMIC)
        y0 = np.array([B5, L5, 0.0, X0_G_C, 0.0, 0.0])
        t, Y, R_C, m_dw = simular(p_larva, p_lecho, ERM_N, ERM_T_C, y0,
                                  ERM_PULSOS, dcs, ERM_T_FIN, t_eval)
        fl = R_C * MGCO2_MIN_POR_GC_D
        cum = cumulative_trapezoid(fl * 1440.0 / 1000.0, t, initial=0.0) / ERM_KG_FW
        return t, fl, cum, m_dw

    def metricas(t, fl, cum, m_dw):
        en = lambda d: float(np.interp(d, t, fl))
        fl2, fl7, fl11 = en(2.0), en(7.0), en(11.0)
        pico = float(t[int(np.argmax(fl))])
        decl = (fl7 - fl11) / fl7 if fl7 > 0 else float("nan")
        cum14 = float(np.interp(14.0, t, cum))
        masa14 = float(np.interp(14.0, t, m_dw))
        crit = {
            "pico_dia_7_pm_1": abs(pico - 7.0) <= 1.0,
            "magnitud_pico_pm25": abs(fl7 - 8.04) / 8.04 <= 0.25,
            "declive_15_35": 0.15 <= decl <= 0.35,
            "acumulado_pm30": abs(cum14 - 92.4) / 92.4 <= 0.30,
        }
        return dict(fl2=fl2, fl7=fl7, fl11=fl11, pico=pico, declive=decl,
                    cum14=cum14, masa14=masa14, criterios=crit)

    # Nivel A: transferencia pura
    tA, flA, cumA, mA = sim(1.0, TP_MIN_PROPIO - ERM_T_STAR, rho, bmax0_def)
    metA = metricas(tA, flA, cumA, mA)

    # Nivel B: (f, tp_min, rho, Bmax0) contra 3 tasas + acumulado.
    # tp_min en el eje del ensayo (dia 0 = larvas de 5 d): cotas [5, 11]
    # equivalen a 10-16 d de edad fisiologica.
    def resid(par):
        f, tp, rho_e, bm = par
        t, fl, cum, _ = sim(f, tp, rho_e, bm)
        mod = np.array([float(np.interp(d, t, fl)) for d in dias_obs])
        r_t = (mod - fl_obs) / sd_obs
        r_a = np.array([(float(np.interp(14.0, t, cum)) - 92.4) / 14.5])
        return np.concatenate([r_t, r_a])

    sol = least_squares(resid, [0.9, TP_MIN_PROPIO - ERM_T_STAR, rho, bmax0_def],
                        bounds=([0.2, 5.0, 0.5, 30.0], [1.2, 11.0, 15.0, 65.0]))
    fB, tpB, rhoB, bmB = [float(v) for v in sol.x]
    tB, flB, cumB, mB = sim(fB, tpB, rhoB, bmB)
    metB = metricas(tB, flB, cumB, mB)

    filas = []
    for _, r in obs[obs["tratamiento"] == "L"].iterrows():
        d = float(r["dia"])
        filas.append(dict(
            dia=d,
            obs_mg_min=r["co2_mg_min"], sd_mg_min=r["sd_mg_min"],
            mod_A_mg_min=float(np.interp(d, tA, flA)),
            mod_B_mg_min=float(np.interp(d, tB, flB)),
            acum_obs_g_kg=r["acum_g_kg_fw"],
            acum_A_g_kg=float(np.interp(d, tA, cumA)),
            acum_B_g_kg=float(np.interp(d, tB, cumB)),
        ))
    tabla = pd.DataFrame(filas)
    parsB = dict(f=fB, tp_min=tpB, rho=rhoB, Bmax0=bmB)
    return tabla, metA, metB, parsB, (tA, flA, cumA), (tB, flB, cumB), Lobs, obs


def correr_schn(gas, crec, a_max, rho, k_ref, dcs, Bs, Ls, t_star):
    """Niveles A y B por dieta; devuelve tabla resumen y curvas."""
    tp = max(TP_MIN_PROPIO - t_star, 1.0)
    t_eval = np.arange(0.0, SCH_T_FIN + 0.02, 0.02)

    def sim(f):
        p_larva = ParametrosEriksen(amax=a_max, rho=rho, tp_min=tp)
        p_lecho = ParametrosLecho(f_dieta=f, K_D=K_D, k_ref=k_ref,
                                  K_SM=K_SM, Y_X=Y_X, Y_Cmic=Y_CMIC)
        y0 = np.array([Bs, Ls, 0.0, X0_G_C, 0.0, 0.0])
        t, Y, R_C, m_dw = simular(p_larva, p_lecho, SCH_N, SCH_T_C, y0,
                                  {0.0: SCH_DM_CAMA}, dcs, SCH_T_FIN, t_eval)
        fl = R_C * MGCO2_MIN_POR_GC_D * ML_POR_MG_CO2
        return t, fl, m_dw

    filas = []
    curvas = {}
    for dieta in SCH_DIETAS:
        o = gas[gas["dieta"] == dieta]
        do = o["dia"].to_numpy(float)
        fo = o["vco2_ml_min"].to_numpy(float)

        tA, flA, mA = sim(1.0)

        def resid(fp):
            t, fl, _ = sim(float(fp[0]))
            return np.interp(do, t, fl) - fo

        sol = least_squares(resid, [0.8], bounds=([0.2], [1.2]))
        fB = float(sol.x[0])
        tB, flB, mB = sim(fB)

        def resumen(t, fl, m):
            total_L = float(cumulative_trapezoid(fl * 1440.0 / 1000.0, t, initial=0.0)[-1])
            pico = float(t[int(np.argmax(fl))])
            masa = float(m[-1])
            return total_L, pico, masa

        totA, picoA, masaA = resumen(tA, flA, mA)
        totB, picoB, masaB = resumen(tB, flB, mB)
        cr = crec[crec["dieta"] == dieta]
        masa_obs = float(cr["masa_mg_ms"].iloc[-1])
        pico_obs = float(do[int(np.argmax(fo))])
        filas.append(dict(
            dieta=dieta, f_nivelB=fB,
            total_obs_L=SCH_VCO2_TABLA[dieta], total_A_L=totA, total_B_L=totB,
            pico_obs_dia=pico_obs, pico_A_dia=picoA, pico_B_dia=picoB,
            masa_obs_mg=masa_obs, masa_A_mg=masaA, masa_B_mg=masaB,
        ))
        curvas[dieta] = dict(obs=(do, fo), A=(tA, flA), B=(tB, flB), crec=cr)
    return pd.DataFrame(filas), curvas, tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa1", default="resultados/calibracion_etapa1.csv")
    ap.add_argument("--etapa2", default="resultados/parametros_etapa2.csv")
    ap.add_argument("--crecimiento", default="datos/procesados/crecimiento_canonico.csv")
    ap.add_argument("--ermolaev", default="datos/crudos/externos/ermolaev_co2.csv")
    ap.add_argument("--schn-gas", default="datos/crudos/externos/schn_gas.csv")
    ap.add_argument("--schn-crec", default="datos/crudos/externos/schn_crecimiento.csv")
    ap.add_argument("--salida-erm", default="resultados/validacion_ermolaev.csv")
    ap.add_argument("--salida-schn", default="resultados/validacion_schn.csv")
    ap.add_argument("--fig-erm", default="figuras/validacion_ermolaev.png")
    ap.add_argument("--fig-schn", default="figuras/validacion_schn.png")
    args = ap.parse_args()

    k_ref, dcs, a_max, rho, f_d4 = cargar_calibracion(args.etapa1, args.etapa2)
    print(f"Transferido: k_ref(D1)={k_ref:.4f}, delta_CS(D1)={dcs:.4f}, "
          f"a_max={a_max:.4f}, rho={rho:.3f} (f_D4={f_d4:.4f} no se usa aqui)")

    crec = pd.read_csv(args.crecimiento)
    t_d1, Y_d1, _, m_d1 = trayectoria_d1_propia(a_max, rho, k_ref, dcs, crec)

    B5, L5, t5 = trasplante(t_d1, Y_d1, m_d1, "tiempo", ERM_T_STAR)
    print(f"Transplante Ermolaev (t*={t5:.2f} d): B={B5:.3f} mg C, Ltot={L5:.3f} mg C")
    Bs, Lss, ts_s = trasplante(t_d1, Y_d1, m_d1, "masa", SCH_MASA_INI_MS)
    print(f"Transplante Schon (t*={ts_s:.2f} d): B={Bs:.3f} mg C, Ltot={Lss:.3f} mg C, "
          f"masa={float(np.interp(ts_s, t_d1, m_d1)):.2f} mg MS (objetivo {SCH_MASA_INI_MS})")

    # ---------------- Ermolaev ----------------
    obs_e = pd.read_csv(args.ermolaev, comment="#")
    tabla_e, metA, metB, parsB, curvaA, curvaB, Lobs, obs_e_full = correr_ermolaev(
        obs_e, a_max, rho, k_ref, dcs, B5, L5)

    print("\n=== Ermolaev et al. (2019), tratamiento L ===")
    print(tabla_e.round(2).to_string(index=False))
    for nombre, met in (("Nivel A (sin reajuste)", metA), ("Nivel B (f, tp, rho, Bmax0)", metB)):
        print(f"\n{nombre}:")
        print(f"  pico: dia {met['pico']:.1f} (obs 7), flujo dia 7 = {met['fl7']:.2f} mg/min (obs 8.04)")
        print(f"  declive 7->11: {100 * met['declive']:.0f} % (obs 31 % en L)")
        print(f"  acumulado dia 14: {met['cum14']:.1f} g/kg FW (obs 92.4 +- 14.5)")
        print(f"  masa final: {met['masa14']:.1f} mg MS/larva (obs {ERM_MASA_FINAL_DM})")
        for k, v in met["criterios"].items():
            print(f"  {k}: {'CUMPLE' if v else 'NO CUMPLE'}")
    print(f"\nNivel B calibrado: f={parsB['f']:.3f}, tp_min={parsB['tp_min']:.2f} d, "
          f"rho={parsB['rho']:.3f} mg C/d, Bmax0={parsB['Bmax0']:.2f} mg C")

    dest = Path(args.salida_erm)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tabla_e.to_csv(dest, index=False)
    print(f"Archivo: {dest}")

    # ---------------- Schon ----------------
    gas_s = pd.read_csv(args.schn_gas, comment="#")
    crec_s = pd.read_csv(args.schn_crec, comment="#")
    tabla_s, curvas_s, tp_sch = correr_schn(gas_s, crec_s, a_max, rho, k_ref, dcs, Bs, Lss, ts_s)

    print(f"\n=== Schon et al. (2026), 4 dietas (tp_min transferido = {tp_sch:.2f} d) ===")
    print(tabla_s.round(2).to_string(index=False))
    fs = tabla_s["f_nivelB"].tolist()
    ok_ord = all(fs[i] >= fs[i + 1] - 0.05 for i in range(len(fs) - 1))
    print(f"\nOrdenamiento f (nivel B) coherente con P:C: {'CUMPLE' if ok_ord else 'NO CUMPLE'}")
    for _, r in tabla_s.iterrows():
        for nivel in ("A", "B"):
            tot = r[f"total_{nivel}_L"]
            ok_t = abs(tot - r["total_obs_L"]) / r["total_obs_L"] <= 0.25
            masa = r[f"masa_{nivel}_mg"]
            ok_m = masa <= 2.0 * r["masa_obs_mg"]
            ok_p = abs(r[f"pico_{nivel}_dia"] - r["pico_obs_dia"]) <= 0.5
            print(f"  {r['dieta']:>8} nivel {nivel}: total +-25 % {'CUMPLE' if ok_t else 'NO CUMPLE'}; "
                  f"pico +-0.5 d {'CUMPLE' if ok_p else 'NO CUMPLE'}; "
                  f"masa <=2x {'CUMPLE' if ok_m else 'NO CUMPLE'}")

    dest = Path(args.salida_schn)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tabla_s.to_csv(dest, index=False)
    print(f"Archivo: {dest}")

    # ---------------- Figuras ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib no disponible; se omiten las figuras")
        return

    tA, flA, cumA = curvaA
    tB, flB, cumB = curvaB
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(tA, flA, "k-", lw=1.5, label="modelo A (sin reajuste)")
    ax[0].plot(tB, flB, "k--", lw=1.2, label="modelo B (f, tp, rho, Bmax0)")
    ax[0].errorbar(Lobs["dia"], Lobs["co2_mg_min"], yerr=Lobs["sd_mg_min"],
                   fmt="s", color="tab:blue", capsize=3, label="L (obs)")
    for tr, mk, c in [("BL", "o", "tab:orange"), ("PBL", "^", "tab:green")]:
        o = obs_e_full[obs_e_full["tratamiento"] == tr].dropna(subset=["co2_mg_min"])
        ax[0].plot(o["dia"], o["co2_mg_min"], mk, color=c, alpha=0.5, label=f"{tr} (obs)")
    for d in ERM_PULSOS:
        if d > 0:
            ax[0].axvline(d, color="gray", ls=":", lw=0.8)
    ax[0].set_xlabel("dia de tratamiento")
    ax[0].set_ylabel("CO2 (mg/min/caja)")
    ax[0].legend(fontsize=8)
    Lac = obs_e_full[obs_e_full["tratamiento"] == "L"].dropna(subset=["acum_g_kg_fw"])
    ax[1].plot(tA, cumA, "k-", lw=1.5, label="modelo A")
    ax[1].plot(tB, cumB, "k--", lw=1.2, label="modelo B")
    ax[1].errorbar(Lac["dia"], Lac["acum_g_kg_fw"], yerr=Lac["sd_acum"].fillna(0),
                   fmt="s", color="tab:blue", capsize=3, label="L (obs)")
    ax[1].set_xlabel("dia de tratamiento")
    ax[1].set_ylabel("CO2 acumulado (g/kg FW)")
    ax[1].legend(fontsize=8)
    fig.suptitle("Validacion externa con Ermolaev et al. (2019), tratamiento L")
    fig.tight_layout()
    ruta = Path(args.fig_erm)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200)
    print(f"Figura: {ruta}")

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for k, dieta in enumerate(SCH_DIETAS):
        a = ax[k // 2][k % 2]
        cv = curvas_s[dieta]
        do, fo = cv["obs"]
        tA, flA = cv["A"]
        tB, flB = cv["B"]
        a.plot(tA, flA, "k-", lw=1.4, label="modelo A (sin reajuste)")
        a.plot(tB, flB, "k--", lw=1.1, label="modelo B (solo f)")
        a.plot(do, fo, "s", color="tab:blue", ms=3, alpha=0.7, label="obs (digitalizado)")
        cr = cv["crec"]
        a2 = a.twinx()
        a2.plot(cr["dia"], cr["masa_mg_ms"], "o-", color="tab:orange", ms=3, lw=1,
                alpha=0.8, label="masa obs (mg MS)")
        a2.set_ylabel("mg MS/larva", fontsize=8)
        fB = float(tabla_s.loc[tabla_s["dieta"] == dieta, "f_nivelB"].iloc[0])
        a.set_title(f"{dieta}  (f nivel B = {fB:.2f})", fontsize=10)
        a.set_xlabel("dias")
        a.set_ylabel("VCO2 (mL/min)")
        a.legend(fontsize=7, loc="upper left")
    fig.suptitle("Validacion externa con Schon et al. (2026), respirometria de flujo abierto")
    fig.tight_layout()
    ruta = Path(args.fig_schn)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200)
    print(f"Figura: {ruta}")


if __name__ == "__main__":
    main()
