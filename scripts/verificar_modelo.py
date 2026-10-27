#!/usr/bin/env python3
"""Verificaciones estructurales del modelo; no usa datos experimentales."""

from pathlib import Path
import sys

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from modelo import (  # noqa: E402
    ParametrosEriksen,
    ParametrosLecho,
    estado_inicial_desde_dw,
    tasas_larva,
    rhs,
    simular,
    componentes,
    carbono_a_ppm_min,
)


def cierre_asimilados(r) -> float:
    return r.rA - (r.rB + r.rL + r.rCm + r.rCB + r.rCL)


def main() -> None:
    p_larva = ParametrosEriksen()
    p_lecho = ParametrosLecho(
        f_dieta=1.0,
        K_D=20.0,
        k_ref=0.05,
        K_SM=5.0,
        Y_X=0.4,
        Y_Cmic=0.3,
        Q10=2.0,
        T_ref_C=25.0,
        theta=0.7,
        K_theta=0.15,
    )

    # 1. El balance del pool de asimilados debe cerrar para estados diversos.
    for t in [0.0, 5.0, 13.0, 18.0]:
        for B in [0.05, 1.0, 5.0, 20.0, 35.0]:
            r = tasas_larva(t, B, 1.0, p_larva)
            assert abs(cierre_asimilados(r)) < 1e-10, (t, B, cierre_asimilados(r))

    # 2. En prepupa no hay asimilación ni crecimiento estructural.
    t_prep = 12.0
    B_prep = p_larva.Bmax0
    r = tasas_larva(t_prep, B_prep, 1.0, p_larva)
    assert r.instar == 7
    assert r.a_eff == 0.0
    assert r.rB == 0.0

    # 3. Sin carbono degradable no hay asimilación desde el alimento.
    y0_sin = estado_inicial_desde_dw(0.6, 0.0, 0.5, p_larva)
    dy_sin = rhs(5.0, y0_sin, p_larva, p_lecho, N=700, T_C=27.0)
    assert abs(dy_sin[2]) < 1e-14

    # 4. Integración, no negatividad y conservación del carbono único.
    t = np.linspace(0.0, 25.0, 501)
    y0 = estado_inicial_desde_dw(0.6, 200.0, 0.5, p_larva)
    sol = simular(t, y0, p_larva, p_lecho, N=700, T_C=27.0)
    assert np.all(np.isfinite(sol.y))
    assert np.min(sol.y) > -1e-7

    B, Ltot, D_C, X, C_CO2, C_res = sol.y
    carbono_total = (
        700 * ((1.0 - p_larva.delta_LB) * B + Ltot) / 1000.0
        + D_C + X + C_CO2 + C_res
    )
    error_rel = np.max(np.abs(carbono_total - carbono_total[0]) / carbono_total[0])
    assert error_rel < 1e-7, error_rel

    # 5. Las fuentes y la conversión a ppm/min deben ser finitas.
    comp = componentes(sol, p_larva, p_lecho, N=700, T_C=27.0)
    ppm = carbono_a_ppm_min(comp["R_C_total"], V_aire_L=12.1, T_K=300.15)
    assert np.all(np.isfinite(ppm))
    assert np.max(ppm) > 0.0

    print("MODELO OK")
    print(f"error relativo de carbono: {error_rel:.3e}")
    print(f"instar final: {tasas_larva(t[-1], B[-1], Ltot[-1], p_larva).instar}")
    print(f"pendiente máxima de prueba: {np.max(ppm):.3f} ppm/min")


if __name__ == "__main__":
    main()
