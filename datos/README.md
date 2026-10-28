# Datos

## Crudos

- `crudos/feeds.csv`: registros crudos de sensores.
- `crudos/Datos pesos trabajo Jonathan y Andres.xlsx`: libro original de crecimiento y datos experimentales asociados.

## Procesados heredados

- `procesados/crecimiento_propio.csv`: promedios y desviaciones estándar de crecimiento ya calculados.
- `procesados/datos_finales_PINN_corregidos.csv`: tasas históricas depuradas. Sus pendientes están en ppm por registro, no en ppm/min; se conservan para trazabilidad. Las tasas canónicas se recalcularán desde `feeds.csv`.

## Verificación

Desde la raíz del repositorio:

```bash
sha256sum -c datos/manifest_sha256.txt

## Sincronizacion gas-crecimiento

Hipotesis documentada (pendiente de confirmacion con bitacora de laboratorio):
el montaje del experimento (dia larvario 0, larvas de ~4 mg PH) fue el
2025-09-05.

Justificacion: con esa fecha, los cierres de gas del 9, 11 y 13 de septiembre
coinciden exactamente con los dias de pesaje 4, 6 y 8; solo el cierre del
17 de septiembre (dia larvario 12) no tiene pesaje y requiere interpolacion.

Correspondencia:
- Cierre 2025-09-09 -> dia larvario 4 (pesaje disponible)
- Cierre 2025-09-11 -> dia larvario 6 (pesaje disponible)
- Cierre 2025-09-13 -> dia larvario 8 (pesaje disponible)
- Cierre 2025-09-17 -> dia larvario 12 (sin pesaje; interpolar entre 10 y 14)

Nota: el encabezado de `crecimiento_propio.csv` dice "ensayo paralelo", pero
el diseno real fue pesaje no destructivo de una muestra de 10 larvas de las
mismas cajas del experimento de gas (mismo lote y mismas dietas).

Nota sobre MS de larvas: el analisis bromatologico del Excel reporta MS de
larva secada para analisis (~97 %), no el contenido de materia seca de larva
viva. La conversion PH -> MS -> carbono se definira aparte.

## Regimen de alimentacion y volumen de camara (correcciones al articulo viejo)

Regimen real de cargas (confirmado por el experimentador):
- Dia larvario 1 (2025-09-06): 250 g tal como se recibio.
- Dia larvario 5 (2025-09-10): 250 g tal como se recibio.
- Total 500 g, sin retiros, hasta el dia 15. Consistente con PAL = 500 g
  de la hoja Bioconversion del Excel de pesos.
- El texto del articulo viejo decia 500 g + 250 g: era incorrecto.
- Controles de alimento: mismo regimen de cargas, sin larvas.
- Larvas en ayuno: camara vacia, sin sustrato.

Materia seca por carga de 250 g (bromatologia propia):
- D1 (MS 86.3 %): 215.75 g MS por carga.
- D4 (MS 80.3 %): 200.75 g MS por carga.

Volumen de aire efectivo (estimacion geometrica, sensibilidad +-0.5 L):
- Camara troncopiramidal: 13.8 L (boca 37x26, base 32x21, h 17 cm).
- Recipiente interno real: 12x12x5 cm = 0.72 L (el suplementario viejo
  decia 15x12x5: era incorrecto).
- V_aire = 13.8 - 0.72 = 13.08 L.
- Temperatura de cierre: T = 300 K.

## Edad larvaria al montaje y desfase de etiquetado (confirmados por el experimentador)

- Las larvas tenian 4.5 d de edad al montaje. Por tanto el tp_min = 13 d de
  Eriksen (desde eclosion) corresponde a tp_min = 8.5 d desde el montaje.
  Se usa fijo, no calibrado.
- Desfase Delta = 2 d entre el etiquetado de medicion y el dia de montaje
  (notas de laboratorio): dia larvario de gas = dia calendario desde
  montaje - 2. Con ello los cierres corresponden a los dias larvarios
  2, 4, 6 y 10, todos con pesaje directo (sincronizacion 4 de 4).
- V_aire: el 12.1 L del articulo viejo era estimado; el valor de trabajo es
  13.08 L (reconstruccion geometrica, seccion anterior).
- delta_DW,0 = 0.26 del articulo viejo no es trazable (posiblemente de la
  base de datos de los experimentadores, sin confirmar); se mantiene
  f_MS = 0.325 de literatura con banda [0.30, 0.35].

## Correccion de la hipotesis de montaje (2025-09-03)

El desfase Delta=2 d del articulo viejo se probo y se rechazo: llevaba el
pico de gas (670 ppm/min) al dia larvario 2, donde 700 larvas tienen ~2 g
de carbono corporal; respirar >=6.1 g C/d es fisicamente imposible
(~3 veces el carbono corporal por dia). Ese Delta se referia al etiquetado
de archivos del articulo viejo, no al calendario.

Nueva hipotesis documentada: montaje = 2025-09-03. Los cierres caen en los
dias larvarios 6, 8, 10 y 14. Justificacion triple:
1. Sincronizacion 4 de 4 con dias de pesaje (6, 8, 10, 14).
2. El pico de gas (dia 6) coincide con el pico de crecimiento observado.
3. El balance de carbono es factible (tasa especifica ~0.34 d^-1 en el
   pico voraz, alta pero plausible).
El ayuno del 17 sep corresponde entonces al dia larvario 14 (prepupa).

## Decisiones de la etapa 2 de calibracion (2025, sesion de reconstruccion)

- **Primer pulso en t = 0**: las larvas llegaron de la crianza con
  contenido intestinal, por lo que la alimentacion efectiva arranca con
  el montaje y no en el dia larvario 1. Aplicado en
  `scripts/calibrar_etapa2.py` y `scripts/comparar_gas.py`.
- **Opcion A (fase tardia)**: el nucleo de Eriksen no cataboliza B ni
  lipidos y mantiene el mantenimiento pleno en prepupa; el descenso de
  PH observado post-pico es mayormente agua. El ajuste de crecimiento
  se restringe a dias <= 8 (fase de crecimiento/pico); los dias 10 y 14
  se reportan solo como diagnostico. Es una limitacion estructural
  documentada del nucleo, no un ajuste ad hoc.
- **K_D = 2 g C**: con K_D = 50 g C la dieta D4 colapsaba por hambre
  (phi_D ~ 0.3-0.46) y el ajuste se iba a las fronteras; K_D = 2 g C
  (como el K_s del articulo anterior) produce un ajuste sano e interior.
- Resultados etapa 2 (maquina canonica): a_max = 1.4949 d^-1 (a 0.3 %
  del tope 1.5, se reporta), rho = 4.446 mg C/d, f_D4 = 0.8006;
  RMSE crecimiento (fase ajustada) = 38.85 mg, gas = 59.74 ppm/min;
  validacion de ayuno (fuera del ajuste) ratio = 6.53.

## Correccion Ec. 19 de Eriksen (factor 1-delta_LB en conversion a peso)

La conversion de (B, Ltot) a peso seco/humedo llevaba B/(delta_CB*delta_O)
sin el factor (1-delta_LB) de la Ec. 19 (los lipidos estructurales ya estan
en Ltot). Corregida en comparar_crecimiento.py, calibrar_etapa2.py y
validar_externos.py; el articulo ya incluye las Ecs. 19-22 exactas.
Nuevos valores canonicos etapa 2: a_max = 1.4995 d^-1 (frontera),
rho = 4.395 mg C/d, f_D4 = 0.8047; RMSE crecimiento = 35.65 mg PH,
RMSE gas = 60.89 ppm/min; ayuno ratio = 6.54. Conclusiones de validacion
externa sin cambio (Ermolaev B 4/4; Schon B totales y masas 4/4).
