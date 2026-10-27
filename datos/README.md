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
