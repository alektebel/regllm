---
id: "alert_disk_full_lgd_partition"
type: "insight"
priority: 0.5
tags: [alert, disk, SAS, storage, partición]
fields: []
articles: []
source: "Alerta Nagios SAS SERVER — /data/lgd 94% lleno (2025-04-12)"
feedback: false
---

# [ALERTA] Disco /data/lgd al 94% — riesgo de fallo en ejecución mensual

Fuente: Nagios
Host: sas-server-02
Alerta: DISK_CRITICAL - /data/lgd 94% usado (4.2TB/4.5TB)

## Causa raíz
Las tablas intermedias del pipeline LGD no se limpian tras cada ejecución mensual. En particular:
- `mylib.cycles_check` (check DQC): 47 particiones × 3 meses retenidos = 141 tablas
- `mylib.lgd_scenarios`: 12 escenarios × 3 meses = 36 tablas
- Logs SAS: ~200MB/día, rotación a 90 días

## Impacto potencial
Si el disco llega al 100%, la ejecución mensual falla y no se genera COREP/FINREP. SLA de reporting: T+5 días hábiles.

## Fix inmediato
```bash
# Limpiar tablas intermedias con más de 60 días
find /data/lgd/sasdata -name 'cycles_check_*' -mtime +60 -exec rm {} \;
find /data/lgd/sasdata -name 'lgd_scenarios_*' -mtime +60 -exec rm {} \;
```

## Fix permanente
Añadir cleanup al final del pipeline:
```sas
PROC DATASETS LIBRARY=mylib NOLIST;
    DELETE cycles_check_: / MAXAGE=60;
    DELETE lgd_scenarios_: / MAXAGE=60;
RUN;
```

## Lección
Los pipelines batch siempre deben incluir cleanup de tablas intermedias. El espacio es barato hasta que se acaba.
