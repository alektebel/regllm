---
id: "alert_sas_session_limit"
type: "insight"
priority: 0.5
tags: [alert, SAS, session, concurrency, batch]
fields: []
articles: []
source: "Alerta SAS Workspace Server — límite sesiones concurrentes alcanzado"
feedback: false
---

# [ALERTA] SAS Workspace Server — límite de 50 sesiones alcanzado

Fuente: SAS Management Console
Servidor: SASApp — Workspace Server
Evento: "Maximum number of workspace server sessions (50) reached"

## Causa
El nuevo proceso de validación DQC interactivo lanza una sesión SAS independiente por cada validación. Los usuarios (equipo de negocio) lanzan validaciones desde la herramienta web que no cierran las sesiones al terminar. Sesiones huérfanas se acumulan.

## Impacto
A las 16:30 se alcanzó el límite de 50 sesiones. La ejecución batch del pipeline LGD (que arranca a las 18:00) encontró 0 sesiones disponibles y falló. El reporting mensual se retrasó 1 día.

## Fix inmediato
- Aumentar límite a 100 sesiones temporalmente
- Matar sesiones inactivas >30 min:
  ```bash
  for pid in $(ls -1 /opt/sas/config/Lev1/SASApp/WorkspaceServer/logs/*.log | xargs grep -l 'inactive for 30 minutes' | sed 's/.*_//;s/\.log//'); do kill -9 $pid; done
  ```

## Fix permanente
- Implementar timeout de sesión en Workspace Server (actualmente no tiene)
- La herramienta web debe cerrar sesiones SAS explícitamente vía API

## Lección
Las sesiones SAS son un recurso finito compartido entre batch e interactivo. Monitorear uso y establecer prioridades.
