# Metodología IRB Fundación (F-IRB)

## Descripción General

El enfoque IRB Fundación (Foundation Internal Ratings-Based) es una metodología intermedia para el cálculo de requisitos de capital por riesgo de crédito, establecida en el Reglamento (UE) 575/2013 (CRR) y las directrices de la EBA.

## Características Principales

### Parámetros de Riesgo

| Parámetro | Fuente | Descripción |
|-----------|--------|-------------|
| **PD** (Probabilidad de Impago) | Estimación propia | La entidad desarrolla sus propios modelos de PD |
| **LGD** (Pérdida en caso de Impago) | Valores supervisores | Prescritos por el regulador (45% senior, 75% subordinado) |
| **EAD** (Exposición en caso de Impago) | Valores supervisores | Basados en factores de conversión regulatorios |
| **M** (Vencimiento) | Fórmula estándar | 2.5 años por defecto o cálculo específico |

### Valores Supervisores de LGD

Según el Artículo 161 del CRR:

- **Exposiciones senior sin garantías**: 45%
- **Exposiciones subordinadas**: 75%
- **Con garantías reales financieras**: Ajuste por colateral
- **Con garantías inmobiliarias**: 35% (residencial) o valor ajustado

### Factores de Conversión de Crédito (CCF)

Para exposiciones fuera de balance:

| Tipo de Exposición | CCF |
|-------------------|-----|
| Líneas de crédito comprometidas | 75% |
| Líneas incondicionales cancelables | 0% |
| Cartas de crédito comerciales | 20% |
| Avales y garantías financieras | 100% |

## Fórmula de RWA

Los activos ponderados por riesgo se calculan como:

```
RWA = K × 12.5 × EAD
```

Donde K es el requerimiento de capital calculado mediante la función de ponderación IRB:

```
K = [LGD × N[(1-R)^-0.5 × G(PD) + (R/(1-R))^0.5 × G(0.999)] - PD × LGD] × MA
```

Donde:
- N = Función de distribución normal acumulativa
- G = Función inversa de la distribución normal
- R = Correlación de activos
- MA = Ajuste por vencimiento

## Requisitos de Implementación

### Requisitos Mínimos

1. **Sistema de calificación interno**: Escala de rating con al menos 7 grados para deudores no impagados
2. **Datos históricos**: Mínimo 5 años de datos para estimación de PD
3. **Validación**: Proceso de validación independiente anual
4. **Documentación**: Políticas y procedimientos documentados

### Proceso de Aprobación

1. Solicitud al supervisor competente (BCE/BdE)
2. Evaluación de cumplimiento de requisitos mínimos
3. Período de uso paralelo (rollout)
4. Autorización permanente o con condiciones

## Ventajas del F-IRB

- Menor complejidad que IRB Avanzado
- Sensibilidad al riesgo mayor que método estándar
- Menor carga de datos que A-IRB
- Valores supervisores proporcionan estabilidad

## Limitaciones

- LGD y EAD no reflejan experiencia real de la entidad
- Menor beneficio de capital que A-IRB
- Dependencia de valores regulatorios que pueden ser conservadores

## Referencias Normativas

- Reglamento (UE) 575/2013, Artículos 142-166
- EBA GL/2017/16 - Directrices sobre estimación de PD
- EBA GL/2019/03 - Directrices sobre requisitos IRB
- Circular 3/2008 del Banco de España
