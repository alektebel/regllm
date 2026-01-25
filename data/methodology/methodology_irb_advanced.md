# Metodología IRB Avanzado (A-IRB)

## Descripción General

El enfoque IRB Avanzado (Advanced Internal Ratings-Based) representa el nivel más sofisticado para el cálculo de requisitos de capital por riesgo de crédito. Permite a las entidades utilizar sus propias estimaciones para todos los parámetros de riesgo.

## Características Principales

### Parámetros de Riesgo - Estimación Propia

| Parámetro | Requisitos | Mínimo de Datos |
|-----------|------------|-----------------|
| **PD** | Modelo interno calibrado | 5 años históricos |
| **LGD** | Estimación downturn | 7 años (incluir ciclo adverso) |
| **EAD** | Factores de conversión propios | 7 años históricos |
| **M** | Cálculo efectivo | Flujos contractuales |

### Estimación de LGD

La LGD bajo A-IRB debe:

1. **Reflejar condiciones económicas adversas** (downturn LGD)
2. **Considerar todos los costes**: directos, indirectos, descuentos temporales
3. **Segmentarse apropiadamente**: por tipo de garantía, antigüedad, producto
4. **Incluir márgenes de conservadurismo**: MoC (Margin of Conservatism)

#### Componentes de LGD

```
LGD = (EAD - Recuperaciones + Costes) / EAD
```

Donde las recuperaciones incluyen:
- Recuperaciones directas de colateral
- Recuperaciones de flujos de caja
- Descuento temporal a tasa de interés efectiva

### Estimación de EAD

Los factores de conversión de crédito (CCF) se estiman internamente:

```
CCF = (EAD en momento de default - Dispuesto actual) / (Límite - Dispuesto actual)
```

Requisitos:
- Horizonte de 12 meses previos al default
- Segmentación por tipo de producto y cliente
- Consideración de líneas incondicionales

## Fórmula de RWA

Idéntica a F-IRB pero con parámetros propios:

```
RWA = K × 12.5 × EAD
```

```
K = [LGD × N[(1-R)^-0.5 × G(PD) + (R/(1-R))^0.5 × G(0.999)] - PD × LGD] × MA
```

### Ajuste por Vencimiento (MA)

```
MA = (1 + (M - 2.5) × b) / (1 - 1.5 × b)
```

Donde:
- M = Vencimiento efectivo (años)
- b = Parámetro de sensibilidad al vencimiento

```
b = (0.11852 - 0.05478 × ln(PD))²
```

## Requisitos de Implementación

### Requisitos de Datos

| Parámetro | Mínimo General | Con Ciclo Completo |
|-----------|----------------|-------------------|
| PD | 5 años | Incluir período de estrés |
| LGD | 7 años | Representativo de downturn |
| EAD | 7 años | Período representativo |

### Validación

1. **Backtesting**: Comparación de predicciones vs. realizaciones
2. **Benchmarking**: Comparación con modelos externos
3. **Análisis de sensibilidad**: Impacto de cambios en inputs
4. **Revisión independiente**: Unidad de validación separada

### Documentación Requerida

- Política de modelos de riesgo de crédito
- Metodología de estimación de cada parámetro
- Procedimientos de recalibración
- Gobierno del sistema de rating
- Uso en la gestión (use test)

## Ventajas del A-IRB

- Mayor sensibilidad al riesgo real de la cartera
- Potencial ahorro significativo de capital
- Estimaciones reflejan experiencia histórica propia
- Incentivo para mejor gestión del riesgo

## Desventajas y Retos

- Alta complejidad de implementación
- Requisitos de datos extensos
- Mayor escrutinio supervisor
- Costes de mantenimiento elevados
- Volatilidad potencial en requerimientos

## Output Floor (Basilea III Final)

A partir de 2028 (implementación gradual):

```
RWA_final = max(RWA_IRB, 72.5% × RWA_estándar)
```

El floor limita el beneficio máximo respecto al método estándar.

## Referencias Normativas

- Reglamento (UE) 575/2013, Artículos 142-191
- EBA GL/2017/16 - Estimación de PD y LGD
- EBA GL/2019/03 - Requisitos IRB
- EBA RTS/2016/03 - Metodología de evaluación IRB
- Basilea III: Finalización del marco post-crisis (diciembre 2017)
