/** In-browser fake backend for the static showcase (GitHub Pages).
 *
 * Active when the page is loaded with `?demo=1` (or built with the demo
 * flag). Replays the same workflow the real pipeline produces — inspection
 * proposal, plan-mode ReAct stream with decision traces, detected cases,
 * evaluation metrics — from canned template data, so the UI can be shown
 * with no API, no model and no AWS.
 */
import { Observable, of } from 'rxjs';
import { delay } from 'rxjs/operators';
import {
  CheckCasesResponse, CheckRecord, CountsResponse, DashboardResponse,
  EvaluateResponse, InspectResponse, StreamEvent,
} from '../models/dqc.model';

export const DEMO_MODE: boolean =
  typeof location !== 'undefined' &&
  (location.search.includes('demo') || location.hostname.endsWith('github.io'));

const COLS = ['PD_ESTIMADA', 'EAD_TOTAL', 'LGD_ESTIMADA', 'ECL', 'STAGE_IFRS9', 'DQC_ID'];

const ROWS: Record<string, Record<string, string>[]> = {
  pd: [
    { PD_ESTIMADA: '1.15', EAD_TOTAL: '12000', LGD_ESTIMADA: '0.45', ECL: '6210', STAGE_IFRS9: '2', DQC_ID: 'DQC_PD_001' },
    { PD_ESTIMADA: '1.3', EAD_TOTAL: '-20', LGD_ESTIMADA: '0.35', ECL: '-9.1', STAGE_IFRS9: '2', DQC_ID: 'DQC_PD_001;DQC_EAD_002' },
  ],
  ead: [
    { PD_ESTIMADA: '0.02', EAD_TOTAL: '-500', LGD_ESTIMADA: '0.4', ECL: '-4', STAGE_IFRS9: '1', DQC_ID: 'DQC_EAD_002' },
    { PD_ESTIMADA: '1.3', EAD_TOTAL: '-20', LGD_ESTIMADA: '0.35', ECL: '-9.1', STAGE_IFRS9: '2', DQC_ID: 'DQC_PD_001;DQC_EAD_002' },
  ],
  ecl: [
    { PD_ESTIMADA: '0.85', EAD_TOTAL: '30000', LGD_ESTIMADA: '0.55', ECL: '99999', STAGE_IFRS9: '3', DQC_ID: 'DQC_ECL_003' },
  ],
  lgd: [
    { PD_ESTIMADA: '0.5', EAD_TOTAL: '1000', LGD_ESTIMADA: '1.2', ECL: '600', STAGE_IFRS9: '2', DQC_ID: '' },
  ],
};

const TRACE_OK = [
  { paso: 'suficiencia', pregunta: '¿Información suficiente?', resultado: 'si',
    detalle: 'Campos implicados identificados en el diccionario' },
  { paso: 'generacion', intento: 1, accion: 'Generar consulta SAS' },
  { paso: 'validacion', pregunta: '¿Consulta válida?', resultado: 'si', detalle: '' },
  { paso: 'resultado', estado: 'completado', n_casos: 2 },
];

const TRACE_CORRECTED = [
  { paso: 'suficiencia', pregunta: '¿Información suficiente?', resultado: 'si',
    detalle: 'EAD_TOTAL es un importe en euros, no negativo' },
  { paso: 'generacion', intento: 1, accion: 'Generar consulta SAS' },
  { paso: 'validacion', pregunta: '¿Consulta válida?', resultado: 'si', detalle: '' },
  { paso: 'juicio', pregunta: '¿El juez la aprueba?', resultado: 'no',
    detalle: 'El sentido está invertido: selecciona las filas que CUMPLEN la regla.' },
  { paso: 'generacion', intento: 2, accion: 'Generar consulta SAS' },
  { paso: 'validacion', pregunta: '¿Consulta válida?', resultado: 'si', detalle: '' },
  { paso: 'juicio', pregunta: '¿El juez la aprueba?', resultado: 'si',
    detalle: 'La consulta selecciona exactamente las filas que violan la regla.' },
  { paso: 'resultado', estado: 'completado', n_casos: 2 },
];

function check(over: Partial<CheckRecord>): CheckRecord {
  return {
    check_id: 'chk_demo', rule_id: null, name: 'dqc_demo', description: '',
    severity: 'HIGH', category: 'rango', sql: '', visible: true,
    status: 'pending', reward: null, variable: null, tipo: 'rango',
    condicion_error: null, campos_entrada: [], referencia_regulatoria: null,
    umbral: null, periodicidad: 'mensual', justificacion: null,
    created_at: new Date().toISOString(), validated_at: null, ...over,
  };
}

const SEED: CheckRecord[] = [
  check({
    check_id: 'chk_pd', rule_id: 'DQC_PD_001', name: 'dqc_pd_estimada_001',
    description: 'La PD estimada debe estar en el rango [0, 1]',
    sql: 'SELECT * FROM mylib.ciclos_recuperacion\n WHERE PD_ESTIMADA > 1 OR PD_ESTIMADA < 0',
    variable: 'PD_ESTIMADA', condicion_error: 'PD_ESTIMADA fuera de [0, 1]',
    campos_entrada: ['PD_ESTIMADA'], referencia_regulatoria: 'EBA GL/2017/16 §63',
    umbral: '0 ≤ PD ≤ 1',
    justificacion: 'La PD es una probabilidad; valores fuera de [0,1] invalidan el cálculo de ECL.',
  }),
  check({
    check_id: 'chk_ead', rule_id: 'DQC_EAD_002', name: 'dqc_ead_total_001',
    description: 'El EAD total no puede ser negativo',
    sql: 'SELECT * FROM mylib.ciclos_recuperacion\n WHERE EAD_TOTAL < 0',
    variable: 'EAD_TOTAL', condicion_error: 'EAD_TOTAL < 0',
    campos_entrada: ['EAD_TOTAL'], referencia_regulatoria: 'Sin referencia en diccionario',
    umbral: 'EAD ≥ 0',
    justificacion: 'Una exposición negativa no tiene sentido económico y distorsiona la ECL agregada.',
  }),
  check({
    check_id: 'chk_ecl', rule_id: 'DQC_ECL_003', name: 'dqc_ecl_001',
    description: 'ECL debe reperformar como PD × LGD × EAD', category: 'formula',
    tipo: 'formula',
    sql: 'SELECT * FROM mylib.ciclos_recuperacion\n WHERE ABS(ECL - PD_ESTIMADA * LGD_ESTIMADA * EAD_TOTAL) > 0.01',
    variable: 'ECL', condicion_error: 'ECL difiere de la fórmula documentada',
    campos_entrada: ['ECL', 'PD_ESTIMADA', 'LGD_ESTIMADA', 'EAD_TOTAL'],
    umbral: 'tolerancia 0.01', status: 'validated',
    justificacion: 'La fórmula está documentada en el diccionario; cualquier desviación es un error de cálculo.',
  }),
  check({
    check_id: 'chk_lgd', name: 'dqc_lgd_estimada_001', severity: 'MED',
    description: 'La LGD estimada debe estar en el rango [0, 1]',
    sql: 'SELECT * FROM mylib.ciclos_recuperacion\n WHERE LGD_ESTIMADA > 1 OR LGD_ESTIMADA < 0',
    variable: 'LGD_ESTIMADA', condicion_error: 'LGD_ESTIMADA fuera de [0, 1]',
    campos_entrada: ['LGD_ESTIMADA'], umbral: '0 ≤ LGD ≤ 1',
    justificacion: 'La LGD es un porcentaje de pérdida; fuera de [0,1] no es interpretable.',
  }),
];

const CASES: Record<string, CheckCasesResponse> = {
  chk_pd: { available: true, evaluated_at: '2026-07-21T09:14:00Z', n_casos: 2,
            columnas: COLS, ejemplos: ROWS['pd'], precision: 1, recall: 1,
            esperados: 2, trace: TRACE_OK as any },
  chk_ead: { available: true, evaluated_at: '2026-07-21T09:14:00Z', n_casos: 2,
             columnas: COLS, ejemplos: ROWS['ead'], precision: 1, recall: 1,
             esperados: 2, trace: TRACE_CORRECTED as any },
  chk_ecl: { available: true, evaluated_at: '2026-07-21T09:14:00Z', n_casos: 1,
             columnas: COLS, ejemplos: ROWS['ecl'], precision: 1, recall: 1,
             esperados: 1, trace: TRACE_OK as any },
  chk_lgd: { available: true, evaluated_at: '2026-07-21T09:14:00Z', n_casos: 1,
             columnas: COLS, ejemplos: ROWS['lgd'], trace: TRACE_OK as any },
};

/** Mutable copy so validar/rechazar work during the demo. Each check is
 * tagged with the project it was generated in, so switching projects
 * shows a genuinely separate set (empty until you generate there). */
type DemoCheck = CheckRecord & { project_id?: string };
let store: DemoCheck[] = SEED.map((c) => ({ ...c }));

function scoped(projectId?: string): DemoCheck[] {
  return projectId ? store.filter((c) => c.project_id === projectId) : store;
}

export class DemoBackend {
  inspect(): Observable<InspectResponse> {
    return of<InspectResponse>({
      sheets: [
        { name: 'Notas', rows: 2, headers: ['Diccionario de campos'], score: 0 },
        { name: 'DICCIONARIO', rows: 6,
          headers: ['Field', 'Type', 'Description', 'Null', 'Formula'], score: 5 },
      ],
      proposed_sheet: 'DICCIONARIO',
      column_mapping: { field: 'Field', type: 'Type', description: 'Description',
                        nullable: 'Null', formula: 'Formula', reg_ref: null },
      confidence: 0.93, source: 'llm', question: null,
      options: ['Notas', 'DICCIONARIO'],
    }).pipe(delay(1400));
  }

  /** Replays a full plan-mode run: plan → per-item phases → done. */
  generateStream(projectId?: string): Observable<StreamEvent> {
    // the run "produces" the seeded checks for this project
    if (projectId) store.forEach((c) => { if (!c.project_id) c.project_id = projectId; });
    const plan = [
      { id: 1, regla: 'La PD estimada debe estar entre 0 y 1', prev_id: 'DQC_PD_001',
        accion: 'Check de rango sobre PD_ESTIMADA (0 ≤ PD ≤ 1)' },
      { id: 2, regla: 'El EAD total no puede ser negativo', prev_id: 'DQC_EAD_002',
        accion: 'Check de rango sobre EAD_TOTAL (≥ 0)' },
      { id: 3, regla: 'Toda operación debe tener el colateral informado', prev_id: '',
        accion: 'Identificar el campo de colateral en el diccionario' },
    ];

    const script: { wait: number; ev: StreamEvent }[] = [
      { wait: 500, ev: { type: 'meta', data: { dictionary_fields: 6, sheet_used: 'DICCIONARIO', formats_inferred: 0, casos: 6 } } },
      { wait: 700, ev: { type: 'plan', data: { items: plan.map((p) => ({ ...p, estado: 'pendiente' })) } } },
      // rule 1 — clean pass
      { wait: 600, ev: { type: 'item', data: { id: 1, estado: 'en_curso', fase: 'suficiencia' } } },
      { wait: 900, ev: { type: 'item', data: { id: 1, estado: 'en_curso', fase: 'generacion', intento: 1 } } },
      { wait: 900, ev: { type: 'item', data: { id: 1, estado: 'en_curso', fase: 'validacion', intento: 1 } } },
      { wait: 700, ev: { type: 'item', data: { id: 1, estado: 'completado', trace: TRACE_OK,
          dqcs: [{ dqc_id: 'DQC_PD_ESTIMADA_001', descripcion: 'La PD estimada debe estar en el rango [0, 1]',
                   severidad: 'bloqueante', condicion_error: 'PD_ESTIMADA fuera de [0, 1]',
                   regla_sql: SEED[0].sql }],
          validacion: { estatica: 'ok', ejecutada: true, n_casos: 2, columnas: COLS,
                        ejemplos: ROWS['pd'], precision: 1, recall: 1, esperados: 2 } } } },
      // rule 2 — judge rejects, corrected on attempt 2
      { wait: 600, ev: { type: 'item', data: { id: 2, estado: 'en_curso', fase: 'suficiencia' } } },
      { wait: 800, ev: { type: 'item', data: { id: 2, estado: 'en_curso', fase: 'generacion', intento: 1 } } },
      { wait: 700, ev: { type: 'item', data: { id: 2, estado: 'en_curso', fase: 'validacion', intento: 1 } } },
      { wait: 700, ev: { type: 'item', data: { id: 2, estado: 'en_curso', fase: 'juicio', intento: 1 } } },
      { wait: 900, ev: { type: 'item', data: { id: 2, estado: 'en_curso', fase: 'generacion', intento: 2 } } },
      { wait: 700, ev: { type: 'item', data: { id: 2, estado: 'en_curso', fase: 'validacion', intento: 2 } } },
      { wait: 700, ev: { type: 'item', data: { id: 2, estado: 'en_curso', fase: 'juicio', intento: 2 } } },
      { wait: 700, ev: { type: 'item', data: { id: 2, estado: 'completado', trace: TRACE_CORRECTED,
          dqcs: [{ dqc_id: 'DQC_EAD_TOTAL_001', descripcion: 'El EAD total no puede ser negativo',
                   severidad: 'bloqueante', condicion_error: 'EAD_TOTAL < 0',
                   regla_sql: SEED[1].sql }],
          validacion: { estatica: 'ok', ejecutada: true, n_casos: 2, columnas: COLS,
                        ejemplos: ROWS['ead'], precision: 1, recall: 1, esperados: 2,
                        juez_ok: true, juez_motivo: 'La consulta selecciona exactamente las filas que violan la regla.' } } } },
      // rule 3 — ambiguous
      { wait: 600, ev: { type: 'item', data: { id: 3, estado: 'en_curso', fase: 'suficiencia' } } },
      { wait: 1000, ev: { type: 'item', data: { id: 3, estado: 'ambigua', campos: [],
          falta: 'El diccionario no contiene ningún campo de colateral o garantías. Indica qué campo lo describe (p.ej. COLATERAL_TIPO) o añádelo.',
          trace: [TRACE_OK[0], { paso: 'resultado', estado: 'ambigua' }] } } },
      { wait: 600, ev: { type: 'done', data: {
          dqcs: [{ dqc_id: 'DQC_PD_ESTIMADA_001' }, { dqc_id: 'DQC_EAD_TOTAL_001' }],
          context_summary: 'Se generaron 2 DQCs a partir de un plan de 3 reglas.',
          dictionary_fields: 6, sheet_used: 'DICCIONARIO', mapping_source: 'llm',
          formats_inferred: 0, agents_used: 9,
          evaluacion: { casos: 6, tests_comprobados: 2, ambiguas: 1,
                        precision_media: 1, recall_medio: 1 } } } },
    ];

    return new Observable<StreamEvent>((observer) => {
      let cancelled = false;
      let i = 0;
      const next = () => {
        if (cancelled || i >= script.length) {
          if (!cancelled) observer.complete();
          return;
        }
        const step = script[i++];
        setTimeout(() => {
          if (cancelled) return;
          observer.next(step.ev);
          next();
        }, step.wait);
      };
      next();
      return () => { cancelled = true; };
    });
  }

  list(status?: string, projectId?: string): Observable<CheckRecord[]> {
    const base = scoped(projectId);
    const rows = status ? base.filter((c) => c.status === status) : base;
    return of(rows.map((c) => ({ ...c }))).pipe(delay(120));
  }

  counts(projectId?: string): Observable<CountsResponse> {
    const rows = scoped(projectId);
    return of({
      pending_visible: rows.filter((c) => c.status === 'pending').length,
      validated: rows.filter((c) => c.status === 'validated').length,
      rejected: rows.filter((c) => c.status === 'rejected').length,
      oculto: 0,
      dashboard_ready: rows.some((c) => c.status === 'validated'),
    }).pipe(delay(80));
  }

  setStatus(checkId: string, status: 'validated' | 'rejected'): Observable<CheckRecord> {
    const c = store.find((x) => x.check_id === checkId);
    if (c) {
      c.status = status;
      c.validated_at = new Date().toISOString();
    }
    return of({ ...(c ?? store[0]) }).pipe(delay(120));
  }

  delete(checkId: string): Observable<unknown> {
    store = store.filter((c) => c.check_id !== checkId);
    return of({}).pipe(delay(120));
  }

  checkCases(checkId: string): Observable<CheckCasesResponse> {
    return of(CASES[checkId] ?? { available: false }).pipe(delay(200));
  }

  evaluate(projectId?: string): Observable<EvaluateResponse> {
    const resultados = scoped(projectId)
      .filter((c) => CASES[c.check_id]?.available)
      .map((c) => {
        const k = CASES[c.check_id];
        return {
          check_id: c.check_id, name: c.name, prev_id: c.rule_id,
          descripcion: c.description, condicion_error: c.condicion_error ?? '',
          ok: true, n_casos: k.n_casos ?? 0, columnas: k.columnas ?? [],
          ejemplos: k.ejemplos ?? [], precision: k.precision ?? null,
          recall: k.recall ?? null, esperados: k.esperados ?? null,
        };
      });
    return of({
      casos: 6, resultados, evaluados: resultados.length, fallidos: 0,
      precision_media: 1, recall_medio: 1,
    }).pipe(delay(900));
  }

  dashboard(): Observable<DashboardResponse> {
    const validated = store.filter((c) => c.status === 'validated');
    return of({
      ready: validated.length > 0,
      pending_visible: store.filter((c) => c.status === 'pending').length,
      validated: validated.length,
      rejected: store.filter((c) => c.status === 'rejected').length,
      oculto: 0,
      sql: validated.map((c) => c.sql).join('\n\nUNION ALL\n\n') || null,
      checks: validated.map((c) => ({ ...c })),
    }).pipe(delay(150));
  }
}
