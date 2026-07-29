export interface DqcItem {
  dqc_id: string;
  variable: string;
  descripcion: string;
  tipo: string;
  severidad: string;
  regla_sql: string;
  condicion_error: string;
  campos_entrada: string[];
  referencia_regulatoria: string;
  umbral: string;
  periodicidad: string;
  justificacion: string;
}

export interface GenerateResponse {
  dqcs: DqcItem[];
  dictionary_fields: number;
  context_summary: string;
  sheet_used?: string;
  mapping_source?: string;
  formats_inferred?: number;
  agents_used?: number;
}

export interface SheetSummary {
  name: string;
  rows: number;
  headers: string[];
  score: number;
}

export interface InspectResponse {
  sheets: SheetSummary[];
  proposed_sheet: string | null;
  column_mapping: Record<string, string | null>;
  confidence: number;
  source: string;
  question: string | null;
  options: string[];
}

/** query validation outcome for one plan item (ReAct pipeline) */
export interface PlanValidation {
  estatica: 'ok' | 'error';
  ejecutada: boolean;
  n_casos?: number;
  columnas?: string[];
  ejemplos?: Record<string, string>[];
  precision?: number;
  recall?: number;
  esperados?: number;
  aciertos?: number;
  /** semantic judge verdict (experimental flag) */
  juez_ok?: boolean;
  juez_motivo?: string;
  juez_confianza?: number;
}

/** one node of the per-DQC decision tree (root on top, animated live) */
export interface TreeNode {
  label: string;
  kind: 'decision' | 'action' | 'leaf';
  estado: 'activo' | 'hecho' | 'ok' | 'fallo';
  detail?: string;
  /** which branch the pipeline took on a decision node */
  taken?: 'yes' | 'no';
  yes?: TreeNode;
  no?: TreeNode;
  /** unary chain for action nodes */
  next?: TreeNode;
}

/** one server-side decision-trace step (persisted for audit) */
export interface TraceStep {
  paso: 'suficiencia' | 'generacion' | 'validacion' | 'juicio' | 'resultado' | string;
  pregunta?: string;
  resultado?: 'si' | 'no';
  detalle?: string;
  intento?: number;
  estado?: string;
  accion?: string;
  n_casos?: number | null;
}

/** one entry of the LLM's DQC action plan, ticked off live as it executes */
export interface PlanItem {
  id: number;
  regla: string;
  accion: string;
  prev_id?: string;
  estado: 'pendiente' | 'en_curso' | 'completado' | 'error' | 'ambigua';
  /** ReAct phase while en_curso: suficiencia | generacion | validacion */
  fase?: string;
  intento?: number;
  /** justification of the missing info when estado === 'ambigua' */
  falta?: string;
  campos?: string[];
  validacion?: PlanValidation;
  dqcs?: DqcItem[];
  error?: string;
  /** client-side: case-details table expanded */
  expanded?: boolean;
  /** live decision tree (built from the stream events) */
  tree?: TreeNode;
  treeOpen?: boolean;
  /** server-side decision trace, delivered with the terminal event */
  trace?: TraceStep[];
  /** internal tree-builder cursors */
  _pending?: TreeNode;
  _tail?: TreeNode;
}

/** typed SSE frame from POST /generate_stream */
export interface StreamEvent {
  type: 'meta' | 'plan' | 'item' | 'done' | string;
  data: any;
}

/** one stored DQC executed against the cases Excel (POST /evaluate) */
export interface EvalResult {
  check_id: string;
  name: string;
  prev_id?: string | null;
  descripcion?: string;
  condicion_error?: string;
  ok: boolean;
  error?: string;
  n_casos: number;
  columnas: string[];
  ejemplos: Record<string, string>[];
  precision?: number | null;
  recall?: number | null;
  esperados?: number | null;
  /** client-side: case-details table expanded */
  expanded?: boolean;
}

/** GET /checks/{id}/cases — last detected cases for a stored check */
export interface CheckCasesResponse {
  available: boolean;
  evaluated_at?: string;
  n_casos?: number;
  columnas?: string[];
  ejemplos?: Record<string, string>[];
  precision?: number | null;
  recall?: number | null;
  esperados?: number | null;
  /** persisted ReAct decision trace, shown before the SQL when editing */
  trace?: TraceStep[];
}

export interface EvaluateResponse {
  casos: number;
  resultados: EvalResult[];
  evaluados: number;
  fallidos: number;
  precision_media?: number | null;
  recall_medio?: number | null;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  dqcs?: DqcItem[];
  /** live DQC action plan rendered as a checklist */
  plan?: PlanItem[];
  /** results of evaluating stored DQCs against a cases Excel */
  evalResults?: EvalResult[];
  /** sheet names rendered as clickable answers to an assistant question */
  options?: string[];
  /** sheet the LLM proposed after inspection (highlighted among options) */
  proposedOption?: string;
  /** option the user clicked; freezes the buttons once chosen */
  selectedOption?: string;
}

// ── Validation pipeline models (mirror api/routers/dqc.py) ─────────────

export interface CheckRecord {
  check_id: string;
  rule_id: string | null;
  name: string;
  description: string;
  severity: string;
  category: string;
  sql: string;
  visible: boolean;
  status: 'pending' | 'validated' | 'rejected';
  reward: number | null;
  variable: string | null;
  tipo: string | null;
  condicion_error: string | null;
  campos_entrada: string[];
  referencia_regulatoria: string | null;
  umbral: string | null;
  periodicidad: string | null;
  justificacion: string | null;
  created_at: string | null;
  validated_at: string | null;
}

export interface CountsResponse {
  pending_visible: number;
  validated: number;
  rejected: number;
  oculto: number;
  dashboard_ready: boolean;
}

export interface DashboardResponse {
  ready: boolean;
  pending_visible: number;
  validated: number;
  rejected: number;
  oculto: number;
  sql: string | null;
  checks: CheckRecord[];
}
