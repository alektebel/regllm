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

export interface RAGSource {
  document: string;
  heading: string;
  snippet: string;
  source_type: string;
}

export interface DqcResponse {
  variable: string;
  dqcs: DqcItem[];
  context_summary: string;
  sources: RAGSource[];
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  dqcs?: DqcItem[];
  sources?: RAGSource[];
}
