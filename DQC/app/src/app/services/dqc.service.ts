import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  GenerateResponse, CheckRecord, CountsResponse, DashboardResponse,
  EvaluateResponse, InspectResponse, StreamEvent,
} from '../models/dqc.model';

@Injectable({ providedIn: 'root' })
export class DqcService {
  private apiUrl = '/api/dqc';

  constructor(private http: HttpClient) {}

  inspect(dictionary: File): Observable<InspectResponse> {
    const form = new FormData();
    form.append('dictionary', dictionary);
    return this.http.post<InspectResponse>(`${this.apiUrl}/inspect_dictionary`, form);
  }

  /** Plan-mode generation over SSE: emits {type: 'meta'|'plan'|'item'|'done'}
   * events as the backend plans the DQCs and executes them one by one.
   * fetch() + ReadableStream because HttpClient buffers the whole body and
   * EventSource is GET-only. HTTP errors reject with {status, error} so the
   * caller can handle 422 needs_sheet_selection like the buffered endpoint. */
  generateStream(
    dictionary: File,
    instructions: string,
    tableName = 'mylib.ciclos_recuperacion',
    sheet?: string,
    columnMapping?: Record<string, string | null>,
    instructionsFile?: File,
    dataFile?: File,
    opts?: { valueGrounding?: boolean; semanticJudge?: boolean },
  ): Observable<StreamEvent> {
    const form = new FormData();
    form.append('dictionary', dictionary);
    form.append('instructions', instructions);
    form.append('table_name', tableName);
    if (sheet) form.append('sheet', sheet);
    if (columnMapping) form.append('column_mapping', JSON.stringify(columnMapping));
    if (instructionsFile) form.append('instructions_file', instructionsFile);
    if (dataFile) form.append('data_file', dataFile);
    if (opts?.valueGrounding) form.append('value_grounding', 'true');
    if (opts?.semanticJudge) form.append('semantic_judge', 'true');

    return new Observable<StreamEvent>((observer) => {
      const controller = new AbortController();
      (async () => {
        const resp = await fetch(`${this.apiUrl}/generate_stream`, {
          method: 'POST', body: form, signal: controller.signal,
        });
        if (!resp.ok || !resp.body) {
          const err: any = new Error(`HTTP ${resp.status}`);
          err.status = resp.status;
          try { err.error = await resp.json(); } catch { /* non-JSON body */ }
          throw err;
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          let sep: number;
          while ((sep = buf.indexOf('\n\n')) >= 0) {
            const frame = buf.slice(0, sep);
            buf = buf.slice(sep + 2);
            let type = 'message';
            const dataLines: string[] = [];
            for (const line of frame.split('\n')) {
              if (line.startsWith('event:')) type = line.slice(6).trim();
              else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
            }
            if (dataLines.length) {
              observer.next({ type, data: JSON.parse(dataLines.join('')) });
            }
          }
        }
        observer.complete();
      })().catch((e) => observer.error(e));
      return () => controller.abort();
    });
  }

  generate(
    dictionary: File,
    instructions: string,
    tableName = 'mylib.ciclos_recuperacion',
    sheet?: string,
    columnMapping?: Record<string, string | null>,
    instructionsFile?: File,
  ): Observable<GenerateResponse> {
    const form = new FormData();
    form.append('dictionary', dictionary);
    form.append('instructions', instructions);
    form.append('table_name', tableName);
    if (sheet) form.append('sheet', sheet);
    if (columnMapping) form.append('column_mapping', JSON.stringify(columnMapping));
    if (instructionsFile) form.append('instructions_file', instructionsFile);
    return this.http.post<GenerateResponse>(`${this.apiUrl}/generate`, form);
  }

  /** Run the stored DQC queries against an extracted-cases Excel (no LLM). */
  evaluate(dataFile: File, tableName = 'mylib.ciclos_recuperacion',
           status?: 'pending' | 'validated' | 'rejected'): Observable<EvaluateResponse> {
    const form = new FormData();
    form.append('data_file', dataFile);
    form.append('table_name', tableName);
    if (status) form.append('status', status);
    return this.http.post<EvaluateResponse>(`${this.apiUrl}/evaluate`, form);
  }

  counts(): Observable<CountsResponse> {
    return this.http.get<CountsResponse>(`${this.apiUrl}/checks/counts`);
  }

  list(status?: 'pending' | 'validated' | 'rejected'): Observable<CheckRecord[]> {
    const q = status ? `?status=${status}` : '';
    return this.http.get<CheckRecord[]>(`${this.apiUrl}/checks${q}`);
  }

  setStatus(checkId: string, status: 'validated' | 'rejected'): Observable<CheckRecord> {
    return this.http.post<CheckRecord>(
      `${this.apiUrl}/checks/${checkId}/status`,
      { status },
    );
  }

  delete(checkId: string): Observable<unknown> {
    return this.http.delete(`${this.apiUrl}/checks/${checkId}`);
  }

  dashboard(): Observable<DashboardResponse> {
    return this.http.get<DashboardResponse>(`${this.apiUrl}/dashboard`);
  }
}
