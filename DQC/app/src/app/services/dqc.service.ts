import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  GenerateResponse, CheckRecord, CountsResponse, DashboardResponse,
  InspectResponse,
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

  // TODO(streaming-reasoning): add a `generateStream()` counterpart that
  // consumes the future POST /api/dqc/generate_stream SSE endpoint.
  // Angular's HttpClient buffers the whole body, so don't use it here (and
  // EventSource is GET-only): use fetch() with the same FormData, then read
  // `response.body.getReader()` chunk by chunk, split on "\n\n", parse the
  // `event:`/`data:` lines, and push typed events ({type: 'step'|'thinking'|
  // 'answer'|'result', ...}) through an RxJS Subject returned as Observable.
  // Docs: MDN "Server-sent events" + "Streams API".
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
