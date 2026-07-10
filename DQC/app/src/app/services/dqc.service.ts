import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  DqcResponse, SimpleDqcResponse, CheckRecord, CountsResponse, DashboardResponse,
} from '../models/dqc.model';

@Injectable({ providedIn: 'root' })
export class DqcService {
  private apiUrl = '/api/dqc';

  constructor(private http: HttpClient) {}

  generate(message: string, sessionId = 'default'): Observable<DqcResponse> {
    return this.http.post<DqcResponse>(`${this.apiUrl}/generate`, {
      message,
      session_id: sessionId,
    });
  }

  // Natural-language expressions + at most one article, no RAG context —
  // see POST /dqc/generate/simple. Pass either articleParagraph (pull §N
  // from the ingested EBA GL/2017/16 corpus) or articleText (paste it),
  // never both.
  generateSimple(
    expressions: string,
    articleParagraph?: number | null,
    articleText?: string | null,
    sessionId = 'default',
  ): Observable<SimpleDqcResponse> {
    return this.http.post<SimpleDqcResponse>(`${this.apiUrl}/generate/simple`, {
      expressions,
      article_paragraph: articleParagraph ?? null,
      article_text: articleText ?? null,
      session_id: sessionId,
    });
  }

  // ── Validation pipeline ───────────────────────────────────────────────

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

  batchStream(sessionId = 'default'): EventSource {
    return new EventSource(`${this.apiUrl}/generate/batch/stream?session_id=${sessionId}`);
  }

  delete(checkId: string): Observable<unknown> {
    return this.http.delete(`${this.apiUrl}/checks/${checkId}`);
  }

  dashboard(): Observable<DashboardResponse> {
    return this.http.get<DashboardResponse>(`${this.apiUrl}/dashboard`);
  }
}
