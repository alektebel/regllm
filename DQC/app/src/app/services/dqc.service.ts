import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  DqcResponse, CheckRecord, CountsResponse, DashboardResponse, TestsResponse,
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

  generateTests(tests: string[], sessionId = 'default'): Observable<TestsResponse> {
    return this.http.post<TestsResponse>(`${this.apiUrl}/generate/tests`, {
      tests,
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
