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

  generate(
    dictionary: File,
    instructions: string,
    tableName = 'mylib.ciclos_recuperacion',
    sheet?: string,
    columnMapping?: Record<string, string | null>,
  ): Observable<GenerateResponse> {
    const form = new FormData();
    form.append('dictionary', dictionary);
    form.append('instructions', instructions);
    form.append('table_name', tableName);
    if (sheet) form.append('sheet', sheet);
    if (columnMapping) form.append('column_mapping', JSON.stringify(columnMapping));
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
