import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { DqcResponse } from '../models/dqc.model';

@Injectable({ providedIn: 'root' })
export class DqcService {
  private apiUrl = '/api/dqc/generate';

  constructor(private http: HttpClient) {}

  generate(message: string, sessionId = 'default'): Observable<DqcResponse> {
    return this.http.post<DqcResponse>(this.apiUrl, { message, session_id: sessionId });
  }
}
