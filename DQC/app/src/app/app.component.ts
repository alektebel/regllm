import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatComponent } from './chat/chat.component';
import { DqcService } from './services/dqc.service';
import { CheckCasesResponse, CheckRecord, TraceStep } from './models/dqc.model';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, ChatComponent],
  template: `
    <div class="shell">
      <aside class="sidebar">
        <div class="sidebar-header">
          <h2>DQCs</h2>
          <div class="sidebar-counts">
            <span class="cnt cnt-pending">{{ pending.length }}</span>
            <span class="cnt cnt-validated">{{ validated.length }}</span>
            <span class="cnt cnt-rejected">{{ rejected.length }}</span>
          </div>
        </div>

        <div class="filter-row">
          <button class="filter-btn" [class.active]="filter === 'all'"
                  (click)="filter = 'all'">Todos</button>
          <button class="filter-btn" [class.active]="filter === 'pending'"
                  (click)="filter = 'pending'">Pendientes</button>
          <button class="filter-btn" [class.active]="filter === 'validated'"
                  (click)="filter = 'validated'">Validados</button>
          <button class="filter-btn" [class.active]="filter === 'rejected'"
                  (click)="filter = 'rejected'">Rechazados</button>
        </div>

        <button class="copy-all-btn" (click)="copyAll()" [disabled]="copiedAll">
          {{ copiedAll ? '✓ Copiado' : (copyAllMsg || 'Copiar todo (consultas + dashboard)') }}
        </button>

        <div class="dqc-list">
          @for (c of filteredChecks; track c.check_id) {
            <div class="dqc-item" [class.selected]="selected?.check_id === c.check_id"
                 [attr.data-status]="c.status" (click)="select(c)">
              <div class="dqc-item-head">
                <code class="dqc-name">{{ c.name }}</code>
                <span class="sev-dot" [class]="'sev-' + c.severity"></span>
              </div>
              <p class="dqc-item-desc">{{ c.description }}</p>
              @if (c.variable) {
                <code class="dqc-item-var">{{ c.variable }}</code>
              }
            </div>
          } @empty {
            <div class="dqc-empty">
              Sin chequeos{{ filter !== 'all' ? ' en estado ' + filter : '' }}.
              Genera nuevos desde el panel derecho.
            </div>
          }
        </div>
      </aside>

      @if (selected) {
        <section class="detail">
          <div class="detail-toolbar">
            <button class="detail-close" (click)="selected = null">&larr; Cerrar</button>
            @if (selected.status === 'pending') {
              <div class="detail-actions">
                <button class="btn-validate" (click)="setStatus(selected, 'validated')">Validar</button>
                <button class="btn-reject" (click)="setStatus(selected, 'rejected')">Rechazar</button>
              </div>
            } @else {
              <div class="detail-actions">
                <span class="status-label" [attr.data-status]="selected.status">
                  {{ selected.status === 'validated' ? 'Validado' : 'Rechazado' }}
                </span>
                @if (selected.status === 'validated') {
                  <button class="btn-reject" (click)="setStatus(selected, 'rejected')">Invalidar</button>
                }
                @if (selected.status === 'rejected') {
                  <span class="terminal-note">El rechazo es definitivo.</span>
                }
              </div>
            }
          </div>

          <div class="detail-body">
            <div class="detail-head">
              <h3>{{ selected.name }}</h3>
              <div class="detail-badges">
                <span class="badge" [class]="sevClass(selected.severity)">{{ selected.severity }}</span>
                <span class="badge badge-cat">{{ selected.category }}</span>
                @if (selected.variable) {
                  <code class="badge-var">{{ selected.variable }}</code>
                }
              </div>
            </div>

            @if (selected.description) {
              <p class="detail-desc">{{ selected.description }}</p>
            }

            @if (selected.status === 'rejected') {
              <div class="retry-panel">
                <label for="retry-instructions">Reformular para crear un DQC nuevo</label>
                <textarea id="retry-instructions" [(ngModel)]="retryInstructions"
                          placeholder="Describe la regla con otras palabras"></textarea>
                <button class="btn-retry" [disabled]="!retryInstructions.trim()"
                        (click)="retryRejected()">Reintentar</button>
              </div>
            }

            <h4>Consulta SQL</h4>
            <pre class="sql-block">{{ selected.sql }}</pre>

            @if (selectedCases?.trace && selectedCases!.trace!.length > 0) {
              <h4>Árbol de decisión</h4>
              <div class="trace-tree">
                @for (step of selectedCases!.trace!; track $index) {
                  <div class="trace-step" [class]="traceClass(step)">
                    <span class="trace-node">{{ traceLabel(step) }}
                      @if (traceOutcome(step)) {
                        <span class="trace-badge">{{ traceOutcome(step) }}</span>
                      }
                    </span>
                    @if (step.detalle) { <div class="trace-detail">{{ step.detalle }}</div> }
                  </div>
                }
              </div>
            }

            @if (selectedCases?.available) {
              <h4>
                Casos detectados ({{ selectedCases!.n_casos }})
                @if (selectedCases!.precision != null) {
                  <span class="cases-pr">
                    P {{ (selectedCases!.precision! * 100).toFixed(0) }}% ·
                    R {{ (selectedCases!.recall! * 100).toFixed(0) }}%
                    ({{ selectedCases!.esperados }} esperados)
                  </span>
                }
              </h4>
              @if (selectedCases!.ejemplos && selectedCases!.ejemplos!.length > 0) {
                <p class="cases-hint">{{ caseHint() }}</p>
                <div class="cases-table">
                  <table>
                    <thead>
                      <tr>
                        @for (c of selectedCases!.columnas; track c) { <th>{{ c }}</th> }
                      </tr>
                    </thead>
                    <tbody>
                      @for (row of selectedCases!.ejemplos; track $index) {
                        <tr>
                          @for (c of selectedCases!.columnas; track c) { <td>{{ row[c] }}</td> }
                        </tr>
                      }
                    </tbody>
                  </table>
                </div>
                <p class="cases-meta">
                  @if ((selectedCases!.n_casos ?? 0) > selectedCases!.ejemplos!.length) {
                    Mostrando {{ selectedCases!.ejemplos!.length }} de {{ selectedCases!.n_casos }} casos.
                  }
                  Última evaluación: {{ selectedCases!.evaluated_at }}
                </p>
              }
            } @else if (selectedCases && !selectedCases.available) {
              <p class="cases-meta">
                Sin casos registrados — este DQC aún no se ha ejecutado sobre
                un Excel de datos (usa "Evaluar Excel de datos" en el chat).
              </p>
            }

            <div class="detail-meta">
              @if (selected.condicion_error) {
                <div class="meta-row">
                  <strong>Error si:</strong> {{ selected.condicion_error }}
                </div>
              }
              @if (selected.referencia_regulatoria) {
                <div class="meta-row">
                  <strong>Referencia:</strong> {{ selected.referencia_regulatoria }}
                </div>
              }
              @if (selected.umbral) {
                <div class="meta-row">
                  <strong>Umbral:</strong> {{ selected.umbral }}
                </div>
              }
              @if (selected.periodicidad) {
                <div class="meta-row">
                  <strong>Periodicidad:</strong> {{ selected.periodicidad }}
                </div>
              }
              @if (selected.campos_entrada && selected.campos_entrada.length) {
                <div class="meta-row">
                  <strong>Campos:</strong>
                  @for (f of selected.campos_entrada; track f) {
                    <code class="field-tag">{{ f }}</code>
                  }
                </div>
              }
              @if (selected.justificacion) {
                <div class="meta-row justification">
                  <strong>Justificacion:</strong> {{ selected.justificacion }}
                </div>
              }
            </div>
          </div>
        </section>
      }

      <main class="main" [class.has-detail]="!!selected">
        <app-chat [retryInstructions]="retryRequest" (dqcGenerated)="onDqcGenerated()" />
      </main>
    </div>
  `,
  styles: [`
    :host { display: block; height: 100vh; overflow: hidden; }
    .shell { display: flex; height: 100%; background: #0f0f1a; }
    .sidebar {
      width: 320px; min-width: 320px; display: flex; flex-direction: column;
      background: #161625; border-right: 1px solid #2a2a40; color: #ccc;
    }
    .sidebar-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 16px 16px 12px; border-bottom: 1px solid #2a2a40;
    }
    .sidebar-header h2 { margin: 0; font-size: 16px; color: #fff; font-weight: 600; }
    .sidebar-counts { display: flex; gap: 6px; }
    .cnt { font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 10px; }
    .cnt-pending   { background: #f9a825; color: #000; }
    .cnt-validated { background: #2e7d32; color: #fff; }
    .cnt-rejected  { background: #c62828; color: #fff; }
    .filter-row { display: flex; padding: 8px 12px; border-bottom: 1px solid #2a2a40; }
    .filter-btn {
      flex: 1; background: transparent; border: 1px solid #2a2a40; color: #888;
      font-size: 11px; padding: 5px 0; cursor: pointer;
    }
    .filter-btn:first-child { border-radius: 4px 0 0 4px; }
    .filter-btn:last-child  { border-radius: 0 4px 4px 0; }
    .filter-btn:not(:last-child) { border-right: none; }
    .filter-btn.active { background: #6c7bbf; color: #fff; border-color: #6c7bbf; }
    .copy-all-btn {
      display: block; width: calc(100% - 24px); margin: 8px 12px 0; padding: 8px;
      background: transparent; color: #6c7bbf; border: 1px solid #2a2a40;
      border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer;
    }
    .copy-all-btn:disabled { color: #4caf50; border-color: #2e7d32; }
    .dqc-list { flex: 1; overflow-y: auto; padding: 8px; }
    .dqc-item {
      padding: 10px 12px; border-radius: 6px; cursor: pointer;
      border-left: 3px solid transparent; margin-bottom: 4px;
    }
    .dqc-item:hover, .dqc-item.selected { background: #1e1e35; }
    .dqc-item.selected { border-left-color: #6c7bbf; }
    .dqc-item[data-status="pending"]   { border-left-color: #f9a825; }
    .dqc-item[data-status="validated"] { border-left-color: #2e7d32; }
    .dqc-item[data-status="rejected"]  { border-left-color: #c62828; }
    .dqc-item-head { display: flex; align-items: center; justify-content: space-between; }
    .dqc-name { font-size: 12px; font-weight: 600; color: #e0e0e0; }
    .sev-dot { width: 8px; height: 8px; border-radius: 50%; }
    .sev-HIGH { background: #c62828; }
    .sev-MED  { background: #f9a825; }
    .sev-LOW  { background: #6c7bbf; }
    .dqc-item-desc { margin: 4px 0 0; font-size: 11px; color: #888; }
    .dqc-item-var {
      font-size: 10px; color: #e65100; background: rgba(230,81,0,0.1);
      padding: 1px 6px; border-radius: 3px; margin-top: 4px; display: inline-block;
    }
    .dqc-empty { text-align: center; color: #666; font-size: 13px; padding: 32px 16px; }
    .detail {
      width: 480px; min-width: 480px; display: flex; flex-direction: column;
      background: #12121f; border-right: 1px solid #2a2a40; color: #ccc;
    }
    .detail-toolbar {
      display: flex; align-items: center; justify-content: space-between;
      padding: 10px 16px; border-bottom: 1px solid #2a2a40;
    }
    .detail-close { background: transparent; border: none; color: #6c7bbf; cursor: pointer; }
    .detail-actions { display: flex; gap: 8px; }
    .btn-validate { background: #2e7d32; color: #fff; border: none; padding: 6px 16px; border-radius: 4px; cursor: pointer; }
    .btn-reject { background: transparent; color: #c62828; border: 1px solid #c62828; padding: 6px 16px; border-radius: 4px; cursor: pointer; }
    .btn-delete { background: #c62828; color: #fff; border: none; padding: 6px 16px; border-radius: 4px; cursor: pointer; }
    .btn-retry { background: #6c7bbf; color: #fff; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; }
    .terminal-note { font-size: 12px; color: #ef5350; }
    .retry-panel { display: grid; gap: 8px; margin: 0 0 16px; padding: 12px; border: 1px solid #5d2b35; border-radius: 6px; }
    .retry-panel label { font-size: 12px; color: #ef9a9a; }
    .retry-panel textarea { min-height: 72px; resize: vertical; background: #0a0a14; border: 1px solid #2a2a40; border-radius: 4px; color: #ddd; padding: 8px; }
    .status-label { font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 4px; }
    .status-label[data-status="validated"] { background: rgba(46,125,50,0.15); color: #4caf50; }
    .status-label[data-status="rejected"]  { background: rgba(198,40,40,0.15); color: #ef5350; }
    .detail-body { flex: 1; overflow-y: auto; padding: 16px; }
    .detail-head h3 { margin: 0 0 8px; font-size: 16px; color: #fff; }
    .detail-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
    .badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: 600; text-transform: uppercase; }
    .sev-high { background: #c62828; color: #fff; }
    .sev-med  { background: #f9a825; color: #000; }
    .sev-low  { background: #6c7bbf; color: #fff; }
    .badge-cat { background: #2a2a40; color: #aaa; }
    .badge-var { font-size: 11px; background: rgba(230,81,0,0.15); color: #ff9800; padding: 2px 8px; border-radius: 10px; }
    .detail-desc { font-size: 14px; color: #bbb; margin: 0 0 16px; line-height: 1.5; }
    .detail-body h4 { font-size: 12px; text-transform: uppercase; color: #666; margin: 0 0 6px; }
    .sql-block {
      background: #0a0a14; color: #d4d4d4; padding: 12px; border-radius: 6px;
      font-family: 'SF Mono', Monaco, Menlo, monospace; font-size: 12px;
      overflow-x: auto; white-space: pre-wrap; word-break: break-word;
      margin: 0 0 16px; border: 1px solid #2a2a40;
    }
    .detail-meta { display: flex; flex-direction: column; gap: 8px; }
    .meta-row { font-size: 12px; color: #999; }
    .meta-row strong { color: #ccc; }
    .field-tag { font-size: 11px; background: #1e1e35; padding: 1px 6px; border-radius: 3px; margin-left: 4px; }
    .justification { font-style: italic; color: #888; }
    .cases-pr { font-size: 11px; color: #4caf50; text-transform: none; margin-left: 8px; }
    .cases-table {
      overflow-x: auto; border: 1px solid #2a2a40; border-radius: 6px;
      margin: 0 0 6px; max-height: 260px; overflow-y: auto;
    }
    .cases-table table { border-collapse: collapse; font-size: 11px; width: 100%; }
    .cases-table th, .cases-table td {
      padding: 4px 10px; text-align: left; white-space: nowrap;
      border-bottom: 1px solid #1e1e35;
    }
    .cases-table th { color: #888; background: #0a0a14; font-weight: 600; position: sticky; top: 0; }
    .cases-table td { color: #aaa; }
    .cases-meta { font-size: 11px; color: #666; margin: 0 0 16px; }
    .cases-hint { font-size: 12px; color: #8a93c4; margin: 0 0 6px; }
    .trace-tree { margin: 0 0 18px; padding-left: 6px; }
    .trace-step { position: relative; padding: 0 0 14px 18px; border-left: 2px solid #2a2a3a; }
    .trace-step:last-child { border-left-color: transparent; padding-bottom: 0; }
    .trace-step::before { content: ''; position: absolute; left: -6px; top: 3px;
      width: 10px; height: 10px; border-radius: 50%; background: #555; }
    .trace-step.trace-good::before { background: #4caf50; }
    .trace-step.trace-bad::before  { background: #e5534b; }
    .trace-node { font-size: 12px; color: #cdd3f0; font-weight: 600; }
    .trace-badge { margin-left: 8px; font-size: 10px; font-weight: 700; text-transform: uppercase;
      padding: 1px 6px; border-radius: 8px; background: #2a2a3a; color: #99a; }
    .trace-step.trace-good .trace-badge { background: rgba(76,175,80,.18); color: #7dda80; }
    .trace-step.trace-bad  .trace-badge { background: rgba(229,83,75,.18); color: #f0938c; }
    .trace-detail { font-size: 11px; color: #888; margin-top: 3px; }
    .main { flex: 1; min-width: 0; display: flex; flex-direction: column; }
  `],
})
export class AppComponent implements OnInit, OnDestroy {
  checks: CheckRecord[] = [];
  selected: CheckRecord | null = null;
  selectedCases: CheckCasesResponse | null = null;
  filter: 'all' | 'pending' | 'validated' | 'rejected' = 'all';
  copiedAll = false;
  copyAllMsg = '';
  retryInstructions = '';
  retryRequest = '';
  private refreshTimer: ReturnType<typeof setInterval> | null = null;

  constructor(private dqc: DqcService) {}

  ngOnInit(): void {
    this.loadChecks();
    this.refreshTimer = setInterval(() => this.loadChecks(), 4000);
  }

  ngOnDestroy(): void {
    if (this.refreshTimer) clearInterval(this.refreshTimer);
  }

  get pending()   { return this.checks.filter(c => c.status === 'pending'); }
  get validated() { return this.checks.filter(c => c.status === 'validated'); }
  get rejected()  { return this.checks.filter(c => c.status === 'rejected'); }

  get filteredChecks(): CheckRecord[] {
    if (this.filter === 'all') return this.checks;
    return this.checks.filter(c => c.status === this.filter);
  }

  select(c: CheckRecord): void {
    if (this.selected?.check_id === c.check_id) {
      this.selected = null;
      this.selectedCases = null;
      return;
    }
    this.selected = c;
    this.selectedCases = null;
    this.dqc.checkCases(c.check_id).subscribe({
      next: (cases) => {
        if (this.selected?.check_id === c.check_id) this.selectedCases = cases;
      },
      error: () => { /* cases are optional detail */ },
    });
  }

  sevClass(sev: string): string {
    if (sev === 'HIGH') return 'sev-high';
    if (sev === 'MED')  return 'sev-med';
    return 'sev-low';
  }

  // ── decision-trace rendering (detail-panel tree) ──────────────────────
  traceLabel(s: TraceStep): string {
    if (s.pregunta) return s.pregunta;
    const map: Record<string, string> = {
      suficiencia: '¿Información suficiente?',
      generacion: (s.intento ?? 1) > 1 ? `Generar consulta (intento ${s.intento})` : 'Generar consulta SAS',
      validacion: '¿Consulta válida?',
      juicio: '¿El juez la aprueba?',
      resultado: 'Resultado',
    };
    return map[s.paso] ?? s.paso;
  }

  traceOutcome(s: TraceStep): string {
    if (s.paso === 'resultado') return s.estado ?? '';
    if (s.resultado) return s.resultado === 'si' ? 'Sí' : 'No';
    return '';
  }

  traceClass(s: TraceStep): string {
    const e = (s.estado ?? s.resultado ?? '').toLowerCase();
    if (['ambigua', 'error', 'no', 'fallo'].includes(e)) return 'trace-bad';
    if (['completado', 'si', 'ok'].includes(e)) return 'trace-good';
    return 'trace-neutral';
  }

  setStatus(c: CheckRecord, status: 'validated' | 'rejected'): void {
    this.dqc.setStatus(c.check_id, status).subscribe({
      next: (updated) => { this.selected = updated; this.loadChecks(); },
    });
  }

  retryRejected(): void {
    if (!this.selected || this.selected.status !== 'rejected' || !this.retryInstructions.trim()) return;
    this.retryRequest = this.retryInstructions.trim();
    this.retryInstructions = '';
  }

  deleteCheck(c: CheckRecord): void {
    this.dqc.delete(c.check_id).subscribe({
      next: () => { this.selected = null; this.loadChecks(); },
    });
  }

  onDqcGenerated(): void { this.loadChecks(); }

  /** One-line explanation of why the detected rows were flagged. */
  caseHint(): string {
    const c = (this.selected?.condicion_error || '').trim();
    const d = (this.selected?.description || '').trim();
    if (c) return `Estas filas se marcan porque cumplen la condición de error: ${c}`;
    if (d) return `Filas que incumplen la regla: ${d}`;
    return 'Filas que la consulta marca como incumplimiento de la regla.';
  }

  copyAll(): void {
    this.dqc.dashboard().subscribe({
      next: (d) => {
        const parts: string[] = [];
        if (d.checks.length) parts.push(d.checks.map(c => c.sql).join('\n\n;\n\n'));
        if (d.sql) parts.push(d.sql);
        if (!parts.length) {
          this.copyAllMsg = 'Sin consultas validadas';
          setTimeout(() => { this.copyAllMsg = ''; }, 2000);
          return;
        }
        navigator.clipboard.writeText(parts.join('\n\n;\n\n')).then(() => {
          this.copiedAll = true;
          setTimeout(() => { this.copiedAll = false; }, 2000);
        });
      },
    });
  }

  private loadChecks(): void {
    this.dqc.list().subscribe({ next: (cs) => { this.checks = cs; } });
  }
}
