import { Component, EventEmitter, Input, OnDestroy, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DqcService } from '../services/dqc.service';
import { CheckCasesResponse, CheckRecord, TraceStep } from '../models/dqc.model';

/** Layer 2b — review and edit the DQCs of a project.
 * The detail panel is ordered for reading: first HOW the DQC was decided
 * (the ReAct trace), then the SQL, then the cases it detects. */
@Component({
  selector: 'app-editor',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="editor">
      <aside class="list-pane">
        <div class="filter-row">
          @for (f of filters; track f.key) {
            <button class="filter-btn" [class.active]="filter === f.key"
                    (click)="filter = f.key">{{ f.label }}</button>
          }
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
              @if (c.variable) { <code class="dqc-item-var">{{ c.variable }}</code> }
            </div>
          } @empty {
            <div class="dqc-empty">
              Sin chequeos{{ filter !== 'all' ? ' en este estado' : '' }}.
              Cambia a la capa «Generar» para crear nuevos.
            </div>
          }
        </div>
      </aside>

      @if (selected) {
        <section class="detail">
          <div class="detail-toolbar">
            <button class="detail-close" (click)="close()">&larr; Cerrar</button>
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
              </div>
            }
          </div>

          <div class="detail-body">
            <div class="detail-head">
              <h3>{{ selected.name }}</h3>
              <div class="detail-badges">
                <span class="badge" [class]="sevClass(selected.severity)">{{ selected.severity }}</span>
                <span class="badge badge-cat">{{ selected.category }}</span>
                @if (selected.variable) { <code class="badge-var">{{ selected.variable }}</code> }
              </div>
            </div>
            @if (selected.description) { <p class="detail-desc">{{ selected.description }}</p> }

            <!-- 1 ─ how it was decided -->
            <h4>Cómo se decidió</h4>
            @if (trace.length) {
              <ol class="trace">
                @for (s of trace; track $index) {
                  <li [attr.data-res]="s.resultado || s.estado || 'accion'">
                    <span class="t-mark">
                      @switch (s.resultado || s.estado) {
                        @case ('si') { ✓ } @case ('no') { ✗ }
                        @case ('completado') { ✓ } @case ('ambigua') { ! }
                        @case ('error') { ✗ } @default { • }
                      }
                    </span>
                    <span class="t-body">
                      <strong>{{ s.pregunta || s.accion || stepLabel(s) }}</strong>
                      @if (s.intento && s.intento > 1) {
                        <em class="t-try">intento {{ s.intento }}</em>
                      }
                      @if (s.detalle) { <span class="t-detail">{{ s.detalle }}</span> }
                      @if (s.n_casos != null) {
                        <span class="t-detail">{{ s.n_casos }} caso(s) detectado(s)</span>
                      }
                    </span>
                  </li>
                }
              </ol>
            } @else {
              <p class="muted">
                Sin traza registrada — este DQC se generó antes de que se
                guardara el árbol de decisión, o con el endpoint clásico.
              </p>
            }

            <!-- 2 ─ the query itself -->
            <h4>Consulta SQL</h4>
            <pre class="sql-block">{{ selected.sql }}</pre>

            <!-- 3 ─ what it catches -->
            @if (cases?.available) {
              <h4>
                Casos detectados ({{ cases!.n_casos }})
                @if (cases!.precision != null) {
                  <span class="cases-pr">
                    P {{ (cases!.precision! * 100).toFixed(0) }}% ·
                    R {{ (cases!.recall! * 100).toFixed(0) }}%
                  </span>
                }
              </h4>
              @if (cases!.ejemplos?.length) {
                <p class="cases-hint">{{ caseHint() }}</p>
                <div class="cases-table">
                  <table>
                    <thead><tr>
                      @for (c of cases!.columnas; track c) { <th>{{ c }}</th> }
                    </tr></thead>
                    <tbody>
                      @for (row of cases!.ejemplos; track $index) {
                        <tr>@for (c of cases!.columnas; track c) { <td>{{ row[c] }}</td> }</tr>
                      }
                    </tbody>
                  </table>
                </div>
                <p class="cases-meta">Última evaluación: {{ cases!.evaluated_at }}</p>
              }
            } @else if (cases) {
              <p class="muted">Sin casos registrados — ejecuta el Excel de datos desde la capa «Generar».</p>
            }

            <div class="detail-meta">
              @if (selected.condicion_error) {
                <div class="meta-row"><strong>Error si:</strong> {{ selected.condicion_error }}</div>
              }
              @if (selected.referencia_regulatoria) {
                <div class="meta-row"><strong>Referencia:</strong> {{ selected.referencia_regulatoria }}</div>
              }
              @if (selected.umbral) {
                <div class="meta-row"><strong>Umbral:</strong> {{ selected.umbral }}</div>
              }
              @if (selected.justificacion) {
                <div class="meta-row justification">{{ selected.justificacion }}</div>
              }
            </div>
          </div>
        </section>
      } @else {
        <section class="placeholder">
          <p>Selecciona un DQC para ver su traza de decisión, la consulta y los casos que detecta.</p>
        </section>
      }
    </div>
  `,
  styles: [`
    :host { display: block; height: 100%; }
    .editor { display: flex; height: 100%; background: #0f0f1a; color: #ccc; }
    .list-pane { width: 320px; min-width: 320px; display: flex; flex-direction: column;
                 background: #161625; border-right: 1px solid #2a2a40; }
    .filter-row { display: flex; padding: 10px 12px; border-bottom: 1px solid #2a2a40; }
    .filter-btn { flex: 1; background: transparent; border: 1px solid #2a2a40; color: #888;
                  font-size: 11px; padding: 5px 0; cursor: pointer; }
    .filter-btn:first-child { border-radius: 4px 0 0 4px; }
    .filter-btn:last-child { border-radius: 0 4px 4px 0; }
    .filter-btn:not(:last-child) { border-right: none; }
    .filter-btn.active { background: #6c7bbf; color: #fff; border-color: #6c7bbf; }
    .copy-all-btn { display: block; width: calc(100% - 24px); margin: 8px 12px 0; padding: 8px;
                    background: transparent; color: #6c7bbf; border: 1px solid #2a2a40;
                    border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; }
    .copy-all-btn:disabled { color: #4caf50; border-color: #2e7d32; }
    .dqc-list { flex: 1; overflow-y: auto; padding: 8px; }
    .dqc-item { padding: 10px 12px; border-radius: 6px; cursor: pointer;
                border-left: 3px solid transparent; margin-bottom: 4px; }
    .dqc-item:hover, .dqc-item.selected { background: #1e1e35; }
    .dqc-item[data-status="pending"] { border-left-color: #f9a825; }
    .dqc-item[data-status="validated"] { border-left-color: #2e7d32; }
    .dqc-item[data-status="rejected"] { border-left-color: #c62828; }
    .dqc-item-head { display: flex; align-items: center; justify-content: space-between; }
    .dqc-name { font-size: 12px; font-weight: 600; color: #e0e0e0; }
    .sev-dot { width: 8px; height: 8px; border-radius: 50%; }
    .sev-HIGH { background: #c62828; } .sev-MED { background: #f9a825; } .sev-LOW { background: #6c7bbf; }
    .dqc-item-desc { margin: 4px 0 0; font-size: 11px; color: #888; }
    .dqc-item-var { font-size: 10px; color: #e65100; background: rgba(230,81,0,.1);
                    padding: 1px 6px; border-radius: 3px; margin-top: 4px; display: inline-block; }
    .dqc-empty { text-align: center; color: #666; font-size: 13px; padding: 32px 16px; }
    .detail, .placeholder { flex: 1; min-width: 0; display: flex; flex-direction: column; background: #12121f; }
    .placeholder { align-items: center; justify-content: center; color: #555; font-size: 13px; padding: 24px; text-align: center; }
    .detail-toolbar { display: flex; align-items: center; justify-content: space-between;
                      padding: 10px 16px; border-bottom: 1px solid #2a2a40; }
    .detail-close { background: transparent; border: none; color: #6c7bbf; cursor: pointer; }
    .detail-actions { display: flex; gap: 8px; align-items: center; }
    .btn-validate { background: #2e7d32; color: #fff; border: none; padding: 6px 16px; border-radius: 4px; cursor: pointer; }
    .btn-reject { background: transparent; color: #c62828; border: 1px solid #c62828; padding: 6px 16px; border-radius: 4px; cursor: pointer; }
    .status-label { font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 4px; }
    .status-label[data-status="validated"] { background: rgba(46,125,50,.15); color: #4caf50; }
    .status-label[data-status="rejected"] { background: rgba(198,40,40,.15); color: #ef5350; }
    .detail-body { flex: 1; overflow-y: auto; padding: 16px 20px 32px; max-width: 860px; }
    .detail-head h3 { margin: 0 0 8px; font-size: 16px; color: #fff; }
    .detail-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
    .badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: 600; text-transform: uppercase; }
    .sev-high { background: #c62828; color: #fff; } .sev-med { background: #f9a825; color: #000; }
    .sev-low { background: #6c7bbf; color: #fff; }
    .badge-cat { background: #2a2a40; color: #aaa; }
    .badge-var { font-size: 11px; background: rgba(230,81,0,.15); color: #ff9800; padding: 2px 8px; border-radius: 10px; }
    .detail-desc { font-size: 14px; color: #bbb; margin: 0 0 18px; line-height: 1.5; }
    .detail-body h4 { font-size: 11px; text-transform: uppercase; letter-spacing: .5px;
                      color: #666; margin: 20px 0 8px; }
    .muted { font-size: 12px; color: #666; margin: 0 0 8px; }
    /* trace — the "why", read first */
    .trace { list-style: none; margin: 0 0 4px; padding: 0; }
    .trace li { display: flex; gap: 10px; padding: 7px 10px; border-left: 2px solid #2a2a40;
                margin-left: 6px; }
    .trace li[data-res="si"], .trace li[data-res="completado"] { border-left-color: #2e7d32; }
    .trace li[data-res="no"], .trace li[data-res="error"] { border-left-color: #c62828; }
    .trace li[data-res="ambigua"] { border-left-color: #f39c12; }
    .t-mark { width: 14px; flex-shrink: 0; text-align: center; font-weight: 700; font-size: 12px; color: #666; }
    .trace li[data-res="si"] .t-mark, .trace li[data-res="completado"] .t-mark { color: #2ecc71; }
    .trace li[data-res="no"] .t-mark, .trace li[data-res="error"] .t-mark { color: #e74c3c; }
    .trace li[data-res="ambigua"] .t-mark { color: #f39c12; }
    .t-body { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
    .t-body strong { font-size: 12px; color: #ccc; font-weight: 600; }
    .t-try { font-size: 10px; color: #6c7bbf; font-style: normal; }
    .t-detail { font-size: 11px; color: #888; }
    .sql-block { background: #0a0a14; color: #7ec8e3; padding: 12px; border-radius: 6px;
                 font-family: 'SF Mono', Monaco, Menlo, monospace; font-size: 12px;
                 overflow-x: auto; white-space: pre-wrap; margin: 0; border: 1px solid #2a2a40; }
    .cases-pr { font-size: 11px; color: #4caf50; text-transform: none; margin-left: 8px; }
    .cases-hint { font-size: 12px; color: #8a93c4; margin: 0 0 6px; }
    .cases-table { overflow: auto; border: 1px solid #2a2a40; border-radius: 6px; max-height: 240px; }
    .cases-table table { border-collapse: collapse; font-size: 11px; width: 100%; }
    .cases-table th, .cases-table td { padding: 4px 10px; text-align: left; white-space: nowrap;
                                       border-bottom: 1px solid #1e1e35; }
    .cases-table th { color: #888; background: #0a0a14; position: sticky; top: 0; }
    .cases-table td { color: #aaa; }
    .cases-meta { font-size: 11px; color: #666; margin: 6px 0 0; }
    .detail-meta { display: flex; flex-direction: column; gap: 8px; margin-top: 20px;
                   border-top: 1px solid #2a2a40; padding-top: 14px; }
    .meta-row { font-size: 12px; color: #999; }
    .meta-row strong { color: #ccc; }
    .justification { font-style: italic; color: #888; }
  `],
})
export class EditorComponent implements OnInit, OnDestroy {
  @Input() refreshToken = 0;
  @Output() checksChanged = new EventEmitter<void>();

  readonly filters = [
    { key: 'all', label: 'Todos' }, { key: 'pending', label: 'Pendientes' },
    { key: 'validated', label: 'Validados' }, { key: 'rejected', label: 'Rechazados' },
  ] as const;

  checks: CheckRecord[] = [];
  selected: CheckRecord | null = null;
  cases: CheckCasesResponse | null = null;
  trace: TraceStep[] = [];
  filter: 'all' | 'pending' | 'validated' | 'rejected' = 'all';
  copiedAll = false;
  copyAllMsg = '';
  private timer: ReturnType<typeof setInterval> | null = null;

  constructor(private dqc: DqcService) {}

  ngOnInit(): void {
    this.load();
    this.timer = setInterval(() => this.load(), 5000);
  }

  ngOnDestroy(): void {
    if (this.timer) clearInterval(this.timer);
  }

  get filteredChecks(): CheckRecord[] {
    return this.filter === 'all'
      ? this.checks : this.checks.filter((c) => c.status === this.filter);
  }

  select(c: CheckRecord): void {
    if (this.selected?.check_id === c.check_id) return this.close();
    this.selected = c;
    this.cases = null;
    this.trace = [];
    this.dqc.checkCases(c.check_id).subscribe({
      next: (res) => {
        if (this.selected?.check_id !== c.check_id) return;
        this.cases = res;
        this.trace = res.trace ?? [];
      },
      error: () => { /* detail is best-effort */ },
    });
  }

  close(): void {
    this.selected = null;
    this.cases = null;
    this.trace = [];
  }

  stepLabel(s: TraceStep): string {
    const labels: Record<string, string> = {
      suficiencia: 'Verificación de información',
      generacion: 'Generación de la consulta',
      validacion: 'Validación de la consulta',
      juicio: 'Revisión semántica',
      resultado: 'Resultado',
    };
    return labels[s.paso] ?? s.paso;
  }

  sevClass(sev: string): string {
    return sev === 'HIGH' ? 'sev-high' : sev === 'MED' ? 'sev-med' : 'sev-low';
  }

  caseHint(): string {
    const c = (this.selected?.condicion_error || '').trim();
    const d = (this.selected?.description || '').trim();
    if (c) return `Estas filas se marcan porque cumplen la condición de error: ${c}`;
    if (d) return `Filas que incumplen la regla: ${d}`;
    return 'Filas que la consulta marca como incumplimiento de la regla.';
  }

  setStatus(c: CheckRecord, status: 'validated' | 'rejected'): void {
    this.dqc.setStatus(c.check_id, status).subscribe({
      next: (updated) => {
        this.selected = updated;
        this.load();
        this.checksChanged.emit();
      },
    });
  }

  copyAll(): void {
    this.dqc.dashboard().subscribe({
      next: (d) => {
        const parts: string[] = [];
        if (d.checks.length) parts.push(d.checks.map((c) => c.sql).join('\n\n;\n\n'));
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

  load(): void {
    this.dqc.list().subscribe({ next: (cs) => { this.checks = cs; } });
  }
}
