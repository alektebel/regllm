import { Component, EventEmitter, Input, OnChanges, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CountsResponse } from '../models/dqc.model';
import { Project, ProjectLayer } from '../models/project.model';

/** Layer "Resumen" — the project's landing screen.
 *
 * Job of the data here is a set of single headline magnitudes, so the form is
 * stat tiles + one hero figure (not a chart), with a meter for review
 * progress. Colours are the reserved STATUS palette (good / warning /
 * critical), each paired with an icon and a label so state never reads by
 * colour alone. Below the numbers, a "siguiente paso" card tells the user
 * what to do next and takes them there.
 */
@Component({
  selector: 'app-summary',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="summary">
      <section class="hero">
        <p class="hero-label">Controles creados en este proyecto</p>
        <p class="hero-value">{{ total }}</p>
        <p class="hero-sub">
          @if (total === 0) {
            Todavía no hay ningún DQC — el asistente te guía para crear los primeros.
          } @else {
            {{ validated }} de {{ total }} revisados y validados
          }
        </p>
      </section>

      <section class="tiles">
        <article class="tile" data-state="validated">
          <span class="t-head"><span class="t-icon" aria-hidden="true">✓</span> Validados</span>
          <span class="t-value">{{ validated }}</span>
          <span class="t-note">Listos para el dashboard</span>
        </article>
        <article class="tile" data-state="pending">
          <span class="t-head"><span class="t-icon" aria-hidden="true">◷</span> Pendientes</span>
          <span class="t-value">{{ pending }}</span>
          <span class="t-note">Esperan tu revisión</span>
        </article>
        <article class="tile" data-state="rejected">
          <span class="t-head"><span class="t-icon" aria-hidden="true">✕</span> Rechazados</span>
          <span class="t-value">{{ rejected }}</span>
          <span class="t-note">Descartados en la revisión</span>
        </article>
        <article class="tile" data-state="neutral">
          <span class="t-head"><span class="t-icon" aria-hidden="true">▤</span> Datos del proyecto</span>
          <span class="t-value small">{{ project?.dataFileName ? 'Sí' : 'No' }}</span>
          <span class="t-note">
            {{ project?.dataFileName || 'Sin Excel de casos adjunto' }}
          </span>
        </article>
      </section>

      <section class="progress" *ngIf="total > 0">
        <div class="p-head">
          <span>Progreso de revisión</span>
          <span class="p-pct">{{ reviewedPct }}%</span>
        </div>
        <div class="meter" role="img"
             [attr.aria-label]="'Revisados ' + reviewedPct + ' por ciento'">
          <span class="m-fill validated" [style.width.%]="validatedPct"></span>
          <span class="m-fill rejected" [style.width.%]="rejectedPct"></span>
        </div>
        <p class="p-legend">
          <span><i class="dot validated"></i> Validados {{ validated }}</span>
          <span><i class="dot rejected"></i> Rechazados {{ rejected }}</span>
          <span><i class="dot track"></i> Pendientes {{ pending }}</span>
        </p>
      </section>

      <section class="next">
        <p class="n-step">Siguiente paso</p>
        <h2>{{ guide.title }}</h2>
        <p class="n-body">{{ guide.body }}</p>
        <button class="n-cta" (click)="go.emit(guide.layer)">{{ guide.cta }}</button>
      </section>

      <ol class="steps">
        @for (s of stepList; track s.n) {
          <li [class.done]="s.done" [class.current]="s.current">
            <span class="s-n">{{ s.done ? '✓' : s.n }}</span>
            <span class="s-t"><strong>{{ s.title }}</strong>{{ s.hint }}</span>
          </li>
        }
      </ol>
    </div>
  `,
  styles: [`
    :host { display: block; height: 100%; overflow-y: auto; background: #0f0f1a; color: #ccc; }
    .summary { max-width: 940px; margin: 0 auto; padding: 26px 24px 48px; }

    /* hero figure — exactly one per view */
    .hero { margin-bottom: 22px; }
    .hero-label { margin: 0; font-size: 12px; text-transform: uppercase;
                  letter-spacing: .5px; color: #777; }
    .hero-value { margin: 2px 0 0; font-size: 56px; line-height: 1; font-weight: 600;
                  color: #fff; }
    .hero-sub { margin: 6px 0 0; font-size: 13px; color: #8a93c4; }

    .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
             gap: 12px; margin-bottom: 22px; }
    .tile { display: flex; flex-direction: column; gap: 4px; padding: 14px 16px;
            background: #161625; border: 1px solid #2a2a40; border-radius: 10px;
            border-left-width: 3px; }
    /* reserved status palette; icon + label carry the meaning, colour supports it */
    .tile[data-state="validated"] { border-left-color: #0ca30c; }
    .tile[data-state="pending"]   { border-left-color: #fab219; }
    .tile[data-state="rejected"]  { border-left-color: #d03b3b; }
    .tile[data-state="neutral"]   { border-left-color: #6c7bbf; }
    .t-head { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #999; }
    .t-icon { font-size: 11px; }
    .tile[data-state="validated"] .t-icon { color: #0ca30c; }
    .tile[data-state="pending"]   .t-icon { color: #fab219; }
    .tile[data-state="rejected"]  .t-icon { color: #d03b3b; }
    .tile[data-state="neutral"]   .t-icon { color: #6c7bbf; }
    .t-value { font-size: 30px; font-weight: 600; color: #fff; line-height: 1.1; }
    .t-value.small { font-size: 20px; }
    .t-note { font-size: 11px; color: #666; overflow: hidden;
              text-overflow: ellipsis; white-space: nowrap; }

    .progress { margin-bottom: 22px; }
    .p-head { display: flex; justify-content: space-between; font-size: 12px;
              color: #999; margin-bottom: 6px; }
    .p-pct { color: #fff; font-weight: 600; }
    .meter { display: flex; height: 10px; border-radius: 5px; overflow: hidden;
             background: #2a2a40; }
    .m-fill { height: 100%; }
    .m-fill.validated { background: #0ca30c; }
    .m-fill.rejected  { background: #d03b3b; }
    .m-fill + .m-fill { box-shadow: -2px 0 0 #0f0f1a; }
    .p-legend { display: flex; gap: 16px; margin: 8px 0 0; font-size: 11px; color: #777; }
    .p-legend span { display: flex; align-items: center; gap: 5px; }
    .dot { width: 8px; height: 8px; border-radius: 2px; display: inline-block; }
    .dot.validated { background: #0ca30c; }
    .dot.rejected  { background: #d03b3b; }
    .dot.track     { background: #2a2a40; }

    .next { background: #161625; border: 1px solid #6c7bbf; border-radius: 10px;
            padding: 18px 20px; margin-bottom: 22px; }
    .n-step { margin: 0; font-size: 11px; text-transform: uppercase;
              letter-spacing: .5px; color: #6c7bbf; font-weight: 700; }
    .next h2 { margin: 4px 0 4px; font-size: 17px; color: #fff; font-weight: 600; }
    .n-body { margin: 0 0 14px; font-size: 13px; color: #999; }
    .n-cta { background: #6c7bbf; color: #fff; border: none; border-radius: 7px;
             padding: 9px 20px; font-size: 13px; font-weight: 600; cursor: pointer; }
    .n-cta:hover { background: #5a69a8; }

    .steps { list-style: none; margin: 0; padding: 0; display: flex;
             flex-direction: column; gap: 2px; }
    .steps li { display: flex; gap: 10px; align-items: flex-start; padding: 8px 10px;
                border-radius: 7px; font-size: 12px; color: #666; }
    .steps li.current { background: #161625; color: #ccc; }
    .steps li.done .s-n { background: #0ca30c; color: #fff; border-color: #0ca30c; }
    .steps li.current .s-n { border-color: #6c7bbf; color: #6c7bbf; }
    .s-n { flex-shrink: 0; width: 19px; height: 19px; border-radius: 50%;
           border: 1px solid #2a2a40; display: flex; align-items: center;
           justify-content: center; font-size: 10px; font-weight: 700; }
    .s-t strong { color: inherit; font-weight: 600; margin-right: 6px; }
  `],
})
export class SummaryComponent implements OnChanges {
  @Input() project: Project | null = null;
  @Input() counts: CountsResponse | null = null;
  @Output() go = new EventEmitter<ProjectLayer>();

  get validated(): number { return this.counts?.validated ?? 0; }
  get pending(): number { return this.counts?.pending_visible ?? 0; }
  get rejected(): number { return this.counts?.rejected ?? 0; }
  get total(): number { return this.validated + this.pending + this.rejected; }

  get validatedPct(): number { return this.total ? (this.validated / this.total) * 100 : 0; }
  get rejectedPct(): number { return this.total ? (this.rejected / this.total) * 100 : 0; }
  get reviewedPct(): number {
    return this.total ? Math.round(((this.validated + this.rejected) / this.total) * 100) : 0;
  }

  /** What the user should do next, given the project's state. */
  get guide(): { title: string; body: string; cta: string; layer: ProjectLayer } {
    if (this.total === 0) {
      return {
        title: 'Genera tus primeros controles',
        body: 'Escribe las reglas en lenguaje natural (una por línea) y el '
            + 'asistente planificará y construirá un DQC por cada una.',
        cta: 'Ir a Generar DQCs', layer: 'generar',
      };
    }
    if (this.pending > 0) {
      return {
        title: `Revisa ${this.pending} control${this.pending > 1 ? 'es' : ''} pendiente${this.pending > 1 ? 's' : ''}`,
        body: 'Para cada uno verás cómo se decidió, la consulta SQL y los casos '
            + 'que detecta; valida o rechaza desde ahí.',
        cta: 'Ir a Editar DQCs', layer: 'editar',
      };
    }
    if (this.validated > 0) {
      return {
        title: 'Todo revisado — exporta el dashboard',
        body: `${this.validated} control${this.validated > 1 ? 'es' : ''} validado${this.validated > 1 ? 's' : ''}. `
            + 'Copia las consultas con «Copiar todo», o genera más reglas.',
        cta: 'Ir a Editar DQCs', layer: 'editar',
      };
    }
    return {
      title: 'Genera nuevos controles',
      body: 'Todos los controles anteriores fueron rechazados. Reformula las '
          + 'reglas y vuelve a generarlas.',
      cta: 'Ir a Generar DQCs', layer: 'generar',
    };
  }

  get stepList() {
    const files = [this.project?.dictionaryName, this.project?.dataFileName]
      .filter(Boolean) as string[];
    const generated = this.total > 0;
    const reviewed = generated && this.pending === 0;
    return [
      { n: 1, title: 'Proyecto creado.',
        hint: files.length ? ` Adjuntos: ${files.join(', ')}.`
                           : ' Puedes adjuntar diccionario y datos al generar.',
        done: true, current: false },
      { n: 2, title: 'Generar DQCs.', hint: ' El asistente planifica y construye cada control.',
        done: generated, current: !generated },
      { n: 3, title: 'Revisar y validar.', hint: ' Traza, SQL y casos detectados de cada DQC.',
        done: reviewed, current: generated && !reviewed },
      { n: 4, title: 'Exportar.', hint: ' Copia las consultas validadas al dashboard.',
        done: false, current: reviewed },
    ];
  }

  ngOnChanges(): void { /* counts arrive via @Input; getters recompute */ }
}
