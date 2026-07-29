import { Component, OnDestroy, OnInit, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChatComponent } from './chat/chat.component';
import { EditorComponent } from './editor/editor.component';
import { ProjectsComponent } from './projects/projects.component';
import { DqcService } from './services/dqc.service';
import { CountsResponse } from './models/dqc.model';
import { Project, ProjectLayer } from './models/project.model';
import { DEMO_MODE } from './demo/demo-backend';

/** App shell — three layers:
 *   1. projects          pick or create a data-quality project
 *   2. project / generar  the ReAct chat that plans and generates DQCs
 *   3. project / editar   review the DQCs: trace, SQL, detected cases
 */
@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, ChatComponent, EditorComponent, ProjectsComponent],
  template: `
    <div class="shell">
      @if (demo) {
        <div class="demo-banner">
          Demo estática — datos de ejemplo, sin backend.
          El flujo (plan, árbol de decisión, casos, métricas) es el real.
        </div>
      }

      @if (!project) {
        <app-projects (open)="openProject($event)" />
      } @else {
        <header class="topbar">
          <button class="back" (click)="closeProject()">← Proyectos</button>
          <div class="titles">
            <h1>{{ project.name }}</h1>
            <span class="table">{{ project.tableName }}</span>
          </div>
          <nav class="layers">
            <button [class.active]="layer === 'generar'" (click)="layer = 'generar'">
              Generar DQCs
            </button>
            <button [class.active]="layer === 'editar'" (click)="layer = 'editar'">
              Editar DQCs
              @if (counts) { <span class="pill">{{ total }}</span> }
            </button>
          </nav>
          @if (counts) {
            <div class="counts">
              <span class="cnt cnt-pending" title="Pendientes">{{ counts.pending_visible }}</span>
              <span class="cnt cnt-validated" title="Validados">{{ counts.validated }}</span>
              <span class="cnt cnt-rejected" title="Rechazados">{{ counts.rejected }}</span>
            </div>
          }
        </header>

        <main class="layer-body">
          @if (layer === 'generar') {
            <app-chat (dqcGenerated)="onGenerated()" />
          } @else {
            <app-editor (checksChanged)="loadCounts()" />
          }
        </main>
      }
    </div>
  `,
  styles: [`
    :host { display: block; height: 100vh; overflow: hidden; }
    .shell { display: flex; flex-direction: column; height: 100%; background: #0f0f1a; color: #ccc; }
    .demo-banner {
      background: rgba(243,156,18,.12); color: #f39c12; font-size: 12px;
      text-align: center; padding: 6px 12px; border-bottom: 1px solid rgba(243,156,18,.25);
    }
    .topbar {
      display: flex; align-items: center; gap: 16px; padding: 10px 18px;
      background: #161625; border-bottom: 1px solid #2a2a40;
    }
    .back { background: transparent; border: none; color: #6c7bbf; font-size: 13px;
            cursor: pointer; padding: 4px 6px; }
    .titles { display: flex; flex-direction: column; min-width: 0; }
    .titles h1 { margin: 0; font-size: 15px; color: #fff; font-weight: 600;
                 overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .table { font-size: 11px; color: #e65100; }
    .layers { display: flex; gap: 6px; margin-left: auto; }
    .layers button {
      background: transparent; border: 1px solid #2a2a40; color: #999;
      padding: 7px 16px; border-radius: 7px; font-size: 13px; font-weight: 600; cursor: pointer;
      display: flex; align-items: center; gap: 6px;
    }
    .layers button:hover { border-color: #6c7bbf; color: #ccc; }
    .layers button.active { background: #6c7bbf; border-color: #6c7bbf; color: #fff; }
    .pill { background: rgba(255,255,255,.2); border-radius: 8px; padding: 0 6px; font-size: 11px; }
    .counts { display: flex; gap: 5px; }
    .cnt { font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 10px; }
    .cnt-pending { background: #f9a825; color: #000; }
    .cnt-validated { background: #2e7d32; color: #fff; }
    .cnt-rejected { background: #c62828; color: #fff; }
    .layer-body { flex: 1; min-height: 0; display: flex; flex-direction: column; }
    app-projects, app-chat, app-editor { flex: 1; min-height: 0; }
  `],
})
export class AppComponent implements OnInit, OnDestroy {
  readonly demo = DEMO_MODE;
  project: Project | null = null;
  layer: ProjectLayer = 'generar';
  counts: CountsResponse | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;

  constructor(private dqc: DqcService) {}

  ngOnInit(): void {
    this.loadCounts();
    this.timer = setInterval(() => this.loadCounts(), 5000);
  }

  ngOnDestroy(): void {
    if (this.timer) clearInterval(this.timer);
  }

  get total(): number {
    const c = this.counts;
    return c ? c.pending_visible + c.validated + c.rejected : 0;
  }

  openProject(p: Project): void {
    this.project = p;
    this.layer = 'generar';
    this.loadCounts();
  }

  closeProject(): void {
    this.project = null;
  }

  onGenerated(): void {
    this.loadCounts();
  }

  loadCounts(): void {
    this.dqc.counts().subscribe({ next: (c) => { this.counts = c; } });
  }
}
