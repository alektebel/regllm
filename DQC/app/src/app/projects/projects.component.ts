import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Project } from '../models/project.model';
import { ProjectService } from '../services/project.service';

/** Layer 1 — pick a data-quality project or create one.
 * Creation is deliberately minimal: name the project, attach the files
 * once, and everything downstream (generar / editar) reuses them. */
@Component({
  selector: 'app-projects',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="projects">
      <header class="ph">
        <div>
          <h1>Proyectos de calidad de datos</h1>
          <p>Cada proyecto agrupa un diccionario, sus datos y los DQCs generados</p>
        </div>
        @if (!creating) {
          <button class="primary" (click)="startCreate()">+ Nuevo proyecto</button>
        }
      </header>

      @if (creating) {
        <section class="card create">
          <h2>Nuevo proyecto</h2>
          <p class="hint">Adjunta todo una vez; después podrás generar o editar DQCs.</p>

          <label class="field">
            <span>Nombre</span>
            <input type="text" [(ngModel)]="draft.name" name="name"
                   placeholder="Ciclos de recuperación — cartera 2026" />
          </label>

          <label class="field">
            <span>Tabla objetivo</span>
            <input type="text" [(ngModel)]="draft.tableName" name="table"
                   placeholder="mylib.ciclos_recuperacion" />
          </label>

          <div class="attach-row">
            <label class="attach" [class.done]="!!draft.dictionaryName">
              <input type="file" accept=".xlsx,.xls" (change)="pick($event, 'dictionaryName')" hidden />
              <strong>Diccionario</strong>
              <span>{{ draft.dictionaryName || 'Adjuntar .xlsx' }}</span>
            </label>
            <label class="attach" [class.done]="!!draft.dataFileName">
              <input type="file" accept=".xlsx,.xls" (change)="pick($event, 'dataFileName')" hidden />
              <strong>Datos (casos)</strong>
              <span>{{ draft.dataFileName || 'Opcional' }}</span>
            </label>
            <label class="attach" [class.done]="!!draft.testsFileName">
              <input type="file" accept=".txt,.md,.csv,.xlsx" (change)="pick($event, 'testsFileName')" hidden />
              <strong>Lista de tests</strong>
              <span>{{ draft.testsFileName || 'Opcional' }}</span>
            </label>
          </div>

          <div class="actions">
            <button class="ghost" (click)="creating = false">Cancelar</button>
            <button class="primary" [disabled]="!draft.name.trim()" (click)="create()">
              Crear proyecto
            </button>
          </div>
        </section>
      }

      <div class="grid">
        @for (p of projects; track p.id) {
          <article class="card project" (click)="open.emit(p)">
            <h3>{{ p.name }}</h3>
            <p class="table">{{ p.tableName }}</p>
            <ul class="files">
              <li [class.on]="!!p.dictionaryName">Diccionario: {{ p.dictionaryName || '—' }}</li>
              <li [class.on]="!!p.dataFileName">Datos: {{ p.dataFileName || '—' }}</li>
            </ul>
            <footer>
              <span class="date">{{ p.createdAt | date:'dd/MM/yyyy' }}</span>
              <button class="danger" (click)="remove($event, p)">Borrar</button>
            </footer>
          </article>
        } @empty {
          @if (!creating) {
            <div class="empty">
              Aún no hay proyectos. Crea el primero para empezar a generar DQCs.
            </div>
          }
        }
      </div>
    </div>
  `,
  styles: [`
    :host { display: block; height: 100%; overflow-y: auto; background: #0f0f1a; color: #ccc; }
    .projects { max-width: 1040px; margin: 0 auto; padding: 28px 24px 48px; }
    .ph { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; margin-bottom: 20px; }
    .ph h1 { margin: 0; font-size: 20px; color: #fff; font-weight: 600; }
    .ph p { margin: 4px 0 0; font-size: 13px; color: #777; }
    .card { background: #161625; border: 1px solid #2a2a40; border-radius: 10px; padding: 18px; }
    .create { margin-bottom: 24px; }
    .create h2 { margin: 0 0 2px; font-size: 15px; color: #fff; }
    .hint { margin: 0 0 14px; font-size: 12px; color: #777; }
    .field { display: flex; flex-direction: column; gap: 4px; margin-bottom: 12px; }
    .field span { font-size: 11px; text-transform: uppercase; letter-spacing: .4px; color: #777; }
    .field input {
      background: #0f0f1a; border: 1px solid #2a2a40; border-radius: 7px;
      padding: 9px 12px; color: #fff; font-size: 14px; outline: none;
    }
    .field input:focus { border-color: #6c7bbf; }
    .attach-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 6px 0 18px; }
    .attach {
      display: flex; flex-direction: column; gap: 3px; cursor: pointer;
      border: 1px dashed #2a2a40; border-radius: 8px; padding: 12px;
    }
    .attach:hover { border-color: #6c7bbf; }
    .attach.done { border-style: solid; border-color: #2e7d32; }
    .attach strong { font-size: 12px; color: #ccc; font-weight: 600; }
    .attach span { font-size: 11px; color: #777; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .actions { display: flex; justify-content: flex-end; gap: 8px; }
    button { border-radius: 7px; font-size: 13px; font-weight: 600; cursor: pointer; padding: 9px 18px; }
    .primary { background: #6c7bbf; color: #fff; border: none; }
    .primary:disabled { opacity: .4; cursor: not-allowed; }
    .ghost { background: transparent; color: #999; border: 1px solid #2a2a40; }
    .danger { background: transparent; color: #c62828; border: none; font-size: 11px; padding: 2px 6px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 14px; }
    .project { cursor: pointer; transition: border-color .15s, transform .15s; }
    .project:hover { border-color: #6c7bbf; transform: translateY(-1px); }
    .project h3 { margin: 0 0 2px; font-size: 15px; color: #fff; }
    .table { margin: 0 0 10px; font-size: 11px; color: #e65100;
             background: rgba(230,81,0,.1); display: inline-block; padding: 1px 7px; border-radius: 4px; }
    .files { list-style: none; margin: 0 0 12px; padding: 0; }
    .files li { font-size: 11px; color: #666; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .files li.on { color: #999; }
    .project footer { display: flex; align-items: center; justify-content: space-between; }
    .date { font-size: 11px; color: #555; }
    .empty { grid-column: 1/-1; text-align: center; color: #666; font-size: 13px; padding: 48px 16px; }
  `],
})
export class ProjectsComponent implements OnInit {
  @Output() open = new EventEmitter<Project>();

  projects: Project[] = [];
  creating = false;
  draft: Partial<Project> & { name: string } = this.blank();

  constructor(private store: ProjectService) {}

  ngOnInit(): void {
    this.projects = this.store.list();
  }

  startCreate(): void {
    this.draft = this.blank();
    this.creating = true;
  }

  pick(ev: Event, field: 'dictionaryName' | 'dataFileName' | 'testsFileName'): void {
    const name = (ev.target as HTMLInputElement).files?.[0]?.name ?? '';
    (this.draft as any)[field] = name;
  }

  create(): void {
    const p = this.store.create(this.draft);
    this.projects = this.store.list();
    this.creating = false;
    this.open.emit(p);
  }

  remove(ev: Event, p: Project): void {
    ev.stopPropagation();
    this.store.remove(p.id);
    this.projects = this.store.list();
  }

  private blank() {
    return {
      name: '', tableName: 'mylib.ciclos_recuperacion',
      dictionaryName: '', dataFileName: '', testsFileName: '', description: '',
    };
  }
}
