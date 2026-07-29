import { Injectable } from '@angular/core';
import { Project } from '../models/project.model';

const KEY = 'regllm.projects';

/** Client-side project store (localStorage). Keeps the workspace usable
 * across reloads and lets the static demo run with no backend at all. */
@Injectable({ providedIn: 'root' })
export class ProjectService {
  list(): Project[] {
    try {
      const raw = localStorage.getItem(KEY);
      return raw ? (JSON.parse(raw) as Project[]) : [];
    } catch {
      return [];
    }
  }

  get(id: string): Project | null {
    return this.list().find((p) => p.id === id) ?? null;
  }

  create(partial: Partial<Project>): Project {
    const project: Project = {
      id: `prj_${Date.now().toString(36)}`,
      name: partial.name?.trim() || 'Proyecto sin nombre',
      tableName: partial.tableName?.trim() || 'mylib.ciclos_recuperacion',
      dictionaryName: partial.dictionaryName || '',
      dataFileName: partial.dataFileName || '',
      testsFileName: partial.testsFileName || '',
      description: partial.description?.trim() || '',
      createdAt: new Date().toISOString(),
    };
    this.save([project, ...this.list()]);
    return project;
  }

  update(id: string, patch: Partial<Project>): Project | null {
    const all = this.list();
    const i = all.findIndex((p) => p.id === id);
    if (i < 0) return null;
    all[i] = { ...all[i], ...patch, id: all[i].id };
    this.save(all);
    return all[i];
  }

  remove(id: string): void {
    this.save(this.list().filter((p) => p.id !== id));
  }

  private save(projects: Project[]): void {
    try {
      localStorage.setItem(KEY, JSON.stringify(projects));
    } catch {
      /* storage full / disabled — the session still works in memory */
    }
  }
}
