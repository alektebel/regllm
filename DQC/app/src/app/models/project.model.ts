/** A data-quality project: one dictionary + optional cases Excel + the
 * DQCs generated for it. Persisted client-side (localStorage) so the
 * workspace survives reloads and the static demo works with no backend. */
export interface Project {
  id: string;
  name: string;
  tableName: string;
  dictionaryName: string;
  dataFileName: string;
  testsFileName: string;
  createdAt: string;
  /** free-text note shown on the project card */
  description: string;
}

export type ProjectLayer = 'generar' | 'editar';
