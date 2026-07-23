import { Component, EventEmitter, Input, NgZone, OnChanges, OnInit, Output, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DqcService } from '../services/dqc.service';
import { ChatMessage, InspectResponse, PlanItem, StreamEvent, TreeNode } from '../models/dqc.model';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css'],
})
export class ChatComponent implements OnInit, OnChanges {
  @Output() dqcGenerated = new EventEmitter<void>();
  @Input() retryInstructions = '';

  messages: ChatMessage[] = [];
  instructions = '';
  tableName = 'mylib.ciclos_recuperacion';
  dictionaryFile: File | null = null;
  dictionaryName = '';
  testsFile: File | null = null;
  testsFileName = '';
  dataFile: File | null = null;
  dataFileName = '';
  isLoading = false;
  isInspecting = false;

  // experimental anti-hallucination features, off by default
  useGrounding = false;   // sample real domain values into the prompt
  useJudge = false;       // LLM-as-judge semantic review per DQC

  // dictionary intelligence state: the LLM's sheet/mapping proposal,
  // confirmed or overridden by the user via the option buttons
  selectedSheet: string | null = null;
  columnMapping: Record<string, string | null> | null = null;

  constructor(private dqcService: DqcService, private zone: NgZone) {}

  ngOnInit(): void {
    this.loadDemoAssets();
  }

  ngOnChanges(changes: SimpleChanges): void {
    const retry = changes['retryInstructions'];
    if (!retry || retry.firstChange || !retry.currentValue) return;
    this.instructions = retry.currentValue;
    this.messages.push({
      role: 'assistant',
      content: 'Regla reformulada preparada. Pulsa Generar para crear una propuesta nueva; el DQC rechazado seguirá rechazado.',
    });
  }

  /** Resolve WHERE the default dictionary/cases/rules load from. Precedence:
   *   1. URL query params  ?dict=<url>&cases=<url>&rules=<url>
   *   2. assets/demo/sources.json  (edit to point at your own hosted files,
   *      e.g. S3 objects — no rebuild needed)
   *   3. the files bundled under assets/demo/  (fallback)
   * Remote URLs must be reachable by the browser (public/presigned + CORS
   * allowing this app's origin). */
  private async resolveDemoSources(): Promise<{ dictionary: string; cases: string; rules: string }> {
    const bundled = {
      dictionary: 'assets/demo/diccionario_demo.xlsx',
      cases: 'assets/demo/casos_demo.xlsx',
      rules: 'assets/demo/reglas_demo.txt',
    };
    let cfg: Record<string, string | null> = {};
    try {
      const r = await fetch('assets/demo/sources.json', { cache: 'no-store' });
      if (r.ok) cfg = await r.json();
    } catch { /* no config file → use bundled */ }
    const q = new URLSearchParams(window.location.search);
    return {
      dictionary: q.get('dict')  || cfg['dictionary'] || bundled.dictionary,
      cases:      q.get('cases') || cfg['cases']      || bundled.cases,
      rules:      q.get('rules') || cfg['rules']      || bundled.rules,
    };
  }

  /** Preload the bundled demo dictionary + cases Excel so the app starts
   * ready to generate. Silent no-op when the assets are absent (e.g. a
   * deployment that strips them) or the user already picked files. */
  private async loadDemoAssets(): Promise<void> {
    const fetchAsset = async (path: string, name: string): Promise<File | null> => {
      try {
        const resp = await fetch(path);
        if (!resp.ok) return null;
        const blob = await resp.blob();
        return new File([blob], name, {
          type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        });
      } catch {
        return null;
      }
    };
    const fetchText = async (path: string): Promise<string | null> => {
      try {
        const resp = await fetch(path);
        if (!resp.ok) return null;
        return (await resp.text()).trim() || null;
      } catch {
        return null;
      }
    };

    const src = await this.resolveDemoSources();
    const [dict, casos, reglas] = await Promise.all([
      fetchAsset(src.dictionary, 'diccionario_demo.xlsx'),
      fetchAsset(src.cases, 'casos_demo.xlsx'),
      fetchText(src.rules),
    ]);
    this.zone.run(() => {
      if (this.dictionaryFile || this.dataFile) return;  // user was faster
      if (dict) {
        this.dictionaryFile = dict;
        this.dictionaryName = dict.name;
      }
      if (casos) {
        this.dataFile = casos;
        this.dataFileName = casos.name;
      }
      if (reglas && !this.instructions) this.instructions = reglas;
      const loaded: string[] = [];
      if (dict) loaded.push(`diccionario: ${dict.name}`);
      if (casos) loaded.push(`casos: ${casos.name}`);
      if (reglas) loaded.push('reglas de ejemplo');
      if (loaded.length) {
        this.messages.push({
          role: 'assistant',
          content: `Datos de ejemplo cargados (${loaded.join(', ')}). ` +
            'Puedes sustituirlos seleccionando tus propios ficheros.',
        });
      }
      if (dict) this.inspectDictionary(dict);
    });
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    this.dictionaryFile = file;
    this.dictionaryName = file?.name ?? '';
    this.selectedSheet = null;
    this.columnMapping = null;
    if (file) this.inspectDictionary(file);
  }

  private inspectDictionary(file: File): void {
    this.isInspecting = true;
    this.dqcService.inspect(file).subscribe({
      next: (res: InspectResponse) => {
        this.isInspecting = false;
        this.columnMapping = res.column_mapping ?? null;
        const sheetNames = res.options.length > 0
          ? res.options
          : res.sheets.map((s) => s.name);
        const mapped = Object.entries(res.column_mapping ?? {})
          .filter(([, v]) => v)
          .map(([k, v]) => `${k}→${v}`)
          .join(', ');

        if (res.question && sheetNames.length > 1) {
          // the model is unsure — ask, render the sheets as buttons
          this.messages.push({
            role: 'assistant',
            content: res.question,
            options: sheetNames,
            proposedOption: res.proposed_sheet ?? undefined,
          });
        } else if (res.proposed_sheet) {
          this.selectedSheet = res.proposed_sheet;
          // confident proposal: pre-select it but still surface the other
          // sheets as buttons so the user can override the LLM's choice
          this.messages.push({
            role: 'assistant',
            content: `Diccionario detectado en la hoja "${res.proposed_sheet}"` +
              (mapped ? ` (columnas: ${mapped})` : '') +
              (sheetNames.length > 1
                ? '. Confirma la hoja o elige otra:'
                : '. Escribe las instrucciones y pulsa Generar.'),
            options: sheetNames.length > 1 ? sheetNames : undefined,
            proposedOption: res.proposed_sheet,
          });
        }
      },
      error: (err) => {
        this.isInspecting = false;
        const detail = err.error?.detail;
        if (typeof detail === 'string') {
          // unreadable workbook (.xls antiguo, corrupto, con contraseña…)
          this.messages.push({ role: 'assistant', content: detail });
        }
        // otherwise inspection stays best-effort; generation can ask via 422
      },
    });
  }

  onDataFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    this.dataFile = file;
    this.dataFileName = file?.name ?? '';
    if (file) {
      this.messages.push({
        role: 'assistant',
        content: `Excel de datos "${file.name}" cargado. Las consultas ` +
          `generadas se ejecutarán sobre estos casos (columna DQC_ID → ` +
          `precisión y recall si las reglas llevan id previo).`,
      });
    }
  }

  onTestsFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    this.testsFile = file;
    this.testsFileName = file?.name ?? '';
    if (file) {
      this.messages.push({
        role: 'assistant',
        content: `Lista de tests "${file.name}" cargada. Se combinará con las instrucciones escritas (una regla por línea/fila).`,
      });
    }
  }

  chooseSheet(msg: ChatMessage, name: string): void {
    msg.selectedOption = name;
    this.selectedSheet = name;
    this.messages.push({ role: 'user', content: `Hoja: ${name}` });
    this.messages.push({
      role: 'assistant',
      content: `Usaré la hoja "${name}". Escribe las instrucciones y pulsa Generar.`,
    });
  }

  generate(): void {
    const text = this.instructions.trim();
    if ((!text && !this.testsFile) || !this.dictionaryFile || this.isLoading) return;

    const parts = [`Diccionario: ${this.dictionaryName}`];
    if (this.testsFileName) parts.push(`Lista de tests: ${this.testsFileName}`);
    if (text) parts.push(text);
    this.messages.push({ role: 'user', content: parts.join('\n\n') });
    this.isLoading = true;

    // Plan-mode ReAct generation: the backend splits the rules into an
    // action plan, then runs the verify→generate→validate loop per item.
    // The plan renders as a live checklist that ticks off item by item.
    this.activePlan = null;

    this.dqcService
      .generateStream(this.dictionaryFile, text, this.tableName,
                      this.selectedSheet ?? undefined, this.columnMapping ?? undefined,
                      this.testsFile ?? undefined, this.dataFile ?? undefined,
                      { valueGrounding: this.useGrounding, semanticJudge: this.useJudge })
      // fetch() resolves outside Angular's zone — re-enter it so the
      // checklist repaints on every event
      .subscribe({
        next: (ev: StreamEvent) => this.zone.run(() => this.onStreamEvent(ev)),
        error: (err) => this.zone.run(() => {
          const detail = err.error?.detail;
          if (detail?.needs_sheet_selection) {
            // backend could not pick a sheet on its own — ask here
            this.columnMapping = detail.column_mapping ?? this.columnMapping;
            this.messages.push({
              role: 'assistant',
              content: detail.question,
              options: detail.options ?? [],
            });
          } else {
            this.messages.push({
              role: 'assistant',
              content: `Error: ${typeof detail === 'string' ? detail : err.message || 'No se pudo conectar con el servidor'}`,
            });
          }
          this.isLoading = false;
        }),
        complete: () => this.zone.run(() => { this.isLoading = false; }),
      });
  }

  /** Run the already-stored DQCs against the uploaded cases Excel (no LLM). */
  evaluateData(): void {
    if (!this.dataFile || this.isLoading) return;
    this.messages.push({ role: 'user', content: `Evaluar Excel de datos: ${this.dataFileName}` });
    this.isLoading = true;

    this.dqcService.evaluate(this.dataFile, this.tableName).subscribe({
      next: (res) => {
        let summary = `Evaluación sobre ${res.casos} casos: ` +
          `${res.evaluados} DQC${res.evaluados !== 1 ? 's' : ''} ejecutado${res.evaluados !== 1 ? 's' : ''}` +
          (res.fallidos ? `, ${res.fallidos} con error` : '');
        if (res.precision_media != null) {
          summary += `. Precisión media ${(res.precision_media * 100).toFixed(0)}%, ` +
            `recall medio ${((res.recall_medio ?? 0) * 100).toFixed(0)}%`;
        }
        summary += '.';
        this.messages.push({
          role: 'assistant',
          content: res.resultados.length ? summary
            : 'No hay DQCs almacenados que evaluar. Genera algunos primero.',
          evalResults: res.resultados,
        });
        this.isLoading = false;
      },
      error: (err) => {
        const detail = err.error?.detail;
        this.messages.push({
          role: 'assistant',
          content: `Error al evaluar: ${typeof detail === 'string' ? detail : err.message || 'No se pudo conectar con el servidor'}`,
        });
        this.isLoading = false;
      },
    });
  }

  /** Build the per-DQC binary decision tree (root on top) from the stream
   * events, mirroring the ReAct loop: ¿información suficiente? → generar
   * consulta → ¿consulta válida? → (¿el juez la aprueba?) → casos. A failed
   * decision hangs the next attempt off its "No" branch, so the correction
   * loop reads as a descending chain. */
  private updateTree(item: PlanItem, d: any): void {
    const resolve = (side: 'yes' | 'no', child: TreeNode) => {
      const dec = item._pending!;
      dec.taken = side;
      dec.estado = 'hecho';
      dec[side] = child;
      item._pending = undefined;
      item._tail = child;
    };

    if (d.estado === 'en_curso' && d.fase === 'suficiencia') {
      item.tree = { label: '¿Información suficiente?', kind: 'decision', estado: 'activo' };
      item.treeOpen = item.treeOpen ?? true;
      item._pending = item.tree;
      item._tail = item.tree;
    } else if (d.estado === 'ambigua' && item._pending) {
      resolve('no', { label: 'Regla ambigua', kind: 'leaf', estado: 'fallo', detail: d.falta });
    } else if (d.estado === 'en_curso' && d.fase === 'generacion') {
      const gen: TreeNode = {
        label: `Generar consulta SAS${d.intento > 1 ? ` (intento ${d.intento})` : ''}`,
        kind: 'action', estado: 'activo',
      };
      if (item._pending) {
        // suficiencia → Sí; a failed validity/judge decision → No (retry)
        resolve(item._pending.label.includes('suficiente') ? 'yes' : 'no', gen);
      } else if (item._tail) {
        item._tail.next = gen;
        item._tail = gen;
      }
    } else if (d.estado === 'en_curso' && d.fase === 'validacion' && item._tail) {
      item._tail.estado = 'hecho';
      const dec: TreeNode = { label: '¿Consulta válida?', kind: 'decision', estado: 'activo' };
      item._tail.next = dec;
      item._pending = dec;
      item._tail = dec;
    } else if (d.estado === 'en_curso' && d.fase === 'juicio' && item._pending) {
      const dec: TreeNode = { label: '¿El juez la aprueba?', kind: 'decision', estado: 'activo' };
      resolve('yes', dec);
      item._pending = dec;
    } else if (d.estado === 'completado' && item._pending) {
      const n = d.validacion?.n_casos;
      resolve('yes', {
        label: n != null
          ? `${n} caso${n !== 1 ? 's' : ''} detectado${n !== 1 ? 's' : ''}`
          : 'DQC validado',
        kind: 'leaf', estado: 'ok',
        detail: d.validacion?.precision != null
          ? `P ${(d.validacion.precision * 100).toFixed(0)}% · R ${(d.validacion.recall * 100).toFixed(0)}%`
          : undefined,
      });
    } else if (d.estado === 'error' && item._pending) {
      resolve('no', { label: 'Agotados los intentos', kind: 'leaf',
                      estado: 'fallo', detail: d.error });
    }
  }

  hasCases(p: PlanItem): boolean {
    return (p.validacion?.ejemplos?.length ?? 0) > 0;
  }

  /** One-line, human explanation of why the shown rows were flagged. */
  caseHint(condicion?: string, descripcion?: string): string {
    const c = (condicion || '').trim();
    const d = (descripcion || '').trim();
    if (c) return `Estas filas se marcan porque cumplen la condición de error: ${c}`;
    if (d) return `Filas que incumplen la regla: ${d}`;
    return 'Filas que la consulta marca como incumplimiento de la regla.';
  }

  /** Load a failed/ambiguous rule back into the input so it can be clarified
   * and regenerated. An ambiguous rule usually needs editing to succeed, so
   * we prefill rather than blindly re-running the same text. */
  retryItem(p: PlanItem): void {
    this.instructions = p.regla;
    this.messages.push({
      role: 'assistant',
      content: p.estado === 'ambigua'
        ? 'Regla cargada en el cuadro de texto. Aclárala (añade el umbral, los campos o el criterio que falta) y pulsa «Generar» para reintentar.'
        : 'Regla cargada en el cuadro de texto. Revísala o reformúlala y pulsa «Generar» para reintentar.',
    });
  }

  faseLabel(p: PlanItem): string {
    const labels: Record<string, string> = {
      suficiencia: 'verificando información…',
      grounding: 'muestreando valores reales…',
      generacion: 'generando consulta SAS…',
      validacion: 'validando consulta…',
      juicio: 'juez semántico revisando…',
    };
    const base = labels[p.fase ?? ''] ?? 'procesando…';
    return p.intento && p.intento > 1 ? `${base} (intento ${p.intento})` : base;
  }

  /** plan checklist of the stream currently in flight */
  private activePlan: PlanItem[] | null = null;

  private onStreamEvent(ev: StreamEvent): void {
    if (ev.type === 'plan') {
      const items: PlanItem[] = ev.data.items ?? [];
      this.activePlan = items;
      this.messages.push({
        role: 'assistant',
        content: `Plan de generación — ${items.length} DQC${items.length !== 1 ? 's' : ''}:`,
        plan: items,
      });
    } else if (ev.type === 'item') {
      const item = this.activePlan?.find((p) => p.id === ev.data.id);
      if (item) {
        item.estado = ev.data.estado;
        item.fase = ev.data.fase;
        item.intento = ev.data.intento;
        if (ev.data.falta) item.falta = ev.data.falta;
        if (ev.data.campos) item.campos = ev.data.campos;
        if (ev.data.validacion) item.validacion = ev.data.validacion;
        if (ev.data.dqcs) item.dqcs = ev.data.dqcs;
        if (ev.data.error) item.error = ev.data.error;
        if (ev.data.trace) item.trace = ev.data.trace;
        this.updateTree(item, ev.data);
      }
    } else if (ev.type === 'done') {
      const d = ev.data;
      const count = d.dqcs?.length ?? 0;
      let summary = count > 0
        ? `Se generaron ${count} DQC${count > 1 ? 's' : ''} (${d.dictionary_fields} campos, hoja "${d.sheet_used}"${d.formats_inferred ? `, ${d.formats_inferred} formatos inferidos` : ''}). Revisa el panel izquierdo.`
        : d.context_summary;
      const ev2 = d.evaluacion;
      if (ev2) {
        summary += ` Evaluación sobre ${ev2.casos} casos: ${ev2.tests_comprobados} test${ev2.tests_comprobados !== 1 ? 's' : ''} comprobado${ev2.tests_comprobados !== 1 ? 's' : ''}`;
        if (ev2.precision_media != null) {
          summary += `, precisión media ${(ev2.precision_media * 100).toFixed(0)}%, recall medio ${(ev2.recall_medio * 100).toFixed(0)}%`;
        }
        if (ev2.ambiguas) summary += `, ${ev2.ambiguas} regla(s) ambigua(s)`;
        summary += '.';
      }
      this.messages.push({ role: 'assistant', content: summary, dqcs: d.dqcs ?? [] });
      if (count > 0) this.dqcGenerated.emit();
    }
    // 'meta' needs no UI — the plan message carries the useful context
  }
}
