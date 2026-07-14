import { Component, EventEmitter, NgZone, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DqcService } from '../services/dqc.service';
import { ChatMessage, InspectResponse, PlanItem, StreamEvent } from '../models/dqc.model';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css'],
})
export class ChatComponent {
  @Output() dqcGenerated = new EventEmitter<void>();

  messages: ChatMessage[] = [];
  instructions = '';
  tableName = 'mylib.ciclos_recuperacion';
  dictionaryFile: File | null = null;
  dictionaryName = '';
  testsFile: File | null = null;
  testsFileName = '';
  isLoading = false;
  isInspecting = false;

  // dictionary intelligence state: the LLM's sheet/mapping proposal,
  // confirmed or overridden by the user via the option buttons
  selectedSheet: string | null = null;
  columnMapping: Record<string, string | null> | null = null;

  constructor(private dqcService: DqcService, private zone: NgZone) {}

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
      error: () => {
        // inspection is best-effort; generation can still ask via 422
        this.isInspecting = false;
      },
    });
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

    // Plan-mode generation: the backend first splits the rules into a JSON
    // action plan, then executes one generation agent per plan item. The
    // plan renders as a live checklist that ticks off item by item.
    let planMsg: ChatMessage | null = null;

    this.dqcService
      .generateStream(this.dictionaryFile, text, this.tableName,
                      this.selectedSheet ?? undefined, this.columnMapping ?? undefined,
                      this.testsFile ?? undefined)
      // fetch() resolves outside Angular's zone — re-enter it so the
      // checklist repaints on every event
      .subscribe({
        next: (ev: StreamEvent) => this.zone.run(() => this.onStreamEvent(ev, {
          setPlan: (m) => { planMsg = m; },
          getPlan: () => planMsg,
        })),
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

  private onStreamEvent(
    ev: StreamEvent,
    planRef: { setPlan: (m: ChatMessage) => void; getPlan: () => ChatMessage | null },
  ): void {
    if (ev.type === 'plan') {
      const items: PlanItem[] = ev.data.items ?? [];
      const msg: ChatMessage = {
        role: 'assistant',
        content: `Plan de generación — ${items.length} DQC${items.length !== 1 ? 's' : ''}:`,
        plan: items,
      };
      planRef.setPlan(msg);
      this.messages.push(msg);
    } else if (ev.type === 'item') {
      const plan = planRef.getPlan()?.plan;
      const item = plan?.find((p) => p.id === ev.data.id);
      if (item) {
        item.estado = ev.data.estado;
        if (ev.data.dqcs) item.dqcs = ev.data.dqcs;
        if (ev.data.error) item.error = ev.data.error;
      }
    } else if (ev.type === 'done') {
      const d = ev.data;
      const count = d.dqcs?.length ?? 0;
      const summary = count > 0
        ? `Se generaron ${count} DQC${count > 1 ? 's' : ''} (${d.dictionary_fields} campos, hoja "${d.sheet_used}"${d.formats_inferred ? `, ${d.formats_inferred} formatos inferidos` : ''}). Revisa el panel izquierdo.`
        : d.context_summary;
      this.messages.push({ role: 'assistant', content: summary, dqcs: d.dqcs ?? [] });
      if (count > 0) this.dqcGenerated.emit();
    }
    // 'meta' needs no UI — the plan message carries the useful context
  }
}
