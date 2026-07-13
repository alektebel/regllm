import { Component, EventEmitter, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DqcService } from '../services/dqc.service';
import { ChatMessage, InspectResponse } from '../models/dqc.model';

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

  constructor(private dqcService: DqcService) {}

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

  // TODO(streaming-reasoning): once DqcService.generateStream() exists,
  // switch generate() to it: subscribe to the event stream and
  //   - on 'step':     push an assistant message for the step name;
  //   - on 'thinking': append the delta to a collapsible "razonamiento"
  //                    section of the current step's bubble (new optional
  //                    `thinking?: string` field on ChatMessage);
  //   - on 'answer'/'result': fill the bubble's normal content/dqcs.
  // fetch() callbacks run outside Angular, so apply updates inside
  // NgZone.run() (or signals) so the chat re-renders per token, and keep
  // the messages list auto-scrolled to the bottom while streaming.
  generate(): void {
    const text = this.instructions.trim();
    if ((!text && !this.testsFile) || !this.dictionaryFile || this.isLoading) return;

    const parts = [`Diccionario: ${this.dictionaryName}`];
    if (this.testsFileName) parts.push(`Lista de tests: ${this.testsFileName}`);
    if (text) parts.push(text);
    this.messages.push({ role: 'user', content: parts.join('\n\n') });
    this.isLoading = true;

    this.dqcService
      .generate(this.dictionaryFile, text, this.tableName,
                this.selectedSheet ?? undefined, this.columnMapping ?? undefined,
                this.testsFile ?? undefined)
      .subscribe({
        next: (res) => {
          const count = res.dqcs.length;
          const summary = count > 0
            ? `Se generaron ${count} DQC${count > 1 ? 's' : ''} (${res.dictionary_fields} campos, hoja "${res.sheet_used}"${res.formats_inferred ? `, ${res.formats_inferred} formatos inferidos` : ''}). Revisa el panel izquierdo.`
            : res.context_summary;

          this.messages.push({ role: 'assistant', content: summary, dqcs: res.dqcs });
          this.isLoading = false;
          if (count > 0) this.dqcGenerated.emit();
        },
        error: (err) => {
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
        },
      });
  }
}
