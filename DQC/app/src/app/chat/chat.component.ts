import { Component, EventEmitter, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DqcService } from '../services/dqc.service';
import { ChatMessage, RAGSource } from '../models/dqc.model';

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
  userMessage = '';
  isLoading = false;
  suggestions = [
    'Verifica que PD_ESTIMADA cumple los suelos regulatorios',
    'Comprueba la consistencia de LGD_ESTIMADA con los floors por segmento',
    'Valida que el provision period cumple los mínimos por fase de ciclo',
    'Genera DQCs para comprobar el cálculo de ECL (PD x LGD x EAD)',
    'Comprueba que STAGE_IFRS9 es coherente con DPDS',
  ];

  // ── Simple mode: one or more NL expressions + at most one article ──────
  // Skips RAG context gathering entirely — a short, single-article prompt
  // for local models (e.g. REGLLM_LLM=gguf) that struggle with the full
  // multi-source context the default mode assembles. See
  // POST /dqc/generate/simple.
  simpleMode = false;
  articleParagraph: number | null = null;
  articleText = '';

  constructor(private dqcService: DqcService) {}

  send(): void {
    const msg = this.userMessage.trim();
    if (!msg || this.isLoading) return;

    this.messages.push({ role: 'user', content: msg });
    this.userMessage = '';
    this.isLoading = true;

    if (this.simpleMode) {
      this.sendSimple(msg);
    } else {
      this.sendDefault(msg);
    }
  }

  private sendDefault(msg: string): void {
    this.dqcService.generate(msg).subscribe({
      next: (res) => {
        const count = res.dqcs.length;
        const summary = count > 0
          ? `Se generaron ${count} DQC${count > 1 ? 's' : ''} para **${res.variable}**. Revisa el panel izquierdo.`
          : res.context_summary;

        this.messages.push({
          role: 'assistant',
          content: summary,
          sources: res.sources,
        });
        this.isLoading = false;

        if (count > 0) {
          this.dqcGenerated.emit();
        }
      },
      error: (err) => this.pushError(err),
    });
  }

  private sendSimple(expressions: string): void {
    const paragraph = this.articleParagraph || null;
    const text = paragraph ? null : (this.articleText.trim() || null);

    this.dqcService.generateSimple(expressions, paragraph, text).subscribe({
      next: (res) => {
        const count = res.dqcs.length;
        const articleNote = res.article_citation ? ` (${res.article_citation})` : '';
        const summary = count > 0
          ? `Se generaron ${count} DQC${count > 1 ? 's' : ''}${articleNote}. Revisa el panel izquierdo.`
          : res.context_summary;

        this.messages.push({ role: 'assistant', content: summary });
        this.isLoading = false;

        if (count > 0) {
          this.dqcGenerated.emit();
        }
      },
      error: (err) => this.pushError(err),
    });
  }

  private pushError(err: any): void {
    this.messages.push({
      role: 'assistant',
      content: `Error: ${err.message || 'No se pudo conectar con el servidor'}`,
    });
    this.isLoading = false;
  }

  useSuggestion(text: string): void {
    this.userMessage = text;
    this.send();
  }
}
