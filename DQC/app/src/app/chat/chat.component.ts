import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DqcService } from '../services/dqc.service';
import { ChatMessage, DqcItem, RAGSource } from '../models/dqc.model';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css'],
})
export class ChatComponent {
  messages: ChatMessage[] = [];
  userMessage = '';
  isLoading = false;
  suggestions = [
    'Verifica que PD_ESTIMADA cumple los suelos regulatorios',
    'Comprueba la consistencia de LGD_ESTIMADA con los floors por segmento',
    'Valida que el provision period cumple los mínimos por fase de ciclo',
    'Genera DQCs para comprobar el cálculo de ECL (PD × LGD × EAD)',
    'Comprueba que STAGE_IFRS9 es coherente con DPDS',
  ];

  constructor(private dqcService: DqcService) {}

  send(): void {
    const msg = this.userMessage.trim();
    if (!msg || this.isLoading) return;

    this.messages.push({ role: 'user', content: msg });
    this.userMessage = '';
    this.isLoading = true;

    this.dqcService.generate(msg).subscribe({
      next: (res) => {
        this.messages.push({
          role: 'assistant',
          content: res.context_summary,
          dqcs: res.dqcs,
          sources: res.sources,
        });
        this.isLoading = false;
      },
      error: (err) => {
        this.messages.push({
          role: 'assistant',
          content: `Error: ${err.message || 'No se pudo conectar con el servidor'}`,
        });
        this.isLoading = false;
      },
    });
  }

  useSuggestion(text: string): void {
    this.userMessage = text;
    this.send();
  }

  severityClass(sev: string): string {
    switch (sev) {
      case 'bloqueante': return 'badge-error';
      case 'advertencia': return 'badge-warn';
      default: return 'badge-info';
    }
  }
}
