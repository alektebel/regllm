import { Component, EventEmitter, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DqcService } from '../services/dqc.service';
import { ChatMessage } from '../models/dqc.model';

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
  isLoading = false;

  constructor(private dqcService: DqcService) {}

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    this.dictionaryFile = file;
    this.dictionaryName = file?.name ?? '';
  }

  generate(): void {
    const text = this.instructions.trim();
    if (!text || !this.dictionaryFile || this.isLoading) return;

    this.messages.push({
      role: 'user',
      content: `Diccionario: ${this.dictionaryName}\n\n${text}`,
    });
    this.isLoading = true;

    this.dqcService.generate(this.dictionaryFile, text, this.tableName).subscribe({
      next: (res) => {
        const count = res.dqcs.length;
        const summary = count > 0
          ? `Se generaron ${count} DQC${count > 1 ? 's' : ''} (${res.dictionary_fields} campos en diccionario). Revisa el panel izquierdo.`
          : res.context_summary;

        this.messages.push({
          role: 'assistant',
          content: summary,
          dqcs: res.dqcs,
        });
        this.isLoading = false;
        if (count > 0) {
          this.dqcGenerated.emit();
        }
      },
      error: (err) => {
        this.messages.push({
          role: 'assistant',
          content: `Error: ${err.error?.detail || err.message || 'No se pudo conectar con el servidor'}`,
        });
        this.isLoading = false;
      },
    });
  }
}
