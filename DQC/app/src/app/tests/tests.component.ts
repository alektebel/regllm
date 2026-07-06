import { Component, EventEmitter, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DqcService } from '../services/dqc.service';
import { TestResult } from '../models/dqc.model';

@Component({
  selector: 'app-tests',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './tests.component.html',
  styleUrls: ['./tests.component.css'],
})
export class TestsComponent {
  @Output() testsGenerated = new EventEmitter<void>();

  testsText = '';
  results: TestResult[] = [];
  isLoading = false;
  error = '';

  constructor(private dqcService: DqcService) {}

  get generatedCount(): number {
    return this.results.filter((r) => r.dqc).length;
  }

  onFile(ev: Event): void {
    const input = ev.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result || '');
      // Prepend to whatever is already there, separated by a newline
      this.testsText = this.testsText
        ? `${this.testsText}\n${text}`
        : text;
    };
    reader.readAsText(file);
    input.value = '';
  }

  parseTests(): string[] {
    // Each non-empty line is a test. Blank lines separate but don't count.
    return this.testsText
      .split(/\r?\n/)
      .map((t) => t.trim())
      .filter((t) => t.length > 0);
  }

  run(): void {
    const tests = this.parseTests();
    if (!tests.length || this.isLoading) return;
    this.isLoading = true;
    this.error = '';
    this.results = [];

    this.dqcService.generateTests(tests).subscribe({
      next: (res) => {
        this.results = res.results;
        this.isLoading = false;
        if (res.results.some((r) => r.dqc)) {
          this.testsGenerated.emit();
        }
      },
      error: (err) => {
        this.error = err.message || 'No se pudo conectar con el servidor';
        this.isLoading = false;
      },
    });
  }

  clear(): void {
    this.testsText = '';
    this.results = [];
    this.error = '';
  }
}
