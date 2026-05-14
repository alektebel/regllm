"use client";

import { ChevronDown } from "lucide-react";

const BACKENDS = [
  { value: "groq", label: "Groq (llama-3.3-70b)" },
  { value: "ollama", label: "Ollama (local GGUF)" },
  { value: "local", label: "Local (QLoRA)" },
];

interface Props {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

export function BackendSwitcher({ value, onChange, disabled }: Props) {
  const current = BACKENDS.find((b) => b.value === value);
  return (
    <div className="relative inline-block">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="appearance-none cursor-pointer rounded-md border border-border bg-card px-3 py-1.5 pr-7 text-xs text-muted-foreground hover:text-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50"
      >
        {BACKENDS.map((b) => (
          <option key={b.value} value={b.value}>
            {b.label}
          </option>
        ))}
      </select>
      <ChevronDown
        size={12}
        className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground"
      />
    </div>
  );
}
