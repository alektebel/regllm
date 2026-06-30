"use client";

import { useRef } from "react";
import { Upload } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  onFile: (file: File) => void;
  loading: boolean;
}

export default function UploadZone({ onFile, loading }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onFile(file);
    e.target.value = "";
  };

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      onClick={() => !loading && inputRef.current?.click()}
      className={cn(
        "flex flex-col items-center justify-center gap-3 border-2 border-dashed border-border rounded-lg",
        "p-10 cursor-pointer transition-colors select-none",
        loading ? "opacity-50 cursor-wait" : "hover:border-primary/60 hover:bg-primary/5"
      )}
    >
      <Upload className="h-8 w-8 text-muted-foreground" />
      <div className="text-center">
        <p className="text-sm font-medium">Drop an Excel or CSV here</p>
        <p className="text-xs text-muted-foreground mt-0.5">or click to browse</p>
        <p className="text-[11px] text-muted-foreground mt-2">.xlsx · .xls · .csv · .tsv</p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".xlsx,.xls,.csv,.tsv"
        className="hidden"
        onChange={handleChange}
      />
    </div>
  );
}
