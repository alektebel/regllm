"use client";

interface ColInfo {
  name: string;
  kind: "numeric" | "categorical" | "text";
  n_unique: number;
}

interface Props {
  columns: ColInfo[];
  idCol: string;
  onIdColChange: (col: string) => void;
  featureCols: string[];
  onFeatureColsChange: (cols: string[]) => void;
  colorCol: string;
  onColorColChange: (col: string) => void;
}

const KIND_COLOR: Record<string, string> = {
  numeric: "text-blue-400",
  categorical: "text-emerald-400",
  text: "text-amber-400",
};

export default function ColumnPicker({
  columns,
  idCol,
  onIdColChange,
  featureCols,
  onFeatureColsChange,
  colorCol,
  onColorColChange,
}: Props) {
  const embeddable = columns.filter(
    (c) => c.name !== idCol && (c.kind === "numeric" || c.kind === "categorical")
  );
  const colorable = columns.filter((c) => c.kind === "categorical" && c.name !== idCol);

  const toggleFeature = (name: string) => {
    onFeatureColsChange(
      featureCols.includes(name)
        ? featureCols.filter((c) => c !== name)
        : [...featureCols, name]
    );
  };

  const selectAll = () => onFeatureColsChange(embeddable.map((c) => c.name));
  const selectNone = () => onFeatureColsChange([]);

  return (
    <div className="space-y-4 text-xs">
      <div>
        <label className="block text-muted-foreground mb-1">ID field (evolving register key)</label>
        <select
          value={idCol}
          onChange={(e) => onIdColChange(e.target.value)}
          className="w-full bg-muted/30 border border-border rounded px-2 py-1.5 text-foreground"
        >
          {columns.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name} · {c.n_unique} unique
            </option>
          ))}
        </select>
      </div>

      {colorable.length > 0 && (
        <div>
          <label className="block text-muted-foreground mb-1">Color points by</label>
          <select
            value={colorCol}
            onChange={(e) => onColorColChange(e.target.value)}
            className="w-full bg-muted/30 border border-border rounded px-2 py-1.5 text-foreground"
          >
            <option value="">— (group by ID)</option>
            {colorable.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
              </option>
            ))}
          </select>
        </div>
      )}

      <div>
        <div className="flex items-center justify-between mb-1">
          <label className="text-muted-foreground">Feature columns for embedding</label>
          <span className="flex gap-2">
            <button onClick={selectAll} className="text-primary hover:underline">all</button>
            <button onClick={selectNone} className="text-muted-foreground hover:underline">none</button>
          </span>
        </div>
        <div className="max-h-48 overflow-y-auto space-y-0.5 border border-border rounded p-1.5 bg-muted/10">
          {embeddable.map((c) => (
            <label key={c.name} className="flex items-center gap-2 py-0.5 px-1 rounded hover:bg-muted/30 cursor-pointer">
              <input
                type="checkbox"
                checked={featureCols.includes(c.name)}
                onChange={() => toggleFeature(c.name)}
                className="accent-primary"
              />
              <span className={KIND_COLOR[c.kind]}>{c.kind[0].toUpperCase()}</span>
              <span className="text-foreground">{c.name}</span>
              <span className="text-muted-foreground ml-auto">{c.n_unique}</span>
            </label>
          ))}
        </div>
        <p className="text-muted-foreground mt-1">
          {featureCols.length}/{embeddable.length} selected ·
          <span className="text-blue-400"> N</span> numeric
          <span className="text-emerald-400"> C</span> categorical
        </p>
      </div>
    </div>
  );
}
