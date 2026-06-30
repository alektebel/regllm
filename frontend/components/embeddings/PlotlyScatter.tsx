"use client";

import { useMemo } from "react";
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

const Plot = createPlotlyComponent(Plotly);

export interface Point {
  id: string;
  path: string;
  heading: string;
  snippet: string;
  level: number;
  x: number;
  y: number;
  z?: number;
  evolving?: boolean;
  colorGroup?: string;   // optional override for color grouping (e.g. SEGMENTO value)
}

interface Props {
  points: Point[];
  dims: 2 | 3;
  pathColors: Map<string, string>;
  onSelect: (point: Point) => void;
}

const PALETTE = [
  "#60a5fa", "#f472b6", "#34d399", "#fbbf24",
  "#a78bfa", "#fb923c", "#22d3ee", "#f87171",
  "#86efac", "#c084fc", "#fdba74", "#38bdf8",
];

export default function PlotlyScatter({ points, dims, pathColors, onSelect }: Props) {
  const traces = useMemo(() => {
    const byGroup = new Map<string, Point[]>();
    for (const p of points) {
      const group = p.colorGroup ?? p.path;
      const list = byGroup.get(group) || [];
      list.push(p);
      byGroup.set(group, list);
    }

    const groupNames = Array.from(byGroup.keys());
    const colorOf = (g: string) =>
      pathColors.get(g) || PALETTE[groupNames.indexOf(g) % PALETTE.length];

    const scatterTraces = Array.from(byGroup.entries()).map(([group, pts]) => {
      const base = {
        name: group,
        text: pts.map(
          (p) =>
            `<b>${p.heading}</b><br><span style="color:#71717a">${p.path}</span><br>${p.snippet.slice(0, 180)}`
        ),
        hovertemplate: "%{text}<extra></extra>",
        marker: {
          size: dims === 3 ? 5 : 8,
          color: colorOf(group),
          line: { width: pts.some((p) => p.evolving) ? 1.5 : 0, color: "#fff" },
        },
        x: pts.map((p) => p.x),
        y: pts.map((p) => p.y),
        _meta: pts,
      };
      if (dims === 3) {
        return { ...base, type: "scatter3d" as const, mode: "markers" as const, z: pts.map((p) => p.z ?? 0) };
      }
      return { ...base, type: "scattergl" as const, mode: "markers" as const };
    });

    // Draw lines connecting evolving registers (same ID, multiple rows)
    const byId = new Map<string, Point[]>();
    for (const p of points) {
      if (!p.evolving) continue;
      const list = byId.get(p.heading) || [];
      list.push(p);
      byId.set(p.heading, list);
    }
    const lineTraces: any[] = [];
    for (const [, pts] of byId) {
      if (pts.length < 2) continue;
      const sorted = [...pts].sort((a, b) => a.path.localeCompare(b.path));
      const base = {
        showlegend: false,
        hoverinfo: "skip",
        line: { color: "rgba(255,255,255,0.15)", width: 1 },
        x: sorted.map((p) => p.x),
        y: sorted.map((p) => p.y),
        mode: "lines" as const,
      };
      if (dims === 3) {
        lineTraces.push({ ...base, type: "scatter3d" as const, z: sorted.map((p) => p.z ?? 0) });
      } else {
        lineTraces.push({ ...base, type: "scattergl" as const });
      }
    }

    return [...scatterTraces, ...lineTraces];
  }, [points, dims, pathColors]);

  const axis = { color: "#a1a1aa", gridcolor: "#3f3f46", zeroline: false };

  return (
    <Plot
      data={traces as any}
      layout={{
        autosize: true,
        margin: { l: 0, r: 0, t: 0, b: 0 },
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#a1a1aa", size: 11 },
        showlegend: true,
        legend: { x: 0, y: 1, bgcolor: "rgba(0,0,0,0)", font: { size: 10 } },
        scene: dims === 3 ? { xaxis: axis, yaxis: axis, zaxis: axis } : undefined,
        xaxis: dims === 2 ? axis : undefined,
        yaxis: dims === 2 ? axis : undefined,
      }}
      config={{ displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] }}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
      onClick={(e: any) => {
        const pt = e.points?.[0];
        if (!pt) return;
        const trace = traces[pt.curveNumber] as any;
        const meta = trace?._meta?.[pt.pointIndex];
        if (meta) onSelect(meta);
      }}
    />
  );
}
