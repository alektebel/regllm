"use client";

import { useMemo } from "react";
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

const Plot = createPlotlyComponent(Plotly);

interface Point {
  id: string;
  path: string;
  heading: string;
  snippet: string;
  level: number;
  x: number;
  y: number;
  z?: number;
}

interface Props {
  points: Point[];
  dims: 2 | 3;
  pathColors: Map<string, string>;
  onSelect: (point: Point) => void;
}

export default function PlotlyScatter({ points, dims, pathColors, onSelect }: Props) {
  const traces = useMemo(() => {
    const byPath = new Map<string, Point[]>();
    for (const p of points) {
      const list = byPath.get(p.path) || [];
      list.push(p);
      byPath.set(p.path, list);
    }
    return Array.from(byPath.entries()).map(([path, pts]) => {
      const base = {
        name: path,
        text: pts.map((p) => `<b>${p.heading}</b><br>${p.path}<br>${p.snippet.slice(0, 160)}`),
        hovertemplate: "%{text}<extra></extra>",
        customdata: pts.map((p) => p.id),
        marker: { size: dims === 3 ? 4 : 7, color: pathColors.get(path) || "#60a5fa" },
        x: pts.map((p) => p.x),
        y: pts.map((p) => p.y),
        meta: pts,
      };
      if (dims === 3) {
        return { ...base, type: "scatter3d" as const, mode: "markers" as const, z: pts.map((p) => p.z ?? 0) };
      }
      return { ...base, type: "scattergl" as const, mode: "markers" as const };
    });
  }, [points, dims, pathColors]);

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
        legend: { x: 0, y: 1, bgcolor: "rgba(0,0,0,0)" },
        scene:
          dims === 3
            ? {
                xaxis: { color: "#a1a1aa", gridcolor: "#3f3f46" },
                yaxis: { color: "#a1a1aa", gridcolor: "#3f3f46" },
                zaxis: { color: "#a1a1aa", gridcolor: "#3f3f46" },
              }
            : undefined,
        xaxis: dims === 2 ? { color: "#a1a1aa", gridcolor: "#3f3f46" } : undefined,
        yaxis: dims === 2 ? { color: "#a1a1aa", gridcolor: "#3f3f46" } : undefined,
      }}
      config={{ displaylogo: false, responsive: true }}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
      onClick={(e: any) => {
        const pt = e.points?.[0];
        if (!pt) return;
        const trace = traces[pt.curveNumber];
        const meta = (trace as any)?.meta?.[pt.pointIndex];
        if (meta) onSelect(meta);
      }}
    />
  );
}
