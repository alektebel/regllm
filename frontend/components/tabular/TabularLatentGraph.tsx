"use client";

import { useCallback, useEffect, useRef } from "react";

export interface TabularGraphNode {
  id: string;
  heading: string;
  path: string;
  cluster: number;
  color: string;
  outlier_score?: number;
  metadata?: Record<string, unknown>;
  x?: number;
  y?: number;
}

export interface TabularGraphEdge {
  source: string;
  target: string;
  weight: number;
}

interface Props {
  nodes: TabularGraphNode[];
  edges: TabularGraphEdge[];
  colorValues: string[];
  onSelect: (node: TabularGraphNode) => void;
  onHover: (node: TabularGraphNode | null) => void;
}

const PALETTE = [
  "#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa",
  "#fb923c", "#22d3ee", "#f87171", "#4ade80", "#e879f9",
  "#facc15", "#38bdf8", "#fb7185", "#a3e635", "#c084fc",
  "#94a3b8", "#fcd34d", "#6ee7b7", "#fda4af", "#93c5fd",
];

function colorFor(value: string, colorValues: string[]): string {
  const idx = colorValues.indexOf(value);
  if (idx >= 0) return PALETTE[idx % PALETTE.length];
  let h = 0;
  for (let i = 0; i < value.length; i++) h = (h * 31 + value.charCodeAt(i)) >>> 0;
  return PALETTE[h % PALETTE.length];
}

function layoutNodes(nodes: TabularGraphNode[]): TabularGraphNode[] {
  const xs = nodes.map((n) => n.x ?? 0);
  const ys = nodes.map((n) => n.y ?? 0);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const scale = 320 / Math.max(spanX, spanY);

  return nodes.map((n) => ({
    ...n,
    x: ((n.x ?? 0) - (minX + maxX) / 2) * scale,
    y: ((n.y ?? 0) - (minY + maxY) / 2) * scale,
  }));
}

export default function TabularLatentGraph({
  nodes,
  edges,
  colorValues,
  onSelect,
  onHover,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<TabularGraphNode[]>([]);
  const edgesRef = useRef<TabularGraphEdge[]>([]);
  const transformRef = useRef({ x: 0, y: 0, k: 1 });
  const hoveredRef = useRef<TabularGraphNode | null>(null);
  const colorValuesRef = useRef(colorValues);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { width, height } = canvas;
    const { x: tx, y: ty, k } = transformRef.current;
    const colors = colorValuesRef.current;

    ctx.clearRect(0, 0, width, height);
    ctx.save();
    ctx.translate(width / 2 + tx, height / 2 + ty);
    ctx.scale(k, k);

    const nodeById = new Map(nodesRef.current.map((n) => [n.id, n]));
    const hovered = hoveredRef.current;

    // Edges (Obsidian-style faint links)
    for (const e of edgesRef.current) {
      const s = nodeById.get(typeof e.source === "string" ? e.source : (e.source as TabularGraphNode).id);
      const t = nodeById.get(typeof e.target === "string" ? e.target : (e.target as TabularGraphNode).id);
      if (!s?.x || !t?.x) continue;
      const alpha = Math.min(0.45, 0.15 + (e.weight - 0.2) * 0.8);
      ctx.strokeStyle = `rgba(161,161,170,${alpha})`;
      ctx.lineWidth = hovered && (hovered.id === s.id || hovered.id === t.id) ? 1.2 : 0.6;
      ctx.beginPath();
      ctx.moveTo(s.x, s.y!);
      ctx.lineTo(t.x, t.y!);
      ctx.stroke();
    }

    // Nodes
    for (const n of nodesRef.current) {
      if (n.x == null) continue;
      const isHovered = hovered?.id === n.id;
      const r = isHovered ? 10 : 5;
      const fill = colorFor(n.color, colors);

      if (isHovered) {
        ctx.shadowColor = fill;
        ctx.shadowBlur = 18;
      } else {
        ctx.shadowBlur = 0;
      }

      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.arc(n.x, n.y!, r, 0, Math.PI * 2);
      ctx.fill();

      if ((n.outlier_score ?? 0) > 0.35) {
        ctx.strokeStyle = "#f87171";
        ctx.lineWidth = isHovered ? 2 : 1;
        ctx.stroke();
      }

      ctx.shadowBlur = 0;
    }

    // Hover ring + label
    if (hovered?.x != null) {
      ctx.strokeStyle = "rgba(250,250,250,0.85)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(hovered.x, hovered.y!, 14, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = "#fafafa";
      ctx.font = "11px ui-monospace, monospace";
      const label = hovered.heading.slice(0, 48);
      ctx.fillText(label, hovered.x + 14, hovered.y! - 10);
    }

    ctx.restore();
  }, []);

  useEffect(() => {
    nodesRef.current = layoutNodes(nodes.map((n) => ({ ...n })));
    edgesRef.current = edges;
    colorValuesRef.current = colorValues;
    draw();
  }, [nodes, edges, colorValues, draw]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      canvas.width = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
      draw();
    });
    ro.observe(canvas);
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    return () => ro.disconnect();
  }, [draw]);

  const hitTest = useCallback((cx: number, cy: number): TabularGraphNode | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const { x: tx, y: ty, k } = transformRef.current;
    const mx = (cx - canvas.width / 2 - tx) / k;
    const my = (cy - canvas.height / 2 - ty) / k;
    let best: TabularGraphNode | null = null;
    let bestDist = Infinity;
    for (const n of nodesRef.current) {
      if (n.x == null) continue;
      const dx = n.x - mx, dy = n.y! - my;
      const d2 = dx * dx + dy * dy;
      const r = hoveredRef.current?.id === n.id ? 196 : 64;
      if (d2 < r && d2 < bestDist) {
        best = n;
        bestDist = d2;
      }
    }
    return best;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let isPanning = false;
    let panStart = { x: 0, y: 0, tx: 0, ty: 0 };

    const onMouseDown = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
      const hit = hitTest(cx, cy);
      if (hit) {
        onSelect(hit);
      } else {
        isPanning = true;
        panStart = { x: e.clientX, y: e.clientY, tx: transformRef.current.x, ty: transformRef.current.y };
      }
    };

    const onMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
      if (isPanning) {
        transformRef.current.x = panStart.tx + (e.clientX - panStart.x);
        transformRef.current.y = panStart.ty + (e.clientY - panStart.y);
        draw();
      } else {
        const hit = hitTest(cx, cy);
        if (hit !== hoveredRef.current) {
          hoveredRef.current = hit;
          onHover(hit);
          canvas.style.cursor = hit ? "pointer" : "grab";
          draw();
        }
      }
    };

    const onMouseUp = () => { isPanning = false; };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      transformRef.current.k = Math.max(0.2, Math.min(8, transformRef.current.k * factor));
      draw();
    };

    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("mouseleave", onMouseUp);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      canvas.removeEventListener("mousedown", onMouseDown);
      canvas.removeEventListener("mousemove", onMouseMove);
      canvas.removeEventListener("mouseup", onMouseUp);
      canvas.removeEventListener("mouseleave", onMouseUp);
      canvas.removeEventListener("wheel", onWheel);
    };
  }, [hitTest, draw, onSelect, onHover]);

  return <canvas ref={canvasRef} className="w-full h-full bg-zinc-950 rounded-lg" />;
}
