"use client";

import { useEffect, useRef, useCallback } from "react";
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
  SimulationNodeDatum,
  SimulationLinkDatum,
} from "d3-force";

interface GraphNode extends SimulationNodeDatum {
  id: string;
  heading: string;
  path: string;
  cluster: number;
}

interface GraphEdge extends SimulationLinkDatum<GraphNode> {
  source: string | GraphNode;
  target: string | GraphNode;
  weight: number;
}

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onSelect: (node: GraphNode) => void;
}

const CLUSTER_COLORS = [
  "#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa",
  "#fb923c", "#22d3ee", "#f87171", "#4ade80", "#e879f9",
  "#facc15", "#38bdf8", "#fb7185", "#a3e635", "#c084fc",
];

export default function ForceGraph({ nodes, edges, onSelect }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<ReturnType<typeof forceSimulation<GraphNode>> | null>(null);
  const nodesRef = useRef<GraphNode[]>([]);
  const edgesRef = useRef<GraphEdge[]>([]);
  const transformRef = useRef({ x: 0, y: 0, k: 1 });
  const dragRef = useRef<{ node: GraphNode | null; startX: number; startY: number }>({
    node: null, startX: 0, startY: 0,
  });
  const hoveredRef = useRef<GraphNode | null>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { width, height } = canvas;
    const { x: tx, y: ty, k } = transformRef.current;

    ctx.clearRect(0, 0, width, height);
    ctx.save();
    ctx.translate(width / 2 + tx, height / 2 + ty);
    ctx.scale(k, k);

    // Edges
    ctx.lineWidth = 0.5;
    for (const e of edgesRef.current) {
      const s = e.source as GraphNode;
      const t = e.target as GraphNode;
      if (s.x == null || t.x == null) continue;
      ctx.strokeStyle = `rgba(161,161,170,${Math.min(0.6, (e.weight - 0.3) * 2)})`;
      ctx.beginPath();
      ctx.moveTo(s.x, s.y!);
      ctx.lineTo(t.x, t.y!);
      ctx.stroke();
    }

    // Nodes
    const hovered = hoveredRef.current;
    for (const n of nodesRef.current) {
      if (n.x == null) continue;
      const r = hovered === n ? 6 : 4;
      ctx.fillStyle = CLUSTER_COLORS[((n.cluster % CLUSTER_COLORS.length) + CLUSTER_COLORS.length) % CLUSTER_COLORS.length];
      ctx.beginPath();
      ctx.arc(n.x, n.y!, r, 0, Math.PI * 2);
      ctx.fill();
    }

    // Hovered label
    if (hovered && hovered.x != null) {
      ctx.fillStyle = "#fafafa";
      ctx.font = "11px monospace";
      ctx.fillText(hovered.heading.slice(0, 60), hovered.x + 8, hovered.y! - 8);
    }

    ctx.restore();
  }, []);

  // Set up simulation
  useEffect(() => {
    if (nodes.length === 0) return;

    const ns: GraphNode[] = nodes.map((n) => ({ ...n }));
    const es: GraphEdge[] = edges.map((e) => ({ ...e }));
    nodesRef.current = ns;
    edgesRef.current = es;

    const sim = forceSimulation<GraphNode>(ns)
      .force("link", forceLink<GraphNode, GraphEdge>(es).id((d) => d.id).distance((e) => 80 * (1 - e.weight)).strength((e) => e.weight))
      .force("charge", forceManyBody().strength(-60))
      .force("center", forceCenter(0, 0))
      .force("collide", forceCollide(8))
      .on("tick", draw);

    simRef.current = sim;
    return () => { sim.stop(); };
  }, [nodes, edges, draw]);

  // Resize
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

  // Find node under cursor
  const hitTest = useCallback((cx: number, cy: number): GraphNode | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const { x: tx, y: ty, k } = transformRef.current;
    const mx = (cx - canvas.width / 2 - tx) / k;
    const my = (cy - canvas.height / 2 - ty) / k;
    for (const n of nodesRef.current) {
      if (n.x == null) continue;
      const dx = n.x - mx, dy = n.y! - my;
      if (dx * dx + dy * dy < 64) return n;
    }
    return null;
  }, []);

  // Mouse interaction
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
        dragRef.current = { node: hit, startX: cx, startY: cy };
        simRef.current?.alphaTarget(0.3).restart();
        hit.fx = hit.x;
        hit.fy = hit.y;
        onSelect(hit);
      } else {
        isPanning = true;
        panStart = { x: e.clientX, y: e.clientY, tx: transformRef.current.x, ty: transformRef.current.y };
      }
    };
    const onMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
      if (dragRef.current.node) {
        const { k } = transformRef.current;
        dragRef.current.node.fx = (cx - canvas.width / 2 - transformRef.current.x) / k;
        dragRef.current.node.fy = (cy - canvas.height / 2 - transformRef.current.y) / k;
      } else if (isPanning) {
        transformRef.current.x = panStart.tx + (e.clientX - panStart.x);
        transformRef.current.y = panStart.ty + (e.clientY - panStart.y);
        draw();
      } else {
        const hit = hitTest(cx, cy);
        hoveredRef.current = hit;
        canvas.style.cursor = hit ? "pointer" : "grab";
        draw();
      }
    };
    const onMouseUp = () => {
      if (dragRef.current.node) {
        simRef.current?.alphaTarget(0);
        dragRef.current.node.fx = null;
        dragRef.current.node.fy = null;
        dragRef.current.node = null;
      }
      isPanning = false;
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      transformRef.current.k = Math.max(0.1, Math.min(10, transformRef.current.k * factor));
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
  }, [hitTest, draw, onSelect]);

  return <canvas ref={canvasRef} className="w-full h-full" />;
}
