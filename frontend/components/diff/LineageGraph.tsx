"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type Simulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";
import { cn } from "@/lib/utils";

// ── Types ────────────────────────────────────────────────────────────────────

export interface LineageNode {
  id: string;
  label: string;
  kind: "input" | "modified" | "computed";
  layer: number;
  data_steps: string[];
}

export interface LineageEdge {
  source: string;
  target: string;
  kind: "assigns" | "conditions" | string;
  data_step: string;
  expr: string;
}

export interface NodeOverlay {
  v2?: unknown;
  v3?: unknown;
  isNumeric?: boolean;
  delta?: number;            // numeric V3 - V2
  contributionGrad?: number; // gradient φ
  contributionShap?: number; // shapley φ
  isBranchFlipVar?: boolean;
  isTarget?: boolean;
  excluded?: boolean;
  agentHighlight?: boolean;  // from agent's lineage_highlight
  codeChanged?: boolean;     // produced or read by a step that differs V2 vs V3
}

export interface BranchFlipEdge {
  /** identifies a predicate that flipped */
  data_step: string;
  vars_in_condition: string[];
}

interface SimNode extends SimulationNodeDatum {
  id: string;
  label: string;
  kind: string;
  layer: number;
  data_steps: string[];
  overlay?: NodeOverlay;
}

interface SimLink extends SimulationLinkDatum<SimNode> {
  source: string | SimNode;
  target: string | SimNode;
  kind: string;
  data_step: string;
  expr: string;
  flow?: number;           // |contribution| flowing through this edge
  flipped?: boolean;       // belongs to a flipped branch predicate
}

// ── Component ────────────────────────────────────────────────────────────────

interface Props {
  nodes: LineageNode[];
  edges: LineageEdge[];
  overlays?: Record<string, NodeOverlay>;
  branchFlips?: BranchFlipEdge[];
  target?: string;
  /** Hop distance from the trace root (0 = target). Used for depth labels. */
  depthMap?: Record<string, number>;
  /** Single-click a node to re-root the backward trace */
  onTraceTarget?: (field: string) => void;
  width?: number;
  height?: number;
  /** Triggers a fresh simulation when this number changes */
  resetSeed?: string | number;
  className?: string;
}

export default function LineageGraph({
  nodes,
  edges,
  overlays = {},
  branchFlips = [],
  target,
  depthMap = {},
  onTraceTarget,
  width = 800,
  height = 560,
  resetSeed = 0,
  className,
}: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const simRef = useRef<Simulation<SimNode, SimLink> | null>(null);
  const [tick, setTick] = useState(0); // re-render trigger
  const [hovered, setHovered] = useState<string | null>(null);
  const [pinned, setPinned] = useState<Set<string>>(new Set());
  const [transform, setTransform] = useState<{ x: number; y: number; k: number }>(
    { x: 0, y: 0, k: 1 },
  );

  // Build the SimNode / SimLink arrays — re-create only when the inputs change
  const data = useMemo(() => {
    const targetU = target?.toUpperCase();
    const flipVars = new Set<string>();
    branchFlips.forEach((bf) => bf.vars_in_condition.forEach((v) => flipVars.add(v.toUpperCase())));

    const simNodes: SimNode[] = nodes.map((n) => {
      const id = n.id.toUpperCase();
      const ov = overlays[id] || overlays[n.id] || {};
      return {
        id,
        label: n.label,
        kind: n.kind,
        layer: n.layer,
        data_steps: n.data_steps || [],
        overlay: {
          ...ov,
          isBranchFlipVar: ov.isBranchFlipVar ?? flipVars.has(id),
          isTarget: ov.isTarget ?? (targetU === id),
        },
      };
    });

    // Calculate per-edge flow (|contribution|) from upstream node's attribution
    const overlayContrib = (id: string) => {
      const o = overlays[id] || overlays[id.toLowerCase()];
      if (!o) return 0;
      return Math.abs(o.contributionGrad ?? o.contributionShap ?? 0);
    };

    // Identify flipped predicate edges: kind=='conditions' and source is a flip var
    const flipKey = new Set<string>();
    branchFlips.forEach((bf) => {
      bf.vars_in_condition.forEach((v) =>
        flipKey.add(`${v.toUpperCase()}::${bf.data_step}`),
      );
    });

    const simLinks: SimLink[] = edges.map((e) => {
      const flow = overlayContrib(e.source);
      const flipped = e.kind === "conditions"
        && flipKey.has(`${e.source.toUpperCase()}::${e.data_step}`);
      return {
        source: e.source.toUpperCase(),
        target: e.target.toUpperCase(),
        kind: e.kind,
        data_step: e.data_step,
        expr: e.expr,
        flow,
        flipped,
      };
    });

    return { simNodes, simLinks };
  }, [nodes, edges, overlays, branchFlips, target]);

  // When hovering a node, highlight it and every upstream ancestor.
  const highlightSet = useMemo(() => {
    if (!hovered) return null;
    const preds: Record<string, string[]> = {};
    for (const e of edges) {
      const t = e.target.toUpperCase();
      preds[t] ??= [];
      preds[t].push(e.source.toUpperCase());
    }
    const out = new Set<string>([hovered]);
    const stack = [hovered];
    while (stack.length) {
      const node = stack.pop()!;
      for (const src of preds[node] ?? []) {
        if (!out.has(src)) {
          out.add(src);
          stack.push(src);
        }
      }
    }
    return out;
  }, [hovered, edges]);

  // ── Simulation ────────────────────────────────────────────────────────────

  useEffect(() => {
    const { simNodes, simLinks } = data;
    if (simNodes.length === 0) return;

    // Compute layer extents for the X-axis force
    const layers = simNodes.map((n) => n.layer);
    const maxLayer = Math.max(0, ...layers);

    const sim = forceSimulation<SimNode, SimLink>(simNodes)
      .force(
        "link",
        forceLink<SimNode, SimLink>(simLinks)
          .id((n: SimNode) => n.id)
          .distance((l) => (l.flow && l.flow > 0 ? 90 : 130))
          .strength(0.6),
      )
      .force("charge", forceManyBody<SimNode>().strength(-260))
      .force("center", forceCenter(width / 2, height / 2))
      .force("collide", forceCollide<SimNode>().radius(34).strength(0.9))
      .force(
        "x",
        forceX<SimNode>((n) => {
          // Align by layer: inputs left, computed right; target far right
          const t = maxLayer === 0 ? 0.5 : n.layer / maxLayer;
          if (n.overlay?.isTarget) return width * 0.85;
          return 0.15 * width + 0.7 * width * t;
        }).strength(0.18),
      )
      .force("y", forceY<SimNode>(height / 2).strength(0.06))
      .alphaDecay(0.04)
      .alpha(1.0);

    sim.on("tick", () => setTick((t) => t + 1));
    simRef.current = sim;
    return () => {
      sim.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, width, height, resetSeed]);

  // Reheat on overlay change so the layout responds to fresh attributions
  useEffect(() => {
    if (simRef.current) {
      simRef.current.alpha(0.6).restart();
    }
  }, [overlays, branchFlips]);

  // ── Pan / zoom ────────────────────────────────────────────────────────────

  const isPanning = useRef(false);
  const panStart = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);

  const onWheel = (e: React.WheelEvent<SVGSVGElement>) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    setTransform((t) => {
      const newK = Math.max(0.3, Math.min(3, t.k * factor));
      // Zoom around the cursor
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect) return { ...t, k: newK };
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const k = newK / t.k;
      const nx = mx - (mx - t.x) * k;
      const ny = my - (my - t.y) * k;
      return { x: nx, y: ny, k: newK };
    });
  };

  const onMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    if ((e.target as Element).closest("[data-node]")) return; // node drag handled below
    isPanning.current = true;
    panStart.current = {
      x: e.clientX, y: e.clientY,
      tx: transform.x, ty: transform.y,
    };
  };

  const onMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!isPanning.current || !panStart.current) return;
    setTransform((t) => ({
      ...t,
      x: panStart.current!.tx + (e.clientX - panStart.current!.x),
      y: panStart.current!.ty + (e.clientY - panStart.current!.y),
    }));
  };

  const onMouseUpOrLeave = () => {
    isPanning.current = false;
    panStart.current = null;
  };

  // ── Node drag ─────────────────────────────────────────────────────────────

  const dragging = useRef<{ id: string; pointerId: number } | null>(null);

  const startDrag = (e: React.PointerEvent, n: SimNode) => {
    e.stopPropagation();
    dragging.current = { id: n.id, pointerId: e.pointerId };
    (e.target as Element).setPointerCapture?.(e.pointerId);
    if (simRef.current) simRef.current.alphaTarget(0.3).restart();
    n.fx = n.x;
    n.fy = n.y;
  };

  const onDrag = (e: React.PointerEvent) => {
    if (!dragging.current || !simRef.current) return;
    const node = simRef.current.nodes().find((n) => n.id === dragging.current!.id);
    if (!node) return;
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = (e.clientX - rect.left - transform.x) / transform.k;
    const y = (e.clientY - rect.top - transform.y) / transform.k;
    node.fx = x;
    node.fy = y;
  };

  const endDrag = (e: React.PointerEvent, n: SimNode) => {
    if (!dragging.current) return;
    const isPin = pinned.has(n.id);
    if (!isPin) {
      n.fx = null;
      n.fy = null;
    }
    if (simRef.current) simRef.current.alphaTarget(0);
    dragging.current = null;
  };

  const togglePin = (n: SimNode) => {
    setPinned((prev) => {
      const next = new Set(prev);
      if (next.has(n.id)) {
        next.delete(n.id);
        n.fx = null;
        n.fy = null;
      } else {
        next.add(n.id);
        n.fx = n.x;
        n.fy = n.y;
      }
      return next;
    });
  };

  // ── Visual helpers ────────────────────────────────────────────────────────

  const maxFlow = useMemo(() => {
    let m = 0;
    for (const l of data.simLinks) {
      m = Math.max(m, l.flow ?? 0);
    }
    return m;
  }, [data.simLinks]);

  const maxContrib = useMemo(() => {
    let m = 0;
    for (const n of data.simNodes) {
      const v = Math.abs(
        n.overlay?.contributionGrad ?? n.overlay?.contributionShap ?? 0,
      );
      m = Math.max(m, v);
    }
    return m;
  }, [data.simNodes]);

  const nodeRadius = (n: SimNode) => {
    const c = Math.abs(n.overlay?.contributionGrad ?? n.overlay?.contributionShap ?? 0);
    const base = n.overlay?.isTarget ? 22 : 14;
    if (maxContrib === 0) return base;
    return base + (c / maxContrib) * 14;
  };

  const nodeFill = (n: SimNode) => {
    if (n.overlay?.isTarget) return "url(#targetGradient)";
    const c = n.overlay?.contributionGrad ?? n.overlay?.contributionShap ?? 0;
    if (Math.abs(c) > 1e-9) {
      // green positive, rose negative
      return c > 0 ? "rgba(16,185,129,0.85)" : "rgba(244,63,94,0.85)";
    }
    if (n.overlay?.delta !== undefined && Math.abs(n.overlay.delta) > 1e-9) {
      return "rgba(245,158,11,0.55)"; // amber: changed but not contributing yet
    }
    if (n.overlay?.v2 !== undefined && n.overlay?.v3 !== undefined && n.overlay.v2 !== n.overlay.v3) {
      return "rgba(245,158,11,0.55)"; // categorical change
    }
    if (n.kind === "input") return "rgba(120,130,150,0.55)";
    if (n.kind === "modified") return "rgba(139,92,246,0.65)";
    return "rgba(99,102,241,0.65)";
  };

  const nodeStroke = (n: SimNode) => {
    if (hovered === n.id) return "#fff";
    if (n.overlay?.codeChanged) return "rgba(217,70,239,0.95)";  // fuchsia
    if (n.overlay?.agentHighlight) return "rgba(168,85,247,0.95)"; // violet
    if (n.overlay?.isBranchFlipVar) return "rgba(244,63,94,0.95)";
    if (n.overlay?.isTarget) return "rgba(56,189,248,0.95)";
    return "rgba(255,255,255,0.18)";
  };

  const linkStroke = (l: SimLink) => {
    if (l.flipped) return "rgba(244,63,94,0.85)";
    if (l.flow && maxFlow > 0) {
      const t = l.flow / maxFlow;
      // amber → emerald gradient
      const r = Math.round(245 - 160 * t);
      const g = Math.round(158 + 27 * t);
      const b = Math.round(11 + 118 * t);
      return `rgba(${r},${g},${b},${0.45 + 0.5 * t})`;
    }
    return l.kind === "conditions" ? "rgba(160,160,170,0.25)" : "rgba(160,160,170,0.45)";
  };

  const linkWidth = (l: SimLink) => {
    if (maxFlow > 0 && l.flow) {
      return 1 + 4 * (l.flow / maxFlow);
    }
    return l.kind === "conditions" ? 1 : 1.5;
  };

  const linkDash = (l: SimLink) => (l.kind === "conditions" || l.flipped ? "4 3" : undefined);

  // ── Tooltip ───────────────────────────────────────────────────────────────

  const hoveredNode = useMemo(() => {
    if (!hovered) return null;
    return data.simNodes.find((n) => n.id === hovered) || null;
  }, [hovered, data.simNodes]);

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className={cn("relative w-full h-full select-none", className)}>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        onWheel={onWheel}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUpOrLeave}
        onMouseLeave={onMouseUpOrLeave}
        onPointerMove={onDrag}
        className="bg-[#0b1020] rounded-md cursor-grab active:cursor-grabbing"
      >
        <defs>
          {/* Radial highlight for the target node */}
          <radialGradient id="targetGradient" cx="50%" cy="50%">
            <stop offset="0%" stopColor="rgba(56,189,248,1)" />
            <stop offset="60%" stopColor="rgba(59,130,246,0.85)" />
            <stop offset="100%" stopColor="rgba(30,64,175,0.4)" />
          </radialGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          {/* Subtle dotted background pattern */}
          <pattern id="dot-grid" x="0" y="0" width="22" height="22" patternUnits="userSpaceOnUse">
            <circle cx="1" cy="1" r="1" fill="rgba(255,255,255,0.04)" />
          </pattern>
          {/* Arrowhead */}
          <marker
            id="arrow"
            viewBox="0 0 10 10"
            refX="10" refY="5"
            markerWidth="5" markerHeight="5"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(200,200,200,0.5)" />
          </marker>
        </defs>

        <rect width={width} height={height} fill="url(#dot-grid)" />

        <g transform={`translate(${transform.x},${transform.y}) scale(${transform.k})`}>
          {/* Edges */}
          {data.simLinks.map((l, i) => {
            const s = l.source as SimNode;
            const t = l.target as SimNode;
            if (!s.x || !s.y || !t.x || !t.y) return null;
            const dx = t.x - s.x;
            const dy = t.y - s.y;
            const len = Math.sqrt(dx * dx + dy * dy);
            if (len === 0) return null;
            const r2 = nodeRadius(t);
            const ex = t.x - (dx / len) * (r2 + 4);
            const ey = t.y - (dy / len) * (r2 + 4);
            const stroke = linkStroke(l);
            const dimmed = highlightSet
              ? !highlightSet.has(s.id) || !highlightSet.has(t.id)
              : hovered && hovered !== s.id && hovered !== t.id;
            return (
              <g key={`l-${i}`} opacity={dimmed ? 0.2 : 1}>
                <line
                  x1={s.x} y1={s.y}
                  x2={ex} y2={ey}
                  stroke={stroke}
                  strokeWidth={linkWidth(l)}
                  strokeDasharray={linkDash(l)}
                  markerEnd="url(#arrow)"
                />
                {l.flow && maxFlow > 0 && (
                  <FlowAnimation
                    x1={s.x!} y1={s.y!} x2={ex} y2={ey}
                    intensity={l.flow / maxFlow}
                    color={stroke}
                  />
                )}
              </g>
            );
          })}

          {/* Nodes */}
          {data.simNodes.map((n) => {
            if (n.x === undefined || n.y === undefined) return null;
            const r = nodeRadius(n);
            const isHover = hovered === n.id;
            const isPinned = pinned.has(n.id);
            const dimmed = highlightSet ? !highlightSet.has(n.id) : false;
            const hop = depthMap[n.id];
            return (
              <g
                key={n.id}
                data-node
                transform={`translate(${n.x},${n.y})`}
                opacity={dimmed ? 0.22 : 1}
                onPointerDown={(e) => startDrag(e, n)}
                onPointerUp={(e) => endDrag(e, n)}
                onMouseEnter={() => setHovered(n.id)}
                onMouseLeave={() => setHovered(null)}
                onClick={() => onTraceTarget?.(n.id)}
                onDoubleClick={() => togglePin(n)}
                style={{ cursor: onTraceTarget ? "pointer" : "grab" }}
              >
                {n.overlay?.isTarget && (
                  <circle r={r + 6} fill="rgba(56,189,248,0.18)" />
                )}
                {n.overlay?.codeChanged && !n.overlay?.isTarget && (
                  <circle r={r + 5} fill="rgba(217,70,239,0.20)" />
                )}
                {n.overlay?.agentHighlight && !n.overlay?.isTarget && !n.overlay?.codeChanged && (
                  <circle r={r + 5} fill="rgba(168,85,247,0.18)" />
                )}
                <circle
                  r={r}
                  fill={nodeFill(n)}
                  stroke={nodeStroke(n)}
                  strokeWidth={isHover ? 2.5 : 1.4}
                  style={n.overlay?.isTarget || isHover ? { filter: "url(#glow)" } : undefined}
                />
                {n.overlay?.excluded && (
                  <circle r={r + 1} fill="none" stroke="rgba(244,63,94,0.85)" strokeWidth={1} strokeDasharray="3 2" />
                )}
                {isPinned && (
                  <circle r={2} cx={r - 4} cy={-r + 4} fill="#fff" />
                )}
                <text
                  textAnchor="middle"
                  y={r + 12}
                  fill="rgba(230,235,245,0.95)"
                  fontSize={11}
                  fontFamily="ui-monospace,SFMono-Regular,Menlo,monospace"
                  style={{ pointerEvents: "none" }}
                >
                  {n.label}
                </text>
                {hop !== undefined && hop > 0 && (
                  <text
                    textAnchor="middle"
                    y={-(r + 6)}
                    fill="rgba(148,163,184,0.85)"
                    fontSize={9}
                    fontFamily="ui-monospace,SFMono-Regular,Menlo,monospace"
                    style={{ pointerEvents: "none" }}
                  >
                    d{hop}
                  </text>
                )}
              </g>
            );
          })}
        </g>

        {/* Floating tooltip */}
        {hoveredNode && (
          <Tooltip
            node={hoveredNode}
            transform={transform}
            width={width}
            height={height}
          />
        )}
      </svg>

      <Legend hasOverlay={Object.keys(overlays).length > 0} />
    </div>
  );
}

// ── Flow animation (dashed line that "moves" along the edge) ─────────────────

function FlowAnimation({
  x1, y1, x2, y2, intensity, color,
}: {
  x1: number; y1: number; x2: number; y2: number; intensity: number; color: string;
}) {
  const speed = 1.5 - intensity * 0.7; // hotter → faster
  return (
    <line
      x1={x1} y1={y1} x2={x2} y2={y2}
      stroke={color}
      strokeWidth={1.5 + intensity * 2.5}
      strokeDasharray="6 10"
      strokeOpacity={0.8}
      style={{
        animation: `flowmove ${speed}s linear infinite`,
      }}
    />
  );
}

// ── Tooltip ──────────────────────────────────────────────────────────────────

function Tooltip({
  node,
  transform,
  width,
  height,
}: {
  node: SimNode;
  transform: { x: number; y: number; k: number };
  width: number;
  height: number;
}) {
  if (node.x === undefined || node.y === undefined) return null;
  const sx = node.x * transform.k + transform.x;
  const sy = node.y * transform.k + transform.y;
  // Anchor tooltip below node, clamped within the svg
  const w = 240;
  const h = computedTooltipHeight(node);
  let tx = sx + 18;
  let ty = sy - h / 2;
  if (tx + w > width - 8) tx = sx - w - 18;
  if (ty < 8) ty = 8;
  if (ty + h > height - 8) ty = height - h - 8;

  const ov = node.overlay || {};
  const c = ov.contributionGrad ?? ov.contributionShap;

  return (
    <foreignObject x={tx} y={ty} width={w} height={h}>
      <div className="rounded-md border border-border bg-card/95 backdrop-blur-sm p-2 text-xs space-y-1 shadow-xl">
        <div className="font-mono font-semibold text-foreground flex items-center gap-2">
          {node.label}
          {ov.isTarget && <span className="text-sky-400">(target)</span>}
          {ov.isBranchFlipVar && <span className="text-rose-400">branch flip</span>}
          {ov.excluded && <span className="text-rose-400">excluded</span>}
        </div>
        <div className="text-muted-foreground">
          <span className="capitalize">{node.kind}</span>
          {node.data_steps.length > 0 && (
            <> · <span className="font-mono">{node.data_steps.join(", ")}</span></>
          )}
        </div>
        {(ov.v2 !== undefined || ov.v3 !== undefined) && (
          <div className="grid grid-cols-2 gap-2 mt-1">
            <div>
              <span className="text-muted-foreground">V2: </span>
              <span className="font-mono text-sky-300">{fmt(ov.v2)}</span>
            </div>
            <div>
              <span className="text-muted-foreground">V3: </span>
              <span className="font-mono text-amber-300">{fmt(ov.v3)}</span>
            </div>
          </div>
        )}
        {ov.delta !== undefined && Math.abs(ov.delta) > 0 && (
          <div>
            <span className="text-muted-foreground">Δ: </span>
            <span className="font-mono">{ov.delta >= 0 ? "+" : ""}{ov.delta.toPrecision(4)}</span>
          </div>
        )}
        {c !== undefined && Math.abs(c) > 1e-9 && (
          <div>
            <span className="text-muted-foreground">contribution: </span>
            <span className={cn("font-mono",
              c > 0 ? "text-emerald-400" : "text-rose-400")}>
              {c >= 0 ? "+" : ""}{c.toPrecision(4)}
            </span>
          </div>
        )}
      </div>
    </foreignObject>
  );
}

function computedTooltipHeight(n: SimNode) {
  const ov = n.overlay || {};
  let h = 56;
  if (ov.v2 !== undefined || ov.v3 !== undefined) h += 22;
  if (ov.delta !== undefined && Math.abs(ov.delta) > 0) h += 18;
  const c = ov.contributionGrad ?? ov.contributionShap;
  if (c !== undefined && Math.abs(c) > 1e-9) h += 18;
  return h;
}

function fmt(v: unknown): string {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return v.toString();
    return v.toPrecision(6);
  }
  return String(v);
}

// ── Legend ───────────────────────────────────────────────────────────────────

function Legend({ hasOverlay }: { hasOverlay: boolean }) {
  const [open, setOpen] = useState(true);
  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="absolute bottom-2 right-2 rounded-md border border-border bg-card/85 backdrop-blur-sm px-2 py-1 text-[10px] hover:bg-card"
        aria-label="Show legend"
      >
        legend
      </button>
    );
  }
  return (
    <div className="absolute bottom-2 right-2 rounded-md border border-border bg-card/85 backdrop-blur-sm p-2 text-[10px] space-y-1">
      <div className="flex items-center justify-between gap-3 mb-0.5">
        <span className="uppercase tracking-wider text-muted-foreground">legend</span>
        <button
          onClick={() => setOpen(false)}
          className="text-muted-foreground hover:text-foreground"
          aria-label="Hide legend"
        >
          ×
        </button>
      </div>
      <Item dot="rgba(56,189,248,0.9)" label="target" />
      <Item dot="rgba(16,185,129,0.85)" label="positive contribution" disabled={!hasOverlay} />
      <Item dot="rgba(244,63,94,0.85)" label="negative contribution" disabled={!hasOverlay} />
      <Item dot="rgba(245,158,11,0.65)" label="changed input" disabled={!hasOverlay} />
      <Item dot="rgba(139,92,246,0.65)" label="modified field" />
      <Item dot="rgba(120,130,150,0.6)" label="unchanged input" />
      <Item ring="rgba(244,63,94,0.95)" label="branch-flip predicate" />
      <Item ring="rgba(217,70,239,0.95)" label="code changed (V2→V3)" />
      <Item ring="rgba(168,85,247,0.95)" label="agent highlight" />
      <div className="pt-1 mt-1 border-t border-border/50 text-muted-foreground/70 leading-tight">
        scroll = zoom · drag canvas = pan<br/>
        drag node = move · double-click = pin
      </div>
    </div>
  );
}

function Item({ dot, ring, label, disabled }: { dot?: string; ring?: string; label: string; disabled?: boolean }) {
  return (
    <div className={cn("flex items-center gap-1.5", disabled && "opacity-40")}>
      <span
        className="inline-block w-2.5 h-2.5 rounded-full"
        style={{
          background: ring ? "transparent" : dot,
          border: ring ? `1.5px solid ${ring}` : undefined,
        }}
      />
      <span className="text-foreground/80">{label}</span>
    </div>
  );
}
