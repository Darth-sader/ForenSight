'use client';

import { useEffect, useRef, useState } from 'react';
import type { GraphData } from '@/lib/api';

interface Props {
  graph?: GraphData;
  nodes?: any[];
  edges?: any[];
  paths?: string[][];
  height?: number;
  threshold?: number;
  onNodeClick?: (id: string) => void;
}

interface Node {
  id: string;
  index: number;
  anomaly?: number;
  degree?: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  fixed?: boolean;
}

export default function GraphCanvas({
  graph,
  nodes: inputNodes,
  edges: inputEdges,
  paths = [],
  height = 540,
  threshold = 0,
  onNodeClick,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<Node[]>([]);
  const edgesRef = useRef<{ s: Node; t: Node; p_next: number; emb_sim: number }[]>([]);
  
  // viewRef supports pan (x, y), zoom (k), and rotation in radians (rot)
  const viewRef = useRef({ x: 0, y: 0, k: 1, rot: 0 });
  const [rotationDeg, setRotationDeg] = useState(0);

  const dragRef = useRef<{
    node?: Node;
    panning?: boolean;
    rotating?: boolean;
    lx: number;
    ly: number;
    startAngle?: number;
    startRot?: number;
    moved?: boolean;
  }>({ lx: 0, ly: 0 });

  const [hover, setHover] = useState<{ n: Node; mx: number; my: number } | null>(null);
  const [physicsActive, setPhysicsActive] = useState(false);
  const pathSet = useRef<Set<string>>(new Set());

  // Build graph state with a stable initial golden-spiral distribution
  useEffect(() => {
    const rawNodes = graph?.nodes || inputNodes || [];
    const rawEdges = graph?.edges || inputEdges || [];

    const rng = mulberry(42);
    const nodes: Node[] = rawNodes.slice(0, 400).map((n: any, i: number) => {
      const a = i * 2.39996; // Golden angle spiral layout
      const r = 50 + 16 * Math.sqrt(i + 1);
      return {
        ...n,
        id: n.id || `frag_${i}`,
        index: n.index ?? i,
        anomaly: n.anomaly ?? 0,
        degree: n.degree ?? 0,
        x: Math.cos(a) * r + (rng() - 0.5) * 20,
        y: Math.sin(a) * r + (rng() - 0.5) * 20,
        vx: 0,
        vy: 0,
        fixed: false,
      };
    });

    const byIndex = new Map(nodes.map((n) => [n.index, n]));
    const edges: { s: Node; t: Node; p_next: number; emb_sim: number }[] = [];
    for (const e of rawEdges) {
      const s = byIndex.get(e.source);
      const t = byIndex.get(e.target);
      if (s && t && (e.p_next == null || e.p_next >= threshold)) {
        edges.push({ s, t, p_next: e.p_next ?? 1.0, emb_sim: e.emb_sim ?? 1.0 });
      }
    }

    nodesRef.current = nodes;
    edgesRef.current = edges;
    pathSet.current = new Set(paths.flat());
    viewRef.current = { x: 0, y: 0, k: 1, rot: 0 };
    setRotationDeg(0);
  }, [graph, inputNodes, inputEdges, threshold, paths]);

  // Physics simulation + rendering loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    let raf = 0;
    let running = true;
    let alpha = 0.3;

    const step = () => {
      if (!running) return;
      const nodes = nodesRef.current;
      const edges = edgesRef.current;
      const W = canvas.width, H = canvas.height;

      // Damped force-directed physics (active only when user toggles physics)
      if (physicsActive && alpha > 0.01) {
        for (let i = 0; i < nodes.length; i++) {
          for (let j = i + 1; j < nodes.length; j++) {
            const a = nodes[i], b = nodes[j];
            if (a.fixed && b.fixed) continue;
            let dx = a.x - b.x, dy = a.y - b.y;
            let d2 = dx * dx + dy * dy;
            if (d2 < 1) { d2 = 1; dx = 1; dy = 0; }
            const f = (600 * alpha) / d2;
            const d = Math.sqrt(d2);
            const fx = (dx / d) * f, fy = (dy / d) * f;
            if (!a.fixed) { a.vx += fx; a.vy += fy; }
            if (!b.fixed) { b.vx -= fx; b.vy -= fy; }
          }
        }
        for (const e of edges) {
          const dx = e.t.x - e.s.x, dy = e.t.y - e.s.y;
          const d = Math.sqrt(dx * dx + dy * dy) || 1;
          const f = (d - 60) * 0.005 * alpha;
          const fx = (dx / d) * f, fy = (dy / d) * f;
          if (!e.s.fixed) { e.s.vx += fx; e.s.vy += fy; }
          if (!e.t.fixed) { e.t.vx -= fx; e.t.vy -= fy; }
        }
        for (const n of nodes) {
          if (!n.fixed) {
            n.vx -= n.x * 0.001 * alpha;
            n.vy -= n.y * 0.001 * alpha;
            n.x += n.vx * 0.4;
            n.y += n.vy * 0.4;
          }
          n.vx *= 0.75;
          n.vy *= 0.75;
        }
        alpha *= 0.99;
      }

      // Clear & render canvas frame
      ctx.fillStyle = '#070b12';
      ctx.fillRect(0, 0, W, H);
      const view = viewRef.current;

      ctx.save();
      // Apply pan, zoom, and rotation transformations
      ctx.translate(W / 2 + view.x, H / 2 + view.y);
      ctx.scale(view.k, view.k);
      ctx.rotate(view.rot);

      // Render graph edges
      for (const e of edges) {
        const hot = pathSet.current.has(e.s.id) && pathSet.current.has(e.t.id);
        const t = Math.max(0, Math.min(1, e.p_next));
        const col = t > 0.66 ? '#f4e04d' : t > 0.33 ? '#35b779' : '#31688e';
        ctx.strokeStyle = hot ? '#22d3ee' : col;
        ctx.globalAlpha = hot ? 0.95 : 0.2 + 0.5 * t;
        ctx.lineWidth = hot ? 2.4 : 0.7 + t;
        ctx.beginPath();
        ctx.moveTo(e.s.x, e.s.y);
        ctx.lineTo(e.t.x, e.t.y);
        ctx.stroke();
      }

      // Render graph nodes
      ctx.globalAlpha = 1;
      for (const n of nodes) {
        const r = 4 + Math.min(6, Math.sqrt(n.degree || 1));
        const hot = pathSet.current.has(n.id);
        const anomalyVal = n.anomaly ?? 0;
        const g = Math.round(255 * Math.min(1, anomalyVal));
        ctx.beginPath();
        ctx.arc(n.x, n.y, hot ? r + 3 : r, 0, 6.283);
        ctx.fillStyle = hot ? '#22d3ee' : `rgb(${g}, ${255 - g}, 60)`;
        ctx.fill();

        if (n.fixed) {
          ctx.strokeStyle = '#38bdf8';
          ctx.lineWidth = 1.8;
          ctx.stroke();
        } else if (anomalyVal > 0.7) {
          ctx.strokeStyle = '#f87171';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }
      ctx.restore();

      // Render node hover tooltip
      if (hover) {
        const anomalyVal = (hover.n.anomaly ?? 0).toFixed(2);
        const degreeVal = hover.n.degree ?? 0;
        const label = `${hover.n.id} · anomaly ${anomalyVal} · deg ${degreeVal}`;
        ctx.font = '11px Consolas, monospace';
        const w = ctx.measureText(label).width + 16;
        ctx.fillStyle = '#101a2b';
        ctx.strokeStyle = '#1c2a42';
        ctx.beginPath();
        if (typeof (ctx as any).roundRect === 'function') {
          (ctx as any).roundRect(hover.mx + 12, hover.my - 12, w, 24, 6);
        } else {
          ctx.rect(hover.mx + 12, hover.my - 12, w, 24);
        }
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = '#d7e1ee';
        ctx.fillText(label, hover.mx + 20, hover.my + 4);
      }

      raf = requestAnimationFrame(step);
    };

    raf = requestAnimationFrame(step);
    return () => {
      running = false;
      cancelAnimationFrame(raf);
    };
  }, [hover, graph, physicsActive]);

  // Convert canvas screen coords (mx, my) to world space coords (x, y) taking rotation into account
  const screenToWorld = (mx: number, my: number) => {
    const canvas = canvasRef.current!;
    const W = canvas.width, H = canvas.height;
    const view = viewRef.current;
    
    const dx0 = (mx - W / 2 - view.x) / view.k;
    const dy0 = (my - H / 2 - view.y) / view.k;
    
    const cos = Math.cos(-view.rot);
    const sin = Math.sin(-view.rot);
    
    return {
      x: dx0 * cos - dy0 * sin,
      y: dx0 * sin + dy0 * cos,
    };
  };

  // Helper to locate node under cursor
  const nodeAt = (mx: number, my: number): Node | null => {
    const world = screenToWorld(mx, my);
    for (const n of nodesRef.current) {
      const dx = n.x - world.x, dy = n.y - world.y;
      if (dx * dx + dy * dy < 144) return n;
    }
    return null;
  };

  const handleResetView = () => {
    viewRef.current = { x: 0, y: 0, k: 1, rot: 0 };
    setRotationDeg(0);
  };

  const handleRotateBy = (deltaDeg: number) => {
    const newDeg = (rotationDeg + deltaDeg) % 360;
    setRotationDeg(newDeg);
    viewRef.current.rot = (newDeg * Math.PI) / 180;
  };

  const handleSliderRotation = (deg: number) => {
    setRotationDeg(deg);
    viewRef.current.rot = (deg * Math.PI) / 180;
  };

  const handleUnpinAll = () => {
    for (const n of nodesRef.current) {
      n.fixed = false;
    }
  };

  return (
    <div style={{ position: 'relative', width: '100%', height }}>
      {/* Interactive Control Overlay Bar */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          right: 12,
          zIndex: 10,
          display: 'flex',
          gap: 8,
          alignItems: 'center',
          background: 'rgba(16, 26, 43, 0.90)',
          backdropFilter: 'blur(8px)',
          padding: '6px 12px',
          borderRadius: 8,
          border: '1px solid var(--border)',
          fontSize: 12,
        }}
      >
        <button
          className="btn"
          style={{ padding: '4px 8px', fontSize: 11.5 }}
          onClick={() => setPhysicsActive(!physicsActive)}
          title="Toggle soft physics layout"
        >
          {physicsActive ? '⏸️ Freeze' : '⚡ Soft Physics'}
        </button>

        {/* Rotation Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, paddingLeft: 6, borderLeft: '1px solid var(--border)' }}>
          <span style={{ fontSize: 11, color: 'var(--text-dim)', fontWeight: 600 }}>🔄 Rotation:</span>
          <button
            className="btn"
            style={{ padding: '2px 6px', fontSize: 11 }}
            onClick={() => handleRotateBy(-45)}
            title="Rotate counter-clockwise 45°"
          >
            ↶ 45°
          </button>
          <input
            type="range"
            min="-180"
            max="180"
            value={rotationDeg}
            onChange={(e) => handleSliderRotation(Number(e.target.value))}
            style={{ width: 70, cursor: 'pointer', accentColor: 'var(--accent)' }}
            title="Drag slider or Shift+Mouse-Drag canvas to rotate graph"
          />
          <span style={{ fontFamily: 'monospace', fontSize: 11, minWidth: 32, color: 'var(--accent)' }}>
            {rotationDeg}°
          </span>
          <button
            className="btn"
            style={{ padding: '2px 6px', fontSize: 11 }}
            onClick={() => handleRotateBy(45)}
            title="Rotate clockwise 45°"
          >
            ↷ 45°
          </button>
        </div>

        <button
          className="btn"
          style={{ padding: '4px 8px', fontSize: 11.5 }}
          onClick={handleResetView}
          title="Reset canvas view & rotation"
        >
          🎯 Reset View
        </button>

        <button
          className="btn"
          style={{ padding: '4px 8px', fontSize: 11.5 }}
          onClick={handleUnpinAll}
          title="Unpin dragged nodes"
        >
          📌 Unpin All
        </button>
      </div>

      <div style={{ position: 'absolute', bottom: 12, left: 12, zIndex: 10, fontSize: 11, color: 'var(--text-dim)', background: 'rgba(5,9,16,0.75)', padding: '4px 8px', borderRadius: 4 }}>
        💡 <strong>Controls:</strong> Left-drag node to move · Left-drag canvas to pan · <strong>Shift+Drag or Right-Drag</strong> to rotate · Scroll to zoom
      </div>

      <canvas
        ref={canvasRef}
        width={1200}
        height={height}
        style={{
          width: '100%',
          height,
          borderRadius: 10,
          border: '1px solid var(--border)',
          cursor: 'grab',
        }}
        onContextMenu={(e) => e.preventDefault()} // Prevent context menu so right-drag rotation works cleanly
        onMouseDown={(e) => {
          const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
          const mx = ((e.clientX - rect.left) / rect.width) * 1200;
          const my = ((e.clientY - rect.top) / rect.height) * height;
          
          const isRightClick = e.button === 2;
          const isShiftKey = e.shiftKey;

          if (isRightClick || isShiftKey) {
            // Initiate rotation dragging
            const W = 1200, H = height;
            const dx = mx - (W / 2 + viewRef.current.x);
            const dy = my - (H / 2 + viewRef.current.y);
            const startAngle = Math.atan2(dy, dx);
            dragRef.current = {
              rotating: true,
              lx: mx,
              ly: my,
              startAngle,
              startRot: viewRef.current.rot,
              moved: false,
            };
          } else {
            const n = nodeAt(mx, my);
            dragRef.current = n
              ? { node: n, lx: mx, ly: my, moved: false }
              : { panning: true, lx: mx, ly: my, moved: false };
          }
        }}
        onMouseMove={(e) => {
          const canvas = e.target as HTMLCanvasElement;
          const rect = canvas.getBoundingClientRect();
          const mx = ((e.clientX - rect.left) / rect.width) * 1200;
          const my = ((e.clientY - rect.top) / rect.height) * height;
          const d = dragRef.current;

          if (d.rotating) {
            d.moved = true;
            const W = 1200, H = height;
            const dx = mx - (W / 2 + viewRef.current.x);
            const dy = my - (H / 2 + viewRef.current.y);
            const currentAngle = Math.atan2(dy, dx);
            const angleDelta = currentAngle - (d.startAngle || 0);
            
            let newRot = (d.startRot || 0) + angleDelta;
            viewRef.current.rot = newRot;
            
            let deg = Math.round((newRot * 180) / Math.PI) % 360;
            if (deg > 180) deg -= 360;
            if (deg < -180) deg += 360;
            setRotationDeg(deg);
          } else if (d.node) {
            d.moved = true;
            const world = screenToWorld(mx, my);
            d.node.x = world.x;
            d.node.y = world.y;
            d.node.fixed = true; // Pin node where user dragged it!
          } else if (d.panning) {
            d.moved = true;
            viewRef.current.x += mx - d.lx;
            viewRef.current.y += my - d.ly;
            d.lx = mx;
            d.ly = my;
          } else {
            const n = nodeAt(mx, my);
            setHover(n ? { n, mx, my } : null);
            canvas.style.cursor = n ? 'pointer' : 'grab';
          }
        }}
        onMouseUp={() => {
          dragRef.current = { lx: 0, ly: 0 };
        }}
        onMouseLeave={() => {
          dragRef.current = { lx: 0, ly: 0 };
          setHover(null);
        }}
        onClick={(e) => {
          if (dragRef.current.moved) return;
          const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
          const mx = ((e.clientX - rect.left) / rect.width) * 1200;
          const my = ((e.clientY - rect.top) / rect.height) * height;
          const n = nodeAt(mx, my);
          if (n && onNodeClick) onNodeClick(n.id);
        }}
        onWheel={(e) => {
          e.preventDefault();
          const view = viewRef.current;
          const factor = e.deltaY < 0 ? 1.12 : 0.89;
          view.k = Math.max(0.25, Math.min(6, view.k * factor));
        }}
      />
    </div>
  );
}

function mulberry(seed: number) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
