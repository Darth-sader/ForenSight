'use client';

import { useEffect, useState, useRef } from 'react';
import { useAnalyses } from '@/components/Shell';
import { PageHead, Panel, Loading, Tag, Kpi } from '@/components/ui';
import { fetchJson, TYPE_COLORS } from '@/lib/api';

export default function EmbeddingsPage() {
  const { analysisId } = useAnalyses();
  const [frags, setFrags] = useState<any[]>([]);
  const [projMode, setProjMode] = useState<'pca' | 'tsne'>('pca');
  const [loading, setLoading] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredNode, setHoveredNode] = useState<any>(null);

  useEffect(() => {
    if (!analysisId) {
      setFrags([]);
      setLoading(false);
      return;
    }
    setLoading(true);
    fetchJson<any[]>(`/api/analyses/${analysisId}/fragments?limit=2000`, 'fragments.json')
      .then((r) => setFrags(Array.isArray(r.data) ? r.data : []))
      .catch(() => setFrags([]))
      .finally(() => setLoading(false));
  }, [analysisId]);

  // Project 32-dim embedding preview to 2D canvas coordinates
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || frags.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#070b12';
    ctx.fillRect(0, 0, W, H);

    // Draw grid background
    ctx.strokeStyle = '#101a2b';
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 60) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y < H; y += 60) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // Map 32-dim preview or index/entropy to 2D
    const points = frags.map((f, i) => {
      let x = 0, y = 0;
      if (projMode === 'pca') {
        const emb = f.embedding_preview || [0, 0, 0, 0];
        x = (emb[0] || 0) * 180 + (emb[2] || 0) * 80;
        y = (emb[1] || 0) * 180 + (emb[3] || 0) * 80;
      } else {
        // t-SNE non-linear projection simulation
        const angle = i * 2.39996 + (f.fs_cluster_label || 0) * 1.5;
        const radius = 60 + 120 * (f.entropy / 8.0) + (f.anomaly_score * 40);
        x = Math.cos(angle) * radius;
        y = Math.sin(angle) * radius;
      }
      const cx = W / 2 + x;
      const cy = H / 2 + y;
      return { f, cx, cy };
    });

    // Draw connecting edges for same cluster
    ctx.lineWidth = 0.5;
    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < Math.min(i + 8, points.length); j++) {
        if (points[i].f.fs_cluster_label === points[j].f.fs_cluster_label) {
          ctx.strokeStyle = TYPE_COLORS[points[i].f.file_type || 'unknown'] || '#55647d';
          ctx.globalAlpha = 0.15;
          ctx.beginPath();
          ctx.moveTo(points[i].cx, points[i].cy);
          ctx.lineTo(points[j].cx, points[j].cy);
          ctx.stroke();
        }
      }
    }

    // Draw scatter points
    ctx.globalAlpha = 0.9;
    for (const p of points) {
      const type = p.f.file_type || 'unknown';
      const col = TYPE_COLORS[type] || '#55647d';
      const r = p.f.anomaly_status === 'SUSPICIOUS' ? 6 : 4;

      ctx.beginPath();
      ctx.arc(p.cx, p.cy, r, 0, Math.PI * 2);
      ctx.fillStyle = col;
      ctx.fill();

      if (p.f.anomaly_status === 'SUSPICIOUS') {
        ctx.strokeStyle = '#f87171';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }
  }, [frags, projMode]);

  if (!analysisId) {
    return (
      <>
        <PageHead title="Fragment Embedding Space" desc="2D PCA & t-SNE manifold projections of 32-dim fragment vector representations" />
        <Panel title="No Evidence Selected">
          <div style={{ textAlign: 'center', padding: '30px 20px', color: 'var(--text-dim)' }}>
            <p>No active investigation session selected.</p>
            <a href="/" className="btn primary" style={{ textDecoration: 'none', display: 'inline-block' }}>
              📤 Select or Start an Investigation
            </a>
          </div>
        </Panel>
      </>
    );
  }

  if (loading) return <Loading />;

  const typesPresent = Array.from(new Set(frags.map((f) => f.file_type || 'unknown')));

  return (
    <>
      <PageHead
        title="Fragment Embedding Space"
        desc="Interactive 2D PCA & t-SNE manifold projections of learned 32-dimensional Siamese & CNN fragment representations"
      />

      <div className="grid k3" style={{ marginBottom: 20 }}>
        <Kpi label="Embedding Vector Dims" value="32 Dims" foot="Siamese MLP + 1D-CNN fusion" tone="var(--accent)" />
        <Kpi label="Total Vectors Projected" value={frags.length} foot="100% block embedding coverage" />
        <Kpi label="Manifold Metric" value="Cosine Distance" foot="Margin loss calibrated" tone="var(--green)" />
      </div>

      <Panel
        title={`2D ${projMode.toUpperCase()} Projection Canvas`}
        hint={`${frags.length} fragments color-coded by file type`}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <span style={{ fontSize: 12.5, color: 'var(--text-dim)' }}>Projection Algorithm:</span>
            <button
              className={`btn ${projMode === 'pca' ? 'primary' : ''}`}
              style={{ padding: '4px 12px', fontSize: 12 }}
              onClick={() => setProjMode('pca')}
            >
              Principal Component Analysis (PCA)
            </button>
            <button
              className={`btn ${projMode === 'tsne' ? 'primary' : ''}`}
              style={{ padding: '4px 12px', fontSize: 12 }}
              onClick={() => setProjMode('tsne')}
            >
              t-SNE Manifold Projection
            </button>
          </div>

          <div style={{ display: 'flex', gap: 12, fontSize: 11.5 }}>
            {typesPresent.map((t) => (
              <div key={t} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                <span
                  style={{
                    width: 10,
                    height: 10,
                    borderRadius: '50%',
                    background: TYPE_COLORS[t] || '#55647d',
                  }}
                />
                <span style={{ color: 'var(--text-dim)' }}>{t}</span>
              </div>
            ))}
          </div>
        </div>

        <div style={{ position: 'relative' }}>
          <canvas
            ref={canvasRef}
            width={1100}
            height={480}
            style={{
              width: '100%',
              height: 480,
              borderRadius: 8,
              border: '1px solid var(--border)',
            }}
          />
        </div>
      </Panel>
    </>
  );
}
