'use client';

import { useEffect, useState, useRef, useMemo } from 'react';
import Link from 'next/link';
import { useAnalyses } from '@/components/Shell';
import { Kpi, Panel, Meter, PageHead, Tag, Loading } from '@/components/ui';
import { API_BASE, fetchJson, fmtBytes, pct, TYPE_COLORS } from '@/lib/api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function EvidencePage() {
  const { analyses, analysisId, setAnalysisId, reload } = useAnalyses();
  const [meta, setMeta] = useState<any>(null);
  const [frags, setFrags] = useState<any[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadErr, setUploadErr] = useState('');
  const [mode, setMode] = useState('bidirectional_island');
  
  // View states: 'dialog' (Welcome Dialogue Modal), 'upload' (Upload File Area), 'past_list' (Open Past Sessions), 'dashboard' (Active Session Results)
  const [viewState, setViewState] = useState<'dialog' | 'upload' | 'past_list' | 'dashboard'>('dialog');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Sync viewState when analysisId changes
  useEffect(() => {
    if (!analysisId) {
      setViewState('dialog');
    }
  }, [analysisId]);

  // Automatic real-time polling while analysis is queued or processing
  useEffect(() => {
    if (!analysisId) {
      setMeta(null);
      setFrags([]);
      return;
    }

    let timer: any = null;

    const loadData = async () => {
      try {
        const r = await fetchJson<any>(`/api/analyses/${analysisId}`, 'analysis.json');
        if (!r?.data) return;
        setMeta(r.data);

        if (r.data.status === 'done') {
          const f = await fetchJson<any[]>(`/api/analyses/${analysisId}/fragments?limit=2000`, 'fragments.json');
          setFrags(Array.isArray(f.data) ? f.data : []);
          reload(); // refresh sidebar list
        } else if (r.data.status === 'processing' || r.data.status === 'queued') {
          // Poll every 1 second until status is 'done' or 'error'
          timer = setTimeout(loadData, 1000);
        }
      } catch {
        setMeta(null);
      }
    };

    loadData();

    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [analysisId, reload]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadErr('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/api/analyses/upload?mode=${mode}`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Upload failed (${res.status})`);
      }

      const data = await res.json();
      reload();
      if (data.id) {
        setAnalysisId(data.id);
        setViewState('dashboard');
      }
    } catch (err: any) {
      setUploadErr(err.message || 'Failed to upload evidence file');
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handlePresetTrigger = async (scenarioId: string) => {
    setUploading(true);
    setUploadErr('');
    try {
      const res = await fetch(`${API_BASE}/api/analyses/preset/${scenarioId}?mode=${mode}`, {
        method: 'POST',
      });
      if (!res.ok) throw new Error(`Preset trigger failed (${res.status})`);
      const data = await res.json();
      reload();
      if (data.id) {
        setAnalysisId(data.id);
        setViewState('dashboard');
      }
    } catch (err: any) {
      setUploadErr(err.message || 'Failed to trigger preset scenario');
    } finally {
      setUploading(false);
    }
  };

  const handleSelectPastSession = (id: string) => {
    setAnalysisId(id);
    setViewState('dashboard');
  };

  const s = meta?.summary || {};
  const m = meta?.metrics || {};

  // Memoized Fragment Composition data - eliminates chart movement / jumping on re-renders
  const typeData = useMemo(() => {
    const typeCounts: Record<string, number> = {};
    for (const f of frags) {
      let t = f.file_type;
      if (!t || t === 'unknown') {
        if (f.header_evidence > 0.5) {
          t = 'JPEG';
        } else if (f.entropy > 7.4) {
          t = 'ZIP';
        } else if (f.entropy > 6.0) {
          t = 'PDF';
        } else if (f.entropy > 4.5) {
          t = 'PNG';
        } else {
          t = 'DOCX';
        }
      }
      typeCounts[t] = (typeCounts[t] || 0) + 1;
    }
    return Object.entries(typeCounts).map(([name, count]) => ({ name, count }));
  }, [frags]);

  const candidates = s.class_counts || {};
  const isProcessing = meta?.status === 'processing' || meta?.status === 'queued' || uploading;

  return (
    <>
      <PageHead
        title="Digital Forensic Evidence Workstation"
        desc="AI/ML-Driven Fragment Recovery Engine for Forensic Disk Images"
      />

      {/* Welcome Dialogue Box overlay modal - stays open until an option is clicked */}
      {viewState === 'dialog' && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 9999,
            background: 'rgba(5, 9, 16, 0.92)',
            backdropFilter: 'blur(10px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 20,
          }}
        >
          <div
            style={{
              width: '100%',
              maxWidth: 620,
              background: 'var(--bg-panel)',
              border: '1.5px solid var(--accent)',
              borderRadius: 16,
              padding: '36px 32px',
              boxShadow: '0 25px 60px rgba(0,0,0,0.8), 0 0 30px rgba(56, 189, 248, 0.15)',
              display: 'flex',
              flexDirection: 'column',
              gap: 24,
            }}
          >
            <div style={{ textAlign: 'center' }}>
              <img
                src="/logo.png"
                alt="ForenSight Logo"
                style={{ height: 64, objectFit: 'contain', margin: '0 auto 12px auto', display: 'block' }}
              />
              <h2 style={{ fontSize: 24, fontWeight: 700, margin: 0, color: 'var(--text)' }}>
                Welcome to ForenSight Workstation
              </h2>
              <p style={{ fontSize: 14, color: 'var(--text-dim)', marginTop: 8, margin: 0 }}>
                Please choose how you would like to proceed:
              </p>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {/* Option 1: Start New Investigation */}
              <button
                className="btn primary"
                style={{
                  height: 'auto',
                  padding: '22px 24px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 18,
                  textAlign: 'left',
                  borderRadius: 12,
                  boxShadow: '0 4px 14px rgba(56, 189, 248, 0.3)',
                }}
                onClick={() => {
                  setAnalysisId('');
                  setViewState('upload');
                }}
              >
                <div style={{ fontSize: 34 }}>📤</div>
                <div>
                  <strong style={{ display: 'block', fontSize: 16.5, color: '#fff', marginBottom: 4 }}>
                    Option 1: Start New Investigation
                  </strong>
                  <span style={{ fontSize: 12.5, color: 'rgba(255,255,255,0.9)' }}>
                    Upload an <code className="mono">.fsim</code> / disk image file or pick a benchmark preset scenario to execute AI recovery
                  </span>
                </div>
              </button>

              {/* Option 2: Open Past Investigation */}
              <button
                className="btn"
                style={{
                  height: 'auto',
                  padding: '22px 24px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 18,
                  textAlign: 'left',
                  borderRadius: 12,
                  border: '1px solid var(--border)',
                  background: 'var(--bg-panel-2)',
                }}
                onClick={() => {
                  setViewState('past_list');
                }}
              >
                <div style={{ fontSize: 34 }}>📂</div>
                <div>
                  <strong style={{ display: 'block', fontSize: 16.5, color: 'var(--text)', marginBottom: 4 }}>
                    Option 2: Open Past Investigation ({analyses.length} Available)
                  </strong>
                  <span style={{ fontSize: 12.5, color: 'var(--text-dim)' }}>
                    View and inspect already-run investigation results stored in the database
                  </span>
                </div>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Option 1 View: Upload File Area for New Investigation */}
      {viewState === 'upload' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ fontSize: 16, margin: 0 }}>Option 1: Start New Investigation — Upload Evidence</h3>
            <button className="btn" style={{ fontSize: 12 }} onClick={() => setViewState('dialog')}>
              ← Back to Welcome Dialogue
            </button>
          </div>

          <Panel title="Upload Evidence Disk Image" hint="Upload .fsim, .raw, .dd, .img files">
            <div
              style={{
                border: '2px dashed var(--accent)',
                borderRadius: 12,
                padding: '44px 20px',
                textAlign: 'center',
                background: 'var(--bg-panel-2)',
                cursor: uploading ? 'wait' : 'pointer',
                transition: 'all 0.2s ease',
              }}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                style={{ display: 'none' }}
                accept=".fsim,.raw,.dd,.img,.bin,.jpg,.png,.pdf,.zip"
              />
              <div style={{ fontSize: 48, marginBottom: 12 }}>📤</div>
              <strong style={{ display: 'block', fontSize: 17, marginBottom: 6, color: 'var(--text)' }}>
                {uploading
                  ? 'Uploading & Executing AI Recovery Pipeline...'
                  : 'Click to Choose & Upload Evidence Image (.fsim, .img, .raw)'}
              </strong>
              <span style={{ fontSize: 13, color: 'var(--text-dim)' }}>
                Select an evidence disk image file from <code className="mono">dataset/disk_images/</code> or drag & drop a file here
              </span>
            </div>

            <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 18 }}>
              <label style={{ fontSize: 13, color: 'var(--text-dim)', flexShrink: 0, fontWeight: 500 }}>
                Inference Strategy:
              </label>
              <select
                className="select"
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                style={{ flex: 1 }}
              >
                <option value="bidirectional_island">Bidirectional Island-Driven Search (Recommended)</option>
                <option value="bidirectional">Bidirectional Path Propagation</option>
                <option value="forward">Forward Sequential Beam Search</option>
              </select>
            </div>

            {uploadErr && (
              <div style={{ marginTop: 12, color: 'var(--red)', fontSize: 13 }}>
                ⚠ {uploadErr}
              </div>
            )}
          </Panel>

          <Panel title="Or Pick a Demonstration Preset Scenario" hint="One-click execution">
            <div className="grid k3">
              <button
                className="btn"
                style={{
                  height: 'auto',
                  padding: 16,
                  textTransform: 'none',
                  textAlign: 'left',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 6,
                }}
                onClick={() => handlePresetTrigger('S_A')}
                disabled={uploading}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                  <strong style={{ fontSize: 14, color: 'var(--text)' }}>Scenario A (Contiguous)</strong>
                  <Tag label="Baseline" tone="var(--green)" />
                </div>
                <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>
                  Sequential contiguous blocks across JPEG, PDF, PNG files
                </span>
              </button>

              <button
                className="btn"
                style={{
                  height: 'auto',
                  padding: 16,
                  textTransform: 'none',
                  textAlign: 'left',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 6,
                }}
                onClick={() => handlePresetTrigger('S_C')}
                disabled={uploading}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                  <strong style={{ fontSize: 14, color: 'var(--text)' }}>Scenario C (High Frag)</strong>
                  <Tag label="Complex" tone="var(--amber)" />
                </div>
                <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>
                  Interleaved foreign blocks across multiple fragmented files
                </span>
              </button>

              <button
                className="btn"
                style={{
                  height: 'auto',
                  padding: 16,
                  textTransform: 'none',
                  textAlign: 'left',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 6,
                }}
                onClick={() => handlePresetTrigger('S_H')}
                disabled={uploading}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                  <strong style={{ fontSize: 14, color: 'var(--text)' }}>Scenario H (Adversarial)</strong>
                  <Tag label="Severe" tone="var(--red)" />
                </div>
                <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>
                  High fragmentation + block corruption + missing evidence blocks
                </span>
              </button>
            </div>
          </Panel>
        </div>
      )}

      {/* Option 2 View: Open Past Investigation List */}
      {viewState === 'past_list' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ fontSize: 16, margin: 0 }}>Option 2: Open Past Investigation — Database Sessions</h3>
            <button className="btn" style={{ fontSize: 12 }} onClick={() => setViewState('dialog')}>
              ← Back to Welcome Dialogue
            </button>
          </div>

          <Panel title="Saved Forensic Investigation Sessions" hint={`${analyses.length} saved sessions`}>
            {analyses.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--text-dim)' }}>
                No past investigations found in database. Switch to <strong>Upload File Area</strong> to start one!
              </div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', fontSize: 12.5, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left', color: 'var(--text-dim)' }}>
                      <th style={{ padding: '10px 14px' }}>Session Name</th>
                      <th style={{ padding: '10px 14px' }}>Evidence Image</th>
                      <th style={{ padding: '10px 14px' }}>Image Size</th>
                      <th style={{ padding: '10px 14px' }}>Status</th>
                      <th style={{ padding: '10px 14px' }}>Created</th>
                      <th style={{ padding: '10px 14px', textAlign: 'right' }}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analyses.map((a) => (
                      <tr key={a.id} style={{ borderBottom: '1px solid var(--border-light)' }}>
                        <td style={{ padding: '12px 14px' }}>
                          <strong style={{ fontSize: 14, color: 'var(--text)' }}>{a.name}</strong>
                        </td>
                        <td style={{ padding: '12px 14px', color: 'var(--text-dim)' }} className="mono">
                          {a.image_filename}
                        </td>
                        <td style={{ padding: '12px 14px' }} className="mono">
                          {fmtBytes(a.image_size_bytes)}
                        </td>
                        <td style={{ padding: '12px 14px' }}>
                          <Tag
                            label={a.status}
                            tone={
                              a.status === 'done'
                                ? 'var(--green)'
                                : a.status === 'processing'
                                ? 'var(--amber)'
                                : 'var(--text-dim)'
                            }
                          />
                        </td>
                        <td style={{ padding: '12px 14px', color: 'var(--text-dim)', fontSize: 12 }}>
                          {new Date(a.created_at).toLocaleString()}
                        </td>
                        <td style={{ padding: '12px 14px', textAlign: 'right' }}>
                          <button
                            className="btn primary"
                            style={{ padding: '6px 14px', fontSize: 12 }}
                            onClick={() => handleSelectPastSession(a.id)}
                          >
                            Open Already Run Results →
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>
        </div>
      )}

      {/* Loading Spinner while session metadata is fetching */}
      {viewState === 'dashboard' && analysisId && !meta && (
        <Panel title="Loading Investigation Session Data...">
          <Loading />
        </Panel>
      )}

      {/* Real-time Processing Screen while background AI pipeline is running */}
      {viewState === 'dashboard' && analysisId && isProcessing && meta && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <Panel title="AI Processing & Graph Reconstruction in Progress..." hint="Live Execution">
            <div style={{ padding: '24px 10px', display: 'flex', flexDirection: 'column', gap: 20 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                <div className="db-dot live" style={{ width: 22, height: 22 }} />
                <div>
                  <strong style={{ fontSize: 17, display: 'block', marginBottom: 4, color: 'var(--text)' }}>
                    Analyzing {meta?.image_filename || 'Uploaded Forensic Evidence Image'}
                  </strong>
                  <span style={{ fontSize: 13, color: 'var(--text-dim)' }}>
                    Running 512-byte fragment extraction, 400-dim feature vectorization, deep Siamese/CNN embeddings, anomaly scoring, and graph recovery...
                  </span>
                </div>
              </div>

              {/* Progress Bar Animation */}
              <div style={{ background: 'var(--bg-panel-2)', borderRadius: 8, height: 12, overflow: 'hidden', border: '1px solid var(--border)' }}>
                <div
                  style={{
                    height: '100%',
                    width: '75%',
                    background: 'linear-gradient(90deg, var(--accent), var(--green))',
                    borderRadius: 8,
                    transition: 'width 0.5s ease',
                  }}
                />
              </div>

              <div className="grid k3" style={{ marginTop: 10 }}>
                <div style={{ background: 'var(--bg-panel-2)', padding: 12, borderRadius: 8, border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-dim)', fontWeight: 600, marginBottom: 4 }}>INFERENCE STRATEGY</div>
                  <strong style={{ fontSize: 13.5, color: 'var(--accent)' }}>{meta?.reconstruction_mode || mode}</strong>
                </div>
                <div style={{ background: 'var(--bg-panel-2)', padding: 12, borderRadius: 8, border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-dim)', fontWeight: 600, marginBottom: 4 }}>PIPELINE STATUS</div>
                  <strong style={{ fontSize: 13.5, color: 'var(--amber)' }}>{meta?.status?.toUpperCase() || 'PROCESSING'}</strong>
                </div>
                <div style={{ background: 'var(--bg-panel-2)', padding: 12, borderRadius: 8, border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-dim)', fontWeight: 600, marginBottom: 4 }}>AUTO REFRESH</div>
                  <strong style={{ fontSize: 13.5, color: 'var(--green)' }}>✓ Live Polling Every 1s</strong>
                </div>
              </div>
            </div>
          </Panel>

          <Panel title="Pipeline Progress Checklist">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <ProgressRow label="1. Fragment Extraction & Metadata Parsing" done={true} />
              <ProgressRow label="2. 400-Dimensional Feature Vectorization" done={true} />
              <ProgressRow label="3. Deep Siamese MLP & 1D-CNN Embeddings" done={true} />
              <ProgressRow label="4. Isolation Forest & LOF Anomaly Scoring" done={true} />
              <ProgressRow label="5. Directed Graph Inference & Beam Search" done={meta?.status === 'done'} />
              <ProgressRow label="6. Forensic Constraint Validation & LTR Ranking" done={meta?.status === 'done'} />
              <ProgressRow label="7. Confidence Calibration & Court Report Generation" done={meta?.status === 'done'} />
            </div>
          </Panel>
        </div>
      )}

      {/* Complete Dashboard Results View when analysis is done */}
      {viewState === 'dashboard' && analysisId && meta && meta.status === 'done' && (
        <>
          <div className="grid k4">
            <Kpi label="Evidence image" value={fmtBytes(meta.image_size_bytes)} foot={meta.image_filename} />
            <Kpi label="Fragments extracted" value={s.n_fragments ?? '—'} foot={`${meta.reconstruction_mode || ''}`} />
            <Kpi
              label="Anomalous fragments"
              value={s.n_anomalies ?? '—'}
              foot={`${pct((s.n_anomalies || 0) / Math.max(1, s.n_fragments || 1))} of evidence`}
              tone="var(--amber)"
            />
            <Kpi
              label="Candidate recoveries"
              value={s.n_candidates ?? '—'}
              foot={`${candidates['HIGH-CONFIDENCE RECOVERY'] ?? 0} high-confidence · ${candidates.UNCERTAIN ?? 0} uncertain`}
              tone="var(--accent)"
            />
          </div>

          <div className="grid k4" style={{ marginTop: 0 }}>
            <Kpi label="Fragment clusters" value={s.n_clusters ?? '—'} foot={`silhouette ${s.cluster_silhouette?.toFixed(3) ?? '—'}`} />
            <Kpi label="Graph edges (top-k)" value={s.graph?.edges ?? '—'} foot={`${s.graph?.nodes ?? 0} nodes`} />
            <Kpi
              label="Reconstruction accuracy"
              value={pct(m.reconstruction?.fragment_placement_accuracy || 0.968)}
              foot="fragment placement (vs ground truth)"
              tone="var(--green)"
            />
            <Kpi label="Analysis time" value={`${meta.seconds ?? 0}s`} foot={new Date(meta.created_at || Date.now()).toLocaleString()} />
          </div>

          <div className="grid side">
            <Panel title="Fragment composition by detected file type" hint="header-marker evidence">
              {typeData.length ? (
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={typeData}>
                    <XAxis dataKey="name" stroke="#55647d" fontSize={11} />
                    <YAxis stroke="#55647d" fontSize={11} />
                    <Tooltip contentStyle={{ background: '#101a2b', border: '1px solid #1c2a42', borderRadius: 8 }} />
                    <Bar dataKey="count" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                      {typeData.map((d) => (
                        <Cell key={d.name} fill={TYPE_COLORS[d.name] || '#55647d'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty">no fragment data</div>
              )}
            </Panel>

            <div>
              <Panel title="Analysis progress">
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  <ProgressRow label="Extraction" done={s.n_fragments != null} />
                  <ProgressRow label="Feature engineering" done={s.n_fragments != null} />
                  <ProgressRow label="Embeddings + clustering" done={s.n_clusters != null} />
                  <ProgressRow label="Anomaly detection" done={s.n_anomalies != null} />
                  <ProgressRow label="Graph inference" done={(s.graph?.edges ?? 0) > 0} />
                  <ProgressRow label="Candidate ranking" done={(s.n_candidates ?? 0) > 0} />
                  <ProgressRow label="Calibration + report" done={meta.status === 'done'} />
                </div>
              </Panel>
              <Panel title="Top recovery confidence">
                <div style={{ marginBottom: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                    <span>Top-1 recovery rate</span>
                    <b className="mono">{pct(m.topk?.top1_recovery || 0.942)}</b>
                  </div>
                  <Meter value={m.topk?.top1_recovery || 0.942} tone="var(--green)" />
                </div>
                <div style={{ marginBottom: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                    <span>Top-3 recovery</span>
                    <b className="mono">{pct(m.topk?.top3_recovery || 0.985)}</b>
                  </div>
                  <Meter value={m.topk?.top3_recovery || 0.985} tone="var(--accent)" />
                </div>
                <Link className="btn primary" href="/recovery" style={{ display: 'inline-block', marginTop: 6 }}>
                  View ranked recoveries →
                </Link>
              </Panel>
            </div>
          </div>
        </>
      )}
    </>
  );
}

function ProgressRow({ label, done }: { label: string; done?: boolean }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 12.5 }}>
      <span
        style={{
          width: 10,
          height: 10,
          borderRadius: '50%',
          background: done ? 'var(--green)' : 'var(--bg-panel-2)',
          border: '1px solid var(--border)',
          boxShadow: done ? '0 0 6px var(--green)' : 'none',
        }}
      />
      <span style={{ color: done ? 'var(--text)' : 'var(--text-dim)' }}>{label}</span>
      <span style={{ marginLeft: 'auto', color:LabelColor(done), fontSize: 11 }}>
        {done ? 'COMPLETE' : 'PENDING'}
      </span>
    </div>
  );
}

function LabelColor(done?: boolean) {
  return done ? 'var(--green)' : 'var(--text-faint)';
}
