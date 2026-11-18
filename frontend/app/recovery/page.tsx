'use client';

import { useEffect, useState } from 'react';
import { useAnalyses } from '@/components/Shell';
import { PageHead, Panel, Tag, Loading, Meter } from '@/components/ui';
import { fetchJson, pct } from '@/lib/api';

export default function RecoveryPage() {
  const { analysisId } = useAnalyses();
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    if (!analysisId) {
      setData([]);
      return;
    }
    fetchJson<any>(`/api/analyses/${analysisId}/candidates`, 'candidates.json')
      .then((r) => setData(r.data))
      .catch(() => setData([]));
  }, [analysisId]);

  if (!analysisId) {
    return (
      <>
        <PageHead title="Candidate Recovery Hypotheses" desc="Ranked reconstruction candidates" />
        <Panel title="No Evidence Selected">
          <div style={{ textAlign: 'center', padding: '30px 20px', color: 'var(--text-dim)' }}>
            <p>No active evidence image analysis selected.</p>
            <a href="/" className="btn primary" style={{ textDecoration: 'none', display: 'inline-block' }}>
              📤 Go to Evidence Upload Center
            </a>
          </div>
        </Panel>
      </>
    );
  }

  if (!data) return <Loading />;

  const candidates = Array.isArray(data) ? data : data?.candidates || [];

  return (
    <>
      <PageHead
        title="Candidate Recovery Hypotheses"
        desc="Ranked reconstruction candidates scored by multi-evidence ML and validated by forensic constraints"
      />

      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {candidates.map((c: any, idx: number) => {
          const tone =
            c.classification === 'HIGH-CONFIDENCE RECOVERY'
              ? 'var(--green)'
              : c.classification === 'UNCERTAIN'
              ? 'var(--amber)'
              : 'var(--red)';
          return (
            <Panel key={c.candidate_id || idx}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <div>
                  <span style={{ fontSize: 13, color: 'var(--text-dim)', marginRight: 8 }}>
                    CANDIDATE #{idx + 1}
                  </span>
                  <strong style={{ fontSize: 16 }}>{c.file_type} File</strong>
                  <span style={{ fontSize: 12, color: 'var(--text-faint)', marginLeft: 12 }}>
                    {(c.estimated_size / 1024).toFixed(1)} KB ({c.estimated_size} bytes)
                  </span>
                </div>
                <Tag label={c.classification} tone={tone} />
              </div>

              <div style={{ marginBottom: 14 }}>
                <div style={{ fontSize: 11.5, color: 'var(--text-dim)', marginBottom: 4 }}>
                  CALIBRATED CONFIDENCE
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div style={{ flex: 1 }}>
                    <Meter value={c.confidence} tone={tone} />
                  </div>
                  <b className="mono" style={{ fontSize: 15, color: tone }}>
                    {pct(c.confidence)}
                  </b>
                </div>
              </div>

              <div style={{ marginBottom: 14 }}>
                <div style={{ fontSize: 11.5, color: 'var(--text-dim)', marginBottom: 6 }}>
                  FRAGMENT RECOVERY SEQUENCE
                </div>
                <div className="mono" style={{ fontSize: 12, wordBreak: 'break-all', background: 'var(--bg-panel-2)', padding: '8px 12px', borderRadius: 6, border: '1px solid var(--border)' }}>
                  {(c.fragment_sequence || []).join(' → ')}
                </div>
              </div>

              <div className="grid k3" style={{ gap: 12 }}>
                <div style={{ background: 'var(--bg-panel-2)', padding: 10, borderRadius: 6, border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-dim)', fontWeight: 600, marginBottom: 6 }}>ML EVIDENCE</div>
                  {Object.entries(c.ml_evidence || {}).map(([k, v]: any) => (
                    <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, marginBottom: 2 }}>
                      <span style={{ color: 'var(--text-dim)' }}>{k}</span>
                      <span className="mono">{typeof v === 'number' ? v.toFixed(3) : String(v)}</span>
                    </div>
                  ))}
                </div>

                <div style={{ background: 'var(--bg-panel-2)', padding: 10, borderRadius: 6, border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-dim)', fontWeight: 600, marginBottom: 6 }}>FORENSIC VALIDATION</div>
                  {Object.entries(c.forensic_evidence || {}).map(([k, v]: any) => (
                    <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, marginBottom: 2 }}>
                      <span style={{ color: 'var(--text-dim)' }}>{k}</span>
                      <span className="mono" style={{ color: v === true ? 'var(--green)' : v === false ? 'var(--red)' : 'var(--text)' }}>
                        {v === true ? '✓ PASSED' : v === false ? '✗ FAILED' : String(v)}
                      </span>
                    </div>
                  ))}
                </div>

                <div style={{ background: 'var(--bg-panel-2)', padding: 10, borderRadius: 6, border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-dim)', fontWeight: 600, marginBottom: 6 }}>ANOMALIES & ISSUES</div>
                  {c.issues && c.issues.length > 0 ? (
                    c.issues.map((issue: string, i: number) => (
                      <div key={i} style={{ color: 'var(--amber)', fontSize: 11.5, marginBottom: 4 }}>
                        ⚠ {issue}
                      </div>
                    ))
                  ) : (
                    <div style={{ color: 'var(--green)', fontSize: 11.5 }}>✓ No major anomalies or violations</div>
                  )}
                </div>
              </div>
            </Panel>
          );
        })}
      </div>
    </>
  );
}
