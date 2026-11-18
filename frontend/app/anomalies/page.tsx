'use client';

import { useEffect, useState } from 'react';
import { useAnalyses } from '@/components/Shell';
import { PageHead, Panel, Tag, Loading, Kpi, Meter } from '@/components/ui';
import { fetchJson, fmtBytes } from '@/lib/api';

export default function AnomaliesPage() {
  const { analysisId } = useAnalyses();
  const [anomalyData, setAnomalyData] = useState<any>(null);

  useEffect(() => {
    if (!analysisId) {
      setAnomalyData(null);
      return;
    }
    fetchJson<any>(`/api/analyses/${analysisId}/anomalies`, 'anomalies.json')
      .then((r) => setAnomalyData(r.data))
      .catch(() => setAnomalyData(null));
  }, [analysisId]);

  if (!analysisId) {
    return (
      <>
        <PageHead title="Anomaly Detection Layer" desc="Multi-detector anomaly scores for suspicious fragment isolation" />
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

  if (!anomalyData) return <Loading />;

  const flaggedCount = anomalyData.flagged ?? 0;
  const fragments = Array.isArray(anomalyData.fragments) ? anomalyData.fragments : [];
  const detectors = anomalyData.detectors || ['Isolation Forest', 'Local Outlier Factor (LOF)'];

  return (
    <>
      <PageHead
        title="Anomaly Detection Layer"
        desc="Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM scores for identifying suspicious and corrupted fragments"
      />

      <div className="grid k3" style={{ marginBottom: 20 }}>
        <Kpi
          label="Flagged Anomalies"
          value={flaggedCount}
          foot="Fragments exceeding 0.70 threshold"
          tone={flaggedCount > 0 ? 'var(--amber)' : 'var(--green)'}
        />
        <Kpi label="Active Detectors" value={detectors.length} foot="Ensemble outlier consensus" />
        <Kpi label="Scored Fragments" value={fragments.length} foot="Full evidence matrix coverage" tone="var(--accent)" />
      </div>

      <div className="grid k2" style={{ marginBottom: 20 }}>
        <Panel title="Isolation Forest & Density Detectors">
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 10 }}>
            {detectors.map((d: string) => (
              <Tag key={d} label={d.replace('_', ' ').toUpperCase()} tone="var(--green)" />
            ))}
          </div>
          <p style={{ fontSize: 12.5, color: 'var(--text-dim)', margin: 0 }}>
            Tree-based partitioning isolates unusual byte distributions and unexpected slack space, while density-based Local Outlier Factor identifies misassigned foreign blocks in fragmented chains.
          </p>
        </Panel>

        <Panel title="Anomaly Scoring Criteria">
          <div style={{ fontSize: 12, color: 'var(--text-dim)', display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>High Entropy Slack Penalty:</span>
              <span className="mono" style={{ color: 'var(--amber)' }}>Score &gt; 0.75</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Header/Footer Magic Mismatch:</span>
              <span className="mono" style={{ color: 'var(--amber)' }}>Score &gt; 0.70</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Cluster Distance Outlier:</span>
              <span className="mono" style={{ color: 'var(--accent)' }}>Score &gt; 0.65</span>
            </div>
          </div>
        </Panel>
      </div>

      <Panel title={`Flagged Suspicious Fragments (${fragments.length})`}>
        {fragments.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '24px 0', color: 'var(--green)', fontSize: 13 }}>
            ✓ No suspicious or corrupted fragments detected in this evidence image.
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left', color: 'var(--text-dim)' }}>
                  <th style={{ padding: '8px 12px' }}>Fragment ID</th>
                  <th style={{ padding: '8px 12px' }}>Offset</th>
                  <th style={{ padding: '8px 12px' }}>File Type</th>
                  <th style={{ padding: '8px 12px' }}>Entropy</th>
                  <th style={{ padding: '8px 12px' }}>Anomaly Score</th>
                  <th style={{ padding: '8px 12px' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {fragments.slice(0, 30).map((f: any) => (
                  <tr key={f.fragment_id} style={{ borderBottom: '1px solid var(--border-light)' }}>
                    <td className="mono" style={{ padding: '8px 12px', color: 'var(--accent)' }}>{f.fragment_id}</td>
                    <td className="mono" style={{ padding: '8px 12px', color: 'var(--text-dim)' }}>{fmtBytes(f.physical_offset)}</td>
                    <td style={{ padding: '8px 12px' }}><strong>{f.file_type || 'unknown'}</strong></td>
                    <td className="mono" style={{ padding: '8px 12px' }}>{f.entropy?.toFixed(3)}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 120 }}>
                        <div style={{ flex: 1 }}>
                          <Meter value={f.anomaly_score} tone={f.anomaly_score > 0.7 ? 'var(--red)' : 'var(--amber)'} />
                        </div>
                        <span className="mono" style={{ fontSize: 11 }}>{f.anomaly_score?.toFixed(3)}</span>
                      </div>
                    </td>
                    <td style={{ padding: '8px 12px' }}>
                      <Tag label={f.anomaly_status || 'SUSPICIOUS'} tone="var(--red)" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </>
  );
}
