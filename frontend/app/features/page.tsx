'use client';

import { useEffect, useState } from 'react';
import { useAnalyses } from '@/components/Shell';
import { PageHead, Panel, Loading, Tag } from '@/components/ui';
import { fetchJson } from '@/lib/api';

export default function FeaturesPage() {
  const { analysisId } = useAnalyses();
  const [frags, setFrags] = useState<any[] | null>(null);

  useEffect(() => {
    if (!analysisId) {
      setFrags([]);
      return;
    }
    fetchJson<any[]>(`/api/analyses/${analysisId}/fragments?limit=50`, 'fragments.json')
      .then((r) => setFrags(Array.isArray(r.data) ? r.data : []))
      .catch(() => setFrags([]));
  }, [analysisId]);

  if (frags === null) return <Loading />;

  const families = [
    { name: 'Byte Histogram & Stats', dims: 260, desc: '256-dim byte frequency histogram, mean, variance, min, max, entropy' },
    { name: 'Statistical Ratios & N-grams', dims: 42, desc: 'Printable, ASCII, zero-byte, high-byte ratios, byte diversity, 2-gram/3-gram frequencies' },
    { name: 'Structural & Format Markers', dims: 32, desc: 'Magic bytes, header/footer markers across 8 file types, format consistency, conflict scores' },
    { name: 'Contextual & Locality', dims: 16, desc: 'Physical sector offset, block distance, filesystem cluster index, allocation status' },
    { name: 'Pairwise Continuity & Transitions', dims: 50, desc: 'Boundary byte continuity, format rule compatibility, prefix-suffix overlap, entropy delta' },
  ];

  return (
    <>
      <PageHead
        title="Feature Engineering Architecture"
        desc="400-dimensional multi-disciplinary feature space combining byte, statistical, structural, contextual and pairwise transition properties"
      />

      <div className="grid k2" style={{ marginBottom: 20 }}>
        {families.map((f) => (
          <Panel key={f.name}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <strong style={{ fontSize: 14 }}>{f.name}</strong>
              <Tag label={`${f.dims} dims`} tone="var(--accent)" />
            </div>
            <p style={{ fontSize: 12.5, color: 'var(--text-dim)', margin: 0 }}>{f.desc}</p>
          </Panel>
        ))}
      </div>

      <Panel title="Extracted Fragment Features (Live Sample)">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left', color: 'var(--text-dim)' }}>
                <th style={{ padding: '8px 12px' }}>ID</th>
                <th style={{ padding: '8px 12px' }}>Entropy</th>
                <th style={{ padding: '8px 12px' }}>Header Marker</th>
                <th style={{ padding: '8px 12px' }}>Footer Marker</th>
                <th style={{ padding: '8px 12px' }}>Format Consistency</th>
                <th style={{ padding: '8px 12px' }}>Anomaly Score</th>
              </tr>
            </thead>
            <tbody>
              {frags.slice(0, 20).map((row: any) => (
                <tr key={row.fragment_id} style={{ borderBottom: '1px solid var(--border-light)' }}>
                  <td style={{ padding: '8px 12px' }} className="mono">{row.fragment_id}</td>
                  <td style={{ padding: '8px 12px' }} className="mono">{row.entropy?.toFixed(3)}</td>
                  <td style={{ padding: '8px 12px' }}>{row.header_evidence > 0.5 ? '✓ Yes' : '—'}</td>
                  <td style={{ padding: '8px 12px' }}>{row.footer_evidence > 0.5 ? '✓ Yes' : '—'}</td>
                  <td style={{ padding: '8px 12px' }} className="mono">{row.format_consistency?.toFixed(3)}</td>
                  <td className="mono" style={{ padding: '8px 12px', color: row.anomaly_score > 0.7 ? 'var(--amber)' : 'var(--text)' }}>
                    {row.anomaly_score?.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </>
  );
}
