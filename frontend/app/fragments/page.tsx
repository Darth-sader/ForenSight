'use client';

import { useEffect, useMemo, useState } from 'react';
import { useAnalyses } from '@/components/Shell';
import { Panel, PageHead, Loading, Meter, Empty } from '@/components/ui';
import { fetchJson, fmtBytes, TYPE_COLORS } from '@/lib/api';

export default function FragmentsPage() {
  const { analysisId } = useAnalyses();
  const [frags, setFrags] = useState<any[] | null>(null);
  const [q, setQ] = useState('');
  const [type, setType] = useState('');
  const [anomalyOnly, setAnomalyOnly] = useState(false);
  const [sort, setSort] = useState<'index' | 'entropy' | 'anomaly'>('index');

  useEffect(() => {
    if (!analysisId) {
      setFrags([]);
      return;
    }
    fetchJson<any[]>(`/api/analyses/${analysisId}/fragments?limit=2000`, 'fragments.json')
      .then((r) => setFrags(r.data)).catch(() => setFrags([]));
  }, [analysisId]);

  const types = useMemo(
    () => Array.from(new Set((frags ?? []).map((f) => f.file_type || 'unknown'))).sort(),
    [frags],
  );

  const filtered = useMemo(() => {
    let out = frags ?? [];
    if (q) out = out.filter((f) => f.fragment_id.toLowerCase().includes(q.toLowerCase()));
    if (type) out = out.filter((f) => (f.file_type || 'unknown') === type);
    if (anomalyOnly) out = out.filter((f) => f.anomaly_status === 'SUSPICIOUS');
    out = [...out].sort((a, b) =>
      sort === 'entropy' ? b.entropy - a.entropy : sort === 'anomaly' ? b.anomaly_score - a.anomaly_score : a.index - b.index);
    return out.slice(0, 400);
  }, [frags, q, type, anomalyOnly, sort]);

  if (!frags) return <Loading />;

  return (
    <>
      <PageHead title="Fragments" desc="Extracted storage blocks with filesystem context and ML scores" />
      <div style={{ display: 'flex', gap: 10, marginBottom: 14, flexWrap: 'wrap' }}>
        <input
          className="select" style={{ minWidth: 190 }} placeholder="Search fragment id…"
          value={q} onChange={(e) => setQ(e.target.value)}
        />
        <select className="select" value={type} onChange={(e) => setType(e.target.value)}>
          <option value="">all types</option>
          {types.map((t) => <option key={t}>{t}</option>)}
        </select>
        <select className="select" value={sort} onChange={(e) => setSort(e.target.value as any)}>
          <option value="index">sort: block order</option>
          <option value="entropy">sort: entropy</option>
          <option value="anomaly">sort: anomaly</option>
        </select>
        <button className={`btn ${anomalyOnly ? 'primary' : ''}`} onClick={() => setAnomalyOnly(!anomalyOnly)}>
          suspicious only
        </button>
        <span style={{ marginLeft: 'auto', alignSelf: 'center', fontSize: 12, color: 'var(--text-dim)' }}>
          {filtered.length} / {frags.length} fragments
        </span>
      </div>
      <Panel>
        {filtered.length === 0 ? (
          <Empty>no fragments match the filter</Empty>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="tbl">
              <thead>
                <tr>
                  <th>Fragment</th><th>Block</th><th>Offset</th><th>Type</th><th>Size</th>
                  <th>Allocation</th><th>Entropy</th><th>Anomaly</th><th>Status</th>
                  <th>Cluster</th><th>Header</th><th>Consistency</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((f) => (
                  <tr key={f.fragment_id}>
                    <td className="mono" style={{ color: 'var(--accent)' }}>{f.fragment_id}</td>
                    <td className="mono">{f.cluster}</td>
                    <td className="mono" style={{ color: 'var(--text-dim)' }}>{fmtBytes(f.physical_offset)}</td>
                    <td>
                      <span style={{ color: TYPE_COLORS[f.file_type || 'unknown'] ?? 'var(--text-dim)', fontWeight: 600 }}>
                        {f.file_type || '—'}
                      </span>
                    </td>
                    <td className="mono">{f.fragment_size}</td>
                    <td style={{ color: f.allocation_status === 'allocated' ? 'var(--green)' : 'var(--amber)' }}>
                      {f.allocation_status}
                    </td>
                    <td className="mono">{f.entropy?.toFixed(2)}</td>
                    <td>
                      <div className="conf-bar" style={{ minWidth: 90 }}>
                        <Meter value={f.anomaly_score} tone={f.anomaly_score > 0.7 ? 'var(--red)' : 'var(--accent)'} />
                        <span className="mono" style={{ fontSize: 11 }}>{f.anomaly_score?.toFixed(2)}</span>
                      </div>
                    </td>
                    <td>
                      <span className={`badge ${f.anomaly_status === 'SUSPICIOUS' ? 'suspicious' : 'normal'}`}>
                        {f.anomaly_status}
                      </span>
                    </td>
                    <td className="mono">C{f.fs_cluster_label}</td>
                    <td className="mono">{f.header_evidence?.toFixed(2)}</td>
                    <td className="mono">{f.format_consistency?.toFixed(2)}</td>
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
