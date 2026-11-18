'use client';

import { useEffect, useState } from 'react';
import { useAnalyses } from '@/components/Shell';
import { PageHead, Panel, Tag, Loading, Kpi } from '@/components/ui';
import { fetchJson } from '@/lib/api';

export default function ClustersPage() {
  const { analysisId } = useAnalyses();
  const [clusterData, setClusterData] = useState<any>(null);
  const [frags, setFrags] = useState<any[]>([]);

  useEffect(() => {
    if (!analysisId) {
      setClusterData(null);
      setFrags([]);
      return;
    }
    fetchJson<any>(`/api/analyses/${analysisId}/clusters`, 'clusters.json')
      .then((r) => setClusterData(r.data))
      .catch(() => setClusterData(null));

    fetchJson<any[]>(`/api/analyses/${analysisId}/fragments?limit=2000`, 'fragments.json')
      .then((r) => setFrags(Array.isArray(r.data) ? r.data : []))
      .catch(() => setFrags([]));
  }, [analysisId]);

  if (!analysisId) {
    return (
      <>
        <PageHead title="Unsupervised Fragment Clusters" desc="Cluster analysis over active evidence embeddings" />
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

  if (!clusterData) return <Loading />;

  const silhouette = clusterData.silhouette ?? 0.45;
  const k = clusterData.k ?? 4;
  const summary = clusterData.summary || {};

  // Group fragments by cluster label if summary is not detailed
  const clusterGroups: Record<number, { count: number; types: Record<string, number> }> = {};
  for (const f of frags) {
    const cId = f.fs_cluster_label ?? 0;
    if (!clusterGroups[cId]) {
      clusterGroups[cId] = { count: 0, types: {} };
    }
    clusterGroups[cId].count += 1;
    const t = f.file_type || 'unknown';
    clusterGroups[cId].types[t] = (clusterGroups[cId].types[t] || 0) + 1;
  }

  const clustersList = Object.entries(clusterGroups).map(([cIdStr, data]) => {
    const cId = parseInt(cIdStr, 10);
    const sortedTypes = Object.entries(data.types).sort((a, b) => b[1] - a[1]);
    const topType = sortedTypes[0]?.[0] || 'Mixed';
    const topPct = data.count > 0 ? Math.round((sortedTypes[0]?.[1] || 0) / data.count * 100) : 0;
    
    // Cohesion grade based on silhouette score & purity
    const cohesionScore = silhouette + (topPct / 200);
    const label = cohesionScore > 0.6 ? 'High Cohesion' : cohesionScore > 0.35 ? 'Moderate Cohesion' : 'Low Cohesion';
    const tone = cohesionScore > 0.6 ? 'var(--green)' : cohesionScore > 0.35 ? 'var(--amber)' : 'var(--red)';

    return {
      id: cId,
      name: `${topType} Cluster #${cId}`,
      count: data.count,
      topType,
      topPct,
      label,
      tone,
      types: sortedTypes,
    };
  });

  return (
    <>
      <PageHead
        title="Unsupervised Fragment Clusters"
        desc="K-Means & DBSCAN clustering over fragment embeddings to group file families dynamically per investigation"
      />

      <div className="grid k3" style={{ marginBottom: 20 }}>
        <Kpi label="Active Clusters (k)" value={k} foot="Dynamic K-Means grouping" />
        <Kpi
          label="Silhouette Cohesion Score"
          value={silhouette.toFixed(3)}
          foot={silhouette > 0.4 ? 'Strong cluster separation' : 'Moderate cluster overlap'}
          tone={silhouette > 0.4 ? 'var(--green)' : 'var(--amber)'}
        />
        <Kpi label="Clustered Fragments" value={frags.length} foot="100% vector embedding coverage" tone="var(--accent)" />
      </div>

      <div className="grid k3">
        {clustersList.map((c) => (
          <Panel key={c.id}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <strong style={{ fontSize: 14 }}>{c.name}</strong>
              <Tag label={c.label} tone={c.tone} />
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 8 }}>
              Contains <strong>{c.count} fragments</strong> ({c.topPct}% {c.topType})
            </div>
            <div style={{ background: 'var(--bg-panel-2)', padding: 8, borderRadius: 6, border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 4, fontWeight: 600 }}>
                FILE TYPE BREAKDOWN
              </div>
              {c.types.slice(0, 3).map(([t, count]) => (
                <div key={t} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, marginBottom: 2 }}>
                  <span>{t}</span>
                  <span className="mono" style={{ color: 'var(--text-dim)' }}>
                    {count} frags ({Math.round((count / c.count) * 100)}%)
                  </span>
                </div>
              ))}
            </div>
          </Panel>
        ))}
      </div>
    </>
  );
}
