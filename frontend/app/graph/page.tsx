'use client';

import { useEffect, useState } from 'react';
import { useAnalyses } from '@/components/Shell';
import { PageHead, Panel, Loading } from '@/components/ui';
import GraphCanvas from '@/components/GraphCanvas';
import { fetchJson } from '@/lib/api';

export default function GraphPage() {
  const { analysisId } = useAnalyses();
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    if (!analysisId) {
      setData({ nodes: [], edges: [] });
      return;
    }
    fetchJson<any>(`/api/analyses/${analysisId}/graph`, 'graph.json')
      .then((r) => setData(r.data))
      .catch(() => setData({ nodes: [], edges: [] }));
  }, [analysisId]);

  if (!analysisId) {
    return (
      <>
        <PageHead title="Fragment Relationship Graph" desc="Directed graph representation" />
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

  return (
    <>
      <PageHead
        title="Fragment Relationship Graph"
        desc="Directed graph representation with weighted ML-scored edges and node anomaly color mapping"
      />
      <Panel title="Interactive Graph Explorer">
        <GraphCanvas nodes={data.nodes || []} edges={data.edges || []} />
      </Panel>
    </>
  );
}
