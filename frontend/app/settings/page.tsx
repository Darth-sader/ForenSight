'use client';

import { PageHead, Panel } from '@/components/ui';

export default function SettingsPage() {
  return (
    <>
      <PageHead
        title="System Settings"
        desc="Global pipeline configuration, fragment sizes, database connection, and API backend settings"
      />
      <Panel title="Configuration Overview">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 12.5 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: 'var(--text-dim)' }}>API Endpoint</span>
            <span className="mono">http://127.0.0.1:8000</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: 'var(--text-dim)' }}>Default Fragment Size</span>
            <span className="mono">512 bytes</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: 'var(--text-dim)' }}>Embedding Dimension</span>
            <span className="mono">32 dims</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: 'var(--text-dim)' }}>Graph Top-K Successors</span>
            <span className="mono">5 edges / node</span>
          </div>
        </div>
      </Panel>
    </>
  );
}
