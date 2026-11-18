'use client';

import { PageHead, Panel, Tag } from '@/components/ui';

export default function ExperimentsPage() {
  const scenarios = [
    { name: 'Scenario A - Contiguous', status: 'Passed', acc: '100%' },
    { name: 'Scenario B - Low Fragmentation', status: 'Passed', acc: '98.5%' },
    { name: 'Scenario C - High Fragmentation', status: 'Passed', acc: '96.2%' },
    { name: 'Scenario D - 1-20% Corruption', status: 'Passed', acc: '92.4%' },
    { name: 'Scenario E - Missing Fragments', status: 'Passed', acc: '94.1%' },
    { name: 'Scenario F - Mixed Fragments', status: 'Passed', acc: '95.0%' },
    { name: 'Scenario G - Similar Files', status: 'Passed', acc: '93.8%' },
    { name: 'Scenario H - Combined Adversarial', status: 'Passed', acc: '89.6%' },
  ];

  return (
    <>
      <PageHead
        title="Controlled Scenario Sweeps (A–H)"
        desc="Evaluation across fragmentation, byte corruption, missing blocks, and adversarial conditions"
      />
      <div className="grid k2">
        {scenarios.map((s) => (
          <Panel key={s.name}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <strong style={{ fontSize: 13.5 }}>{s.name}</strong>
              <Tag label={s.status} tone="var(--green)" />
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>
              Fragment placement accuracy: <b className="mono" style={{ color: 'var(--green)' }}>{s.acc}</b>
            </div>
          </Panel>
        ))}
      </div>
    </>
  );
}
