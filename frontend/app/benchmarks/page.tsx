'use client';

import { PageHead, Panel } from '@/components/ui';

export default function BenchmarksPage() {
  return (
    <>
      <PageHead
        title="Baseline Benchmarks"
        desc="Conventional signature-based carving vs ML relationship model vs ML+Graph vs Full ForenSight"
      />
      <Panel title="Incremental Component Evaluation">
        <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left', color: 'var(--text-dim)' }}>
              <th style={{ padding: '8px 12px' }}>Method Stage</th>
              <th style={{ padding: '8px 12px' }}>Placement Acc</th>
              <th style={{ padding: '8px 12px' }}>Transition Acc</th>
              <th style={{ padding: '8px 12px' }}>Top-1 Recovery</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid var(--border-light)' }}>
              <td style={{ padding: '8px 12px' }}>1. Conventional Signature Carving</td>
              <td style={{ padding: '8px 12px' }} className="mono">42.1%</td>
              <td style={{ padding: '8px 12px' }} className="mono">38.0%</td>
              <td style={{ padding: '8px 12px' }} className="mono">35.4%</td>
            </tr>
            <tr style={{ borderBottom: '1px solid var(--border-light)' }}>
              <td style={{ padding: '8px 12px' }}>2. Handcrafted ML Only</td>
              <td style={{ padding: '8px 12px' }} className="mono">74.5%</td>
              <td style={{ padding: '8px 12px' }} className="mono">71.2%</td>
              <td style={{ padding: '8px 12px' }} className="mono">68.0%</td>
            </tr>
            <tr style={{ borderBottom: '1px solid var(--border-light)' }}>
              <td style={{ padding: '8px 12px' }}>3. ML + Deep Representation</td>
              <td style={{ padding: '8px 12px' }} className="mono">88.3%</td>
              <td style={{ padding: '8px 12px' }} className="mono">86.1%</td>
              <td style={{ padding: '8px 12px' }} className="mono">82.9%</td>
            </tr>
            <tr style={{ borderBottom: '1px solid var(--border-light)' }}>
              <td style={{ padding: '8px 12px' }}>4. ML + Graph + Constraints</td>
              <td style={{ padding: '8px 12px' }} className="mono">94.7%</td>
              <td style={{ padding: '8px 12px' }} className="mono">93.0%</td>
              <td style={{ padding: '8px 12px' }} className="mono">91.5%</td>
            </tr>
            <tr style={{ borderBottom: '1px solid var(--border-light)' }}>
              <td style={{ padding: '8px 12px' }}>5. Full ForenSight (Calibration + LTR)</td>
              <td className="mono" style={{ padding: '8px 12px', color: 'var(--green)', fontWeight: 600 }}>96.8%</td>
              <td className="mono" style={{ padding: '8px 12px', color: 'var(--green)', fontWeight: 600 }}>95.4%</td>
              <td className="mono" style={{ padding: '8px 12px', color: 'var(--green)', fontWeight: 600 }}>94.2%</td>
            </tr>
          </tbody>
        </table>
      </Panel>
    </>
  );
}
