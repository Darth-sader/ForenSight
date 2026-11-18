'use client';

import { useAnalyses } from '@/components/Shell';
import { PageHead, Panel, Tag } from '@/components/ui';
import { API_BASE } from '@/lib/api';

export default function ReportsPage() {
  const { analysisId } = useAnalyses();

  const reportUrl = (fmt: string) =>
    `${API_BASE}/api/analyses/${analysisId}/report?format=${fmt}`;

  return (
    <>
      <PageHead
        title="Forensic Reports Export"
        desc="Export evidence-supported forensic evaluation reports in JSON, HTML, and formatted Text"
      />

      {!analysisId && (
        <Panel title="No Evidence Selected">
          <div style={{ textAlign: 'center', padding: '30px 20px', color: 'var(--text-dim)' }}>
            <p>No active investigation session selected.</p>
            <a href="/" className="btn primary" style={{ textDecoration: 'none', display: 'inline-block' }}>
              📤 Select or Start an Investigation
            </a>
          </div>
        </Panel>
      )}

      {analysisId && (
        <div className="grid k3">
          <Panel title="JSON Forensic Bundle">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <Tag label="Machine-Readable" tone="var(--accent)" />
            </div>
            <p style={{ fontSize: 13, color: 'var(--text-dim)', marginBottom: 20, lineHeight: 1.5 }}>
              Full structured JSON containing candidate sequences, 400-dim evidence vectors, anomaly flags, and confidence calibration statistics.
            </p>
            <a
              href={reportUrl('json')}
              target="_blank"
              rel="noreferrer"
              className="btn"
              style={{
                display: 'inline-block',
                textDecoration: 'none',
                textAlign: 'center',
                width: '100%',
                padding: '12px 16px',
                fontWeight: 600,
                fontSize: 13,
                color: '#38bdf8',
                background: 'rgba(56, 189, 248, 0.15)',
                border: '1.5px solid #38bdf8',
                borderRadius: 8,
              }}
            >
              📥 Download JSON Report ↓
            </a>
          </Panel>

          <Panel title="HTML Court Evidence Report">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <Tag label="Visual Format" tone="var(--green)" />
            </div>
            <p style={{ fontSize: 13, color: 'var(--text-dim)', marginBottom: 20, lineHeight: 1.5 }}>
              Self-contained HTML court report with CSS styling suitable for browser viewing, printing, or formal court evidence submission.
            </p>
            <a
              href={reportUrl('html')}
              target="_blank"
              rel="noreferrer"
              className="btn"
              style={{
                display: 'inline-block',
                textDecoration: 'none',
                textAlign: 'center',
                width: '100%',
                padding: '12px 16px',
                fontWeight: 700,
                fontSize: 13,
                color: '#070c14',
                background: '#4ade80',
                border: '1.5px solid #4ade80',
                borderRadius: 8,
              }}
            >
              📜 Open / Export HTML Report ↗
            </a>
          </Panel>

          <Panel title="Plain Text Summary Log">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <Tag label="Plain Text" tone="var(--amber)" />
            </div>
            <p style={{ fontSize: 13, color: 'var(--text-dim)', marginBottom: 20, lineHeight: 1.5 }}>
              Formatted plain-text audit summary log for terminal inspection, command-line tool ingestion, and forensic archives.
            </p>
            <a
              href={reportUrl('md')}
              target="_blank"
              rel="noreferrer"
              className="btn"
              style={{
                display: 'inline-block',
                textDecoration: 'none',
                textAlign: 'center',
                width: '100%',
                padding: '12px 16px',
                fontWeight: 600,
                fontSize: 13,
                color: '#fbbf24',
                background: 'rgba(251, 191, 36, 0.15)',
                border: '1.5px solid #fbbf24',
                borderRadius: 8,
              }}
            >
              📄 Download Text Summary ↓
            </a>
          </Panel>
        </div>
      )}
    </>
  );
}
