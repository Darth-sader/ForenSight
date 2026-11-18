'use client';

import { useAnalyses } from '@/components/Shell';
import { fmtBytes } from '@/lib/api';

export default function TopBar() {
  const { analyses, analysisId, setAnalysisId, live } = useAnalyses();
  const current = analyses.find((a) => a.id === analysisId);
  return (
    <header className="topbar">
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <img
          src="/logo.png"
          alt="ForenSight Logo"
          style={{ height: 28, width: 28, objectFit: 'contain' }}
        />
        <div>
          <span className="topbar-title">Forensic Investigation Workstation</span>
          {current && (
            <span className="topbar-sub">
              {' '}
              · {current.image_filename} · {fmtBytes(current.image_size_bytes)}
            </span>
          )}
        </div>
      </div>
      <div className="topbar-right">
        {current && (
          <span className={`badge ${current.status === 'done' ? 'normal' : current.status === 'error' ? 'suspicious' : 'uncertain'}`}>
            {current.status.toUpperCase()}
          </span>
        )}
        <button
          className="btn primary"
          style={{ padding: '6px 14px', fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}
          onClick={() => setAnalysisId('')}
          title="Start New Investigation"
        >
          <span>➕</span>
          <span>New Investigation</span>
        </button>

        <select
          className="select"
          value={analysisId}
          onChange={(e) => setAnalysisId(e.target.value)}
          aria-label="Select past investigation"
          style={{ minWidth: 210 }}
        >
          <option value="" disabled>📂 Select Past Investigation...</option>
          {analyses.map((a) => (
            <option key={a.id} value={a.id}>
              📁 {a.name} ({a.status})
            </option>
          ))}
        </select>
        {!live && <span className="badge neutral">DEMO DATA</span>}
      </div>
    </header>
  );
}
