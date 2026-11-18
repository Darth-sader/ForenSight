'use client';

import { type ReactNode } from 'react';

export function Panel({ title, hint, children, className }: { title?: string; hint?: string; children: ReactNode; className?: string }) {
  return (
    <section className={`panel ${className ?? ''}`}>
      {title && (
        <div className="panel-title">
          {title}
          {hint && <span className="hint">{hint}</span>}
        </div>
      )}
      {children}
    </section>
  );
}

export function Kpi({ label, value, foot, tone }: { label: string; value: ReactNode; foot?: string; tone?: string }) {
  return (
    <section className="panel kpi">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value" style={tone ? { color: tone } : undefined}>{value}</div>
      {foot && <div className="kpi-foot">{foot}</div>}
    </section>
  );
}

export function Meter({ value, tone }: { value: number; tone?: string }) {
  const v = Math.max(0, Math.min(1, value || 0));
  return (
    <div className="meter">
      <i style={{ width: `${v * 100}%`, background: tone ?? 'var(--accent)' }} />
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function Loading() {
  return <div className="empty">Loading forensic data…</div>;
}

export function PageHead({ title, desc, right }: { title: string; desc?: string; right?: ReactNode }) {
  return (
    <div className="page-head">
      <h1>{title}</h1>
      {desc && <span className="desc">{desc}</span>}
      {right && <div style={{ marginLeft: 'auto' }}>{right}</div>}
    </div>
  );
}

export function Tag({ label, tone }: { label: string; tone?: string }) {
  return (
    <span
      style={{
        display: 'inline-block',
        padding: '2px 8px',
        borderRadius: 4,
        fontSize: 11,
        fontWeight: 600,
        textTransform: 'uppercase',
        background: tone ? `color-mix(in srgb, ${tone} 15%, transparent)` : 'var(--bg-panel-2)',
        color: tone ?? 'var(--text-dim)',
        border: `1px solid ${tone ?? 'var(--border)'}`,
      }}
    >
      {label}
    </span>
  );
}
