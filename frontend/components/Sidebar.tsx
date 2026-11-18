'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState, type ReactNode } from 'react';

/* ------------------------------------------------------------------ icons */
function I(props: { d: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d={props.d} />
    </svg>
  );
}
export const IconDisk = () => <I d="M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zm0 6a3 3 0 1 0 0 6 3 3 0 0 0 0-6z" />;
export const IconGrid = () => <I d="M4 4h6v6H4V4zm10 0h6v6h-6V4zM4 14h6v6H4v-6zm10 0h6v6h-6v-6z" />;
export const IconBars = () => <I d="M5 20V10m6 10V4m6 16v-7" />;
export const IconBox = () => <I d="M21 8l-9-5-9 5v8l9 5 9-5V8zM3 8l9 5 9-5M12 13v8" />;
export const IconBrain = () => <I d="M9 3a3 3 0 0 0-3 3v1a3 3 0 0 0 0 6v1a3 3 0 0 0 3 3h1V3H9zm6 0a3 3 0 0 1 3 3v1a3 3 0 0 1 0 6v1a3 3 0 0 1-3 3h-1V3h1z" />;
export const IconScatter = () => <I d="M6 8a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zm12 4a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zM9 19a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zm8-1a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zM4 21h16" />;
export const IconCircles = () => <I d="M9 9a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm8 4a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm-4 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6z" />;
export const IconAlert = () => <I d="M12 3l10 18H2L12 3zm0 7v5m0 3v.5" />;
export const IconShare = () => <I d="M6 12a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5zm12 6a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5zm0-12a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5zM8.2 10.8l7.6-3.6M8.2 13.2l7.6 3.6" />;
export const IconList = () => <I d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" />;
export const IconEye = () => <I d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7zm10 3a3 3 0 1 0 0-6 3 3 0 0 0 0 6z" />;
export const IconFlask = () => <I d="M9 3v6l-5 9a2 2 0 0 0 1.8 3h12.4a2 2 0 0 0 1.8-3l-5-9V3M8 3h8M7 15h10" />;
export const IconGauge = () => <I d="M12 15l4-6M3.5 18a10 10 0 1 1 17 0" />;
export const IconDoc = () => <I d="M6 2h9l5 5v15H6V2zm9 0v5h5M9 13h6M9 17h6" />;
export const IconCog = () => <I d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm8.4-3a8.4 8.4 0 0 0-.1-1.3l2-1.5-2-3.5-2.4 1a8.5 8.5 0 0 0-2.2-1.3L15.3 2h-4l-.4 2.4a8.5 8.5 0 0 0-2.2 1.3l-2.4-1-2 3.5 2 1.5a8.4 8.4 0 0 0 0 2.6l-2 1.5 2 3.5 2.4-1a8.5 8.5 0 0 0 2.2 1.3l.4 2.4h4l.4-2.4a8.5 8.5 0 0 0 2.2-1.3l2.4 1 2-3.5-2-1.5c.06-.43.1-.86.1-1.3z" />;

const NAV: { group: string; items: { href: string; label: string; icon: ReactNode }[] }[] = [
  {
    group: 'Overview',
    items: [
      { href: '/', label: 'Evidence', icon: <IconDisk /> },
      { href: '/fragments', label: 'Fragments', icon: <IconGrid /> },
      { href: '/features', label: 'Features', icon: <IconBars /> },
      { href: '/recovery', label: 'Recovery', icon: <IconBox /> },
    ],
  },
  {
    group: 'AI Intelligence',
    items: [
      { href: '/models', label: 'Models', icon: <IconBrain /> },
      { href: '/embeddings', label: 'Embeddings', icon: <IconScatter /> },
      { href: '/clusters', label: 'Clusters', icon: <IconCircles /> },
      { href: '/anomalies', label: 'Anomalies', icon: <IconAlert /> },
      { href: '/graph', label: 'Relationship Graph', icon: <IconShare /> },
      { href: '/ranking', label: 'Ranking', icon: <IconList /> },
      { href: '/explainability', label: 'Explainability', icon: <IconEye /> },
    ],
  },
  {
    group: 'Research & Evaluation',
    items: [
      { href: '/experiments', label: 'Experiments', icon: <IconFlask /> },
      { href: '/benchmarks', label: 'Benchmarks', icon: <IconGauge /> },
    ],
  },
  {
    group: 'Output',
    items: [{ href: '/reports', label: 'Reports', icon: <IconDoc /> }],
  },
  {
    group: 'System',
    items: [{ href: '/settings', label: 'Settings', icon: <IconCog /> }],
  },
];

export default function Sidebar({ live }: { live: boolean }) {
  const pathname = usePathname();
  return (
    <aside className="sidebar">
      <div className="brand">
        <img
          src="/logo.png"
          alt="ForenSight Logo"
          style={{ width: 34, height: 34, objectFit: 'contain', borderRadius: 6 }}
        />
        <div>
          <div className="brand-name">ForenSight</div>
          <div className="brand-sub">Fragment Recovery ML</div>
        </div>
      </div>
      <nav>
        {NAV.map((g) => (
          <div className="nav-group" key={g.group}>
            <div className="nav-group-title">{g.group}</div>
            {g.items.map((it) => (
              <Link
                key={it.href}
                href={it.href}
                className={`nav-item ${pathname === it.href ? 'active' : ''}`}
              >
                {it.icon}
                {it.label}
              </Link>
            ))}
          </div>
        ))}
      </nav>
      <div className="sidebar-footer">
        <span className={`db-dot ${live ? 'live' : 'demo'}`} />
        {live ? 'Live backend' : 'Demo dataset'} · v1.0
      </div>
    </aside>
  );
}
