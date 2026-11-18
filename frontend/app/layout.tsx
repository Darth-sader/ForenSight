import type { Metadata } from 'next';
import './globals.css';
import Shell from '@/components/Shell';

export const metadata: Metadata = {
  title: 'ForenSight — Intelligent Fragment Recovery',
  description:
    'AI/ML-driven digital forensic fragment recovery: embeddings, clustering, anomaly detection, graph inference and calibrated reconstruction.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
