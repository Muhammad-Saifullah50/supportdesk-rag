import type { Metadata } from 'next';
import '../src/index.css';

export const metadata: Metadata = {
  title: 'SupportDesk-RAG | Grounded Document Q&A',
  description:
    'Ask questions about an indexed document corpus and inspect the cited source passages.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
