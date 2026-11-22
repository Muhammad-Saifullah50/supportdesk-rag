import type { NextConfig } from 'next';

const backendUrl = (process.env.BACKEND_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '');

const nextConfig: NextConfig = {
  turbopack: {
    root: process.cwd(),
  },
  async rewrites() {
    return [
      { source: '/api/:path*', destination: `${backendUrl}/api/:path*` },
      { source: '/docs/:path*', destination: `${backendUrl}/docs/:path*` },
    ];
  },
};

export default nextConfig;
