import type { NextConfig } from 'next'

// @ts-ignore – next-pwa is CommonJS
const withPWA = require('next-pwa')({
  dest: 'public',
  register: true,
  skipWaiting: true,
})

const nextConfig: NextConfig = {
  output: 'export',
  experimental: {
    serverActions: {
      bodySizeLimit: '500000mb',
    },
  },
}

export default withPWA(nextConfig)
