"use client"

import dynamic from 'next/dynamic'

const GestureRecorder = dynamic(
  () => import('@/app/components/GestureRecorder'),
  { ssr: false }
)

export default function Home() {
  return <GestureRecorder />
}
