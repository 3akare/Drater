import type React from "react"
import type { Metadata } from "next"
import { JetBrains_Mono } from "next/font/google"
import "./globals.css"
import { ThemeProvider } from "@/app/components/theme-provider"

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
})

export const metadata: Metadata = {
  title: "Drater",
  description: "Multi-modal system for real-time sign language translation",
  manifest: "/mainfest.json"
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="light" style={{ colorScheme: "light" }}>
      <head>
        <link rel="icon" href="/favicon.ico" />
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
      </head>
      <body className={`${jetbrainsMono.variable} font-mono`}>
        <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false} disableTransitionOnChange>
          <Toaster expand={true} />
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}

import './globals.css'
import { Toaster } from "sonner"
