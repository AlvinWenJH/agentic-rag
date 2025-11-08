"use client"

import { useRef } from "react"
import Link from "next/link"
import { Mail, Linkedin, Github } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  return (
    <main className="relative min-h-screen overflow-hidden bg-background">
      <canvas ref={canvasRef} className="fixed inset-0 -z-10" />

      {/* Content Overlay */}
      <div className="relative z-10 flex min-h-screen items-center justify-center px-4 py-12">
        <div className="max-w-2xl text-center">
          {/* Main Title */}
          <div className="mb-8">
            <h1 className="text-5xl sm:text-6xl font-bold tracking-tight mb-3">
              <span className="ml-3 bg-gradient-to-r from-pink-500 via-purple-500 to-pink-500 bg-clip-text text-transparent">
                Archivist
              </span>
            </h1>
            <p className="text-lg sm:text-xl text-muted-foreground font-light">Concept tree and Agentic powered RAG</p>
          </div>

          {/* CTA Button */}
          <div className="mb-16">
            <Link href="/login">
              <Button size="lg" className="px-8">
                Get started
              </Button>
            </Link>
          </div>

          {/* Social Icons */}
          <div className="flex justify-center gap-6">
            <a
              href="https://github.com/AlvinWenJH/agentic-rag"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground transition-colors"
              aria-label="GitHub"
            >
              <Github size={24} />
            </a>
            <a
              href="mailto:alvin.wenjianhong@gmail.com"
              className="text-muted-foreground hover:text-foreground transition-colors"
              aria-label="Email"
            >
              <Mail size={24} />
            </a>
            <a
              href="https://www.linkedin.com/in/alvin-wen"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground transition-colors"
              aria-label="LinkedIn"
            >
              <Linkedin size={24} />
            </a>
          </div>
        </div>
      </div>
    </main>
  )
}
