"use client"

import Link from "next/link"
import * as React from "react"
import { useTheme } from "next-themes"
import { Mail, Linkedin, Github, Sun, Moon } from "lucide-react"
import { Button } from "@/components/ui/button"
import ModernTreeBackground from "@/components/modern-tree-background"

export default function Home() {
  const { theme, resolvedTheme, setTheme } = useTheme()
  const [mounted, setMounted] = React.useState(false)
  React.useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <main className="relative min-h-screen overflow-hidden bg-background">
      <ModernTreeBackground />

      <div className="relative z-10 flex min-h-screen items-center justify-center px-4 py-12">
        <div className="max-w-2xl text-center">
          <div className="mx-auto w-fit">
            <div className="relative rounded-3xl border bg-white/5 dark:bg-white/5 backdrop-blur-md px-8 py-8 shadow-lg">
              {mounted && (
                <div className="absolute right-3 top-3">
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => setTheme((theme || resolvedTheme) === "dark" ? "light" : "dark")}
                    aria-label="Toggle theme"
                  >
                    {(theme || resolvedTheme) === "dark" ? (
                      <Sun className="size-5" />
                    ) : (
                      <Moon className="size-5" />
                    )}
                  </Button>
                </div>
              )}
              <div className="mb-6">
                <h1 className="text-5xl sm:text-6xl font-bold tracking-tight mb-3">
                  <span
                    className={`ml-3 bg-clip-text text-transparent ${
                      (theme || resolvedTheme) === "dark"
                        ? "bg-gradient-to-r from-white via-rose-300 to-pink-500 drop-shadow-[0_2px_3px_rgba(180,180,180,0.12)]"
                        : "bg-gradient-to-r from-pink-500 via-rose-500 to-[#6b6b6b] drop-shadow-[0_2px_3px_rgba(160,160,160,0.18)]"
                    }`}
                  >
                    Archivist
                  </span>
                </h1>
                <p className="text-lg sm:text-xl text-foreground/60 font-small">Concept tree and Agentic powered RAG</p>
              </div>

              <div className="mb-4 flex justify-center">
                <Link href="/login">
                  <Button size="lg" className="px-8">
                    Get started
                  </Button>
                </Link>
              </div>

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
        </div>
      </div>
    </main>
  )
}
