"use client"

import * as React from "react"
import { useParams, useRouter } from "next/navigation"
import { ArrowLeft, FileText, AlertCircle, Timer, ArrowLeftRight, ZoomIn, ZoomOut, RefreshCcw } from "lucide-react"
import { toast } from "sonner"

import { AppSidebar } from "@/components/app-sidebar"
import BreadcrumbCurrentPage from "@/components/breadcrumb-current-page"
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"
import { Separator } from "@/components/ui/separator"
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from "@/components/ui/accordion"
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet"
// badges styled to match documents dashboard table
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip"

import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { getBackendUrl } from "@/lib/env"

type AnalysisResultItem = {
  question: string
  pass: boolean
  score?: number
  reason: string
  context?: string
  sources: {
    query_paths?: { document_id: string; path: string; depth: number }[]
    retrieved_pages?: string[]
    [key: string]: unknown
  }
}

type AnalysisResult = {
  id: string
  document_id: string
  analysis_id: string
  analysis_title: string
  document_title: string
  status: string
  results: AnalysisResultItem[]
  total_items: number
  completed_items: number
  processing_time?: number
  usage?: {
    input_tokens?: number
    output_tokens?: number
    total_tokens?: number
  }
  created_at: string
  updated_at: string
}

export default function AnalysisResultPage() {
  const params = useParams()
  const router = useRouter()
  const backendUrl = getBackendUrl()
  const [loading, setLoading] = React.useState(true)
  const [result, setResult] = React.useState<AnalysisResult | null>(null)
  const [openRefsItemIndex, setOpenRefsItemIndex] = React.useState<number | null>(null)
  const [refsQueryTree, setRefsQueryTree] = React.useState<any | null>(null)
  const [refsQueryLoading, setRefsQueryLoading] = React.useState(false)
  const [refsPagesImages, setRefsPagesImages] = React.useState<Record<string, string>>({})
  const [refsPagesLoading, setRefsPagesLoading] = React.useState(false)
  const [previewPageSrc, setPreviewPageSrc] = React.useState<string | null>(null)
  const [previewZoom, setPreviewZoom] = React.useState<number>(1.5)
  const [previewFitWidth, setPreviewFitWidth] = React.useState<number>(0)
  const [previewPan, setPreviewPan] = React.useState<{ x: number; y: number }>({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = React.useState<boolean>(false)
  const previewContainerRef = React.useRef<HTMLDivElement>(null)
  const previewImgRef = React.useRef<HTMLImageElement>(null)
  const dragStartRef = React.useRef<{ x: number; y: number } | null>(null)
  const dragPanStartRef = React.useRef<{ x: number; y: number }>({ x: 0, y: 0 })
  const previewAspectRef = React.useRef<number | null>(null)


  const analysisId = params.id as string
  const documentId = params.documentId as string

  React.useEffect(() => {
    async function fetchResult() {
      try {
        const res = await fetch(
          `${backendUrl}/api/v1/analysis/${analysisId}/document/${documentId}`,
          { headers: { accept: "application/json" } }
        )
        if (!res.ok) throw new Error("Failed to fetch result")
        const data = await res.json()
        setResult(data)
      } catch (err) {
        console.error(err)
        toast.error("Failed to load analysis result")
      } finally {
        setLoading(false)
      }
    }
    fetchResult()
  }, [analysisId, documentId, backendUrl])

  // Merge tree function
  function mergeTrees(a: any, b: any): any {
    if (!a) return b
    if (!b) return a
    const titleA = a?.title
    const titleB = b?.title
    const same = titleA && titleB && titleA === titleB && a?.node_type === b?.node_type
    if (!same) {
      const children = Array.isArray(a?.children) ? a.children.slice() : []
      children.push(b)
      return { ...a, children }
    }
    const pages = Array.isArray(a?.pages) || Array.isArray(b?.pages) ? Array.from(new Set([...(a?.pages || []), ...(b?.pages || [])])) : undefined
    const map = new Map<string, any>()
    const ac = Array.isArray(a?.children) ? a.children : []
    const bc = Array.isArray(b?.children) ? b.children : []
    for (const c of ac) map.set(`${c.title}|${c.node_type}`, c)
    for (const c of bc) {
      const key = `${c.title}|${c.node_type}`
      if (map.has(key)) map.set(key, mergeTrees(map.get(key), c))
      else map.set(key, c)
    }
    return { ...a, pages, children: Array.from(map.values()) }
  }

  function recomputePreviewFit() {
    try {
      const cont = previewContainerRef.current
      const imgEl = previewImgRef.current
      if (!cont || !imgEl) return
      const cw = cont.clientWidth
      const ch = cont.clientHeight
      const iw = imgEl.naturalWidth || imgEl.width
      const ih = imgEl.naturalHeight || imgEl.height
      if (!cw || !ch || !iw || !ih) return
      const scale = Math.min(cw / iw, ch / ih)
      const fitWidth = Math.floor(iw * scale)
      setPreviewFitWidth(fitWidth)
      previewAspectRef.current = ih / iw
      clampPan(previewPan.x, previewPan.y)
    } catch { }
  }

  function clampPan(x: number, y: number) {
    const cont = previewContainerRef.current
    const ar = previewAspectRef.current ?? null
    if (!cont || !ar || previewFitWidth <= 0) {
      setPreviewPan({ x: 0, y: 0 })
      return
    }
    const cw = cont.clientWidth
    const ch = cont.clientHeight
    const widthPx = Math.round(previewFitWidth * previewZoom)
    const heightPx = Math.round(widthPx * ar)
    const maxX = Math.max(0, Math.floor((widthPx - cw) / 2))
    const maxY = Math.max(0, Math.floor((heightPx - ch) / 2))
    const nx = Math.max(-maxX, Math.min(maxX, x))
    const ny = Math.max(-maxY, Math.min(maxY, y))
    setPreviewPan({ x: nx, y: ny })
  }

  React.useEffect(() => {
    clampPan(previewPan.x, previewPan.y)
  }, [previewZoom, previewFitWidth])

  // Load query tree and pages when references opened
  React.useEffect(() => {
    if (openRefsItemIndex == null || !result) {
      setRefsQueryTree(null)
      setRefsPagesImages({})
      setRefsQueryLoading(false)
      setRefsPagesLoading(false)
      setPreviewPageSrc(null)
      return
    }
    const item = result.results[openRefsItemIndex]
    if (!item) return

    const qps: Array<{ document_id: string; path: string; depth?: number }> = Array.isArray(item.sources?.query_paths) ? item.sources.query_paths : []
    const rps: string[] = Array.isArray(item.sources?.retrieved_pages) ? item.sources.retrieved_pages : []

    if (qps.length) {
      setRefsQueryLoading(true)
        ; (async () => {
          try {
            let merged: any | null = null
            await Promise.all(qps.map(async (q) => {
              const u = `${backendUrl}/api/v1/documents/${q.document_id}/tree/path?path=${encodeURIComponent(q.path)}&depth=${q.depth ?? 2}&serialize=false`
              const res = await fetch(u, { headers: { accept: "application/json" } })
              const json = await res.json().catch(() => ({}))
              const subtree = json?.subtree || null
              if (subtree) merged = merged ? mergeTrees(merged, subtree) : subtree
            }))
            setRefsQueryTree(merged)
          } catch {
            setRefsQueryTree(null)
          } finally {
            setRefsQueryLoading(false)
          }
        })()
    } else {
      setRefsQueryTree(null)
      setRefsQueryLoading(false)
    }

    if (rps.length) {
      setRefsPagesLoading(true)
        ; (async () => {
          try {
            const entries = await Promise.all(rps.map(async (pageStr) => {
              // Parse "document_id:page" format
              const parts = pageStr.split(':')
              const docId = parts[0]
              const page = parseInt(parts[1] || '1')
              const u = `${backendUrl}/api/v1/documents/${docId}/page/${page}`
              const res = await fetch(u, { headers: { accept: "application/json" } })
              const json = await res.json().catch(() => null)
              const b64: string | undefined = json?.page_base64
              const src = b64 ? (b64.startsWith("data:image") ? b64 : `data:image/png;base64,${b64}`) : ""
              return [pageStr, src] as const
            }))
            const next: Record<string, string> = {}
            for (const [k, v] of entries) next[k] = v
            setRefsPagesImages(next)
          } catch {
            setRefsPagesImages({})
          } finally {
            setRefsPagesLoading(false)
          }
        })()
    } else {
      setRefsPagesImages({})
      setRefsPagesLoading(false)
    }
  }, [openRefsItemIndex, result, backendUrl])

  function ScoreIndicator({ score }: { score?: number }) {
    const s = Math.max(0, Math.min(3, typeof score === "number" ? score : 0))
    const percent = (s / 3) * 100
    const labels = [
      "Failed",
      "There is non explicit evidence",
      "Partially comply",
      "Fully comply",
    ]
    const colors = [
      "text-red-600",
      "text-orange-500",
      "text-yellow-500",
      "text-green-600",
    ]
    const color = colors[s]
    const r = 16
    const c = 2 * Math.PI * r
    const offset = c * (1 - percent / 100)
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <div className={`relative inline-flex items-center justify-center ${color}`} style={{ width: 32, height: 32 }}>
            <svg viewBox="0 0 40 40" width={32} height={32}>
              <circle cx={20} cy={20} r={r} className="text-muted-foreground/25" stroke="currentColor" strokeWidth={4} fill="none" />
              <circle cx={20} cy={20} r={r} stroke="currentColor" strokeWidth={4} fill="none" strokeDasharray={c} strokeDashoffset={offset} strokeLinecap="round" transform="rotate(-90 20 20)" />
              <text x="20" y="20" textAnchor="middle" dominantBaseline="central" fill="currentColor" className="text-[11px] font-medium text-foreground">
                {s}
              </text>
            </svg>
          </div>
        </TooltipTrigger>
        <TooltipContent>
          {`Score ${s}: ${labels[s]}`}
        </TooltipContent>
      </Tooltip>
    )
  }

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 border-b">
          <div className="flex items-center gap-2 px-3">
            <SidebarTrigger />
            <Separator orientation="vertical" className="mr-2 h-4" />
            <Breadcrumb>
              <BreadcrumbList>
                <BreadcrumbItem className="hidden md:block">
                  <BreadcrumbLink href="#">Archivist</BreadcrumbLink>
                </BreadcrumbItem>
                <BreadcrumbSeparator className="hidden md:block" />
                <BreadcrumbItem className="hidden md:block">
                  <BreadcrumbLink href="/analysis">Analysis</BreadcrumbLink>
                </BreadcrumbItem>
                <BreadcrumbSeparator className="hidden md:block" />
                <BreadcrumbItem>
                  <BreadcrumbPage>
                    <BreadcrumbCurrentPage />
                  </BreadcrumbPage>
                </BreadcrumbItem>
              </BreadcrumbList>
            </Breadcrumb>
          </div>
        </header>
        <div className="flex flex-1 flex-col gap-4 p-4">
          {loading ? (
            <div className="container max-w-5xl py-8 space-y-8">
              <div className="flex items-center gap-4">
                <Skeleton className="h-10 w-10 rounded-md" />
                <div className="space-y-2">
                  <Skeleton className="h-8 w-64" />
                  <Skeleton className="h-4 w-48" />
                </div>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <Skeleton className="h-24" />
                <Skeleton className="h-24" />
              </div>
              <Skeleton className="h-96" />
            </div>
          ) : !result ? (
            <div className="container max-w-5xl py-8 flex flex-col items-center justify-center min-h-[50vh] gap-4">
              <AlertCircle className="size-12 text-muted-foreground" />
              <h2 className="text-xl font-semibold">Result Not Found</h2>
              <Button onClick={() => router.back()}>Go Back</Button>
            </div>
          ) : (
            <div className="container max-w-6xl py-4">
              <div className="flex items-start gap-4 mb-4">
                <Button variant="outline" size="icon" onClick={() => router.back()} className="mt-1">
                  <ArrowLeft className="size-4" />
                </Button>
                <div className="space-y-1">
                  <h1 className="text-2xl font-bold tracking-tight">Analysis Results</h1>
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <FileText className="size-4" />
                    <span>{result.document_title}</span>
                    <span>•</span>
                    <span>{result.analysis_title}</span>
                  </div>
                  <div className="mt-2 flex flex-wrap items-center gap-2">
                    <span className="inline-flex items-center gap-1 rounded bg-muted px-2 py-0.5 text-xs">
                      Completed {result.completed_items}/{result.total_items}
                    </span>
                    {typeof result.processing_time === "number" ? (
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span className="inline-flex items-center gap-1 rounded bg-muted px-2 py-0.5 text-xs cursor-help">
                            <Timer className="size-3" />
                            {result.processing_time.toFixed(2)}s
                          </span>
                        </TooltipTrigger>
                        <TooltipContent>Latency (processing time)</TooltipContent>
                      </Tooltip>
                    ) : null}
                    {typeof result.usage?.input_tokens === "number" || typeof result.usage?.output_tokens === "number" ? (
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span className="inline-flex items-center gap-1 rounded bg-muted px-2 py-0.5 text-xs cursor-help">
                            <ArrowLeftRight className="size-3" />
                            {typeof result.usage?.input_tokens === "number" ? result.usage!.input_tokens : null}
                            {typeof result.usage?.input_tokens === "number" && typeof result.usage?.output_tokens === "number" ? (
                              <span className="mx-1">/</span>
                            ) : null}
                            {typeof result.usage?.output_tokens === "number" ? result.usage!.output_tokens : null}
                          </span>
                        </TooltipTrigger>
                        <TooltipContent>Input / Output tokens</TooltipContent>
                      </Tooltip>
                    ) : null}
                  </div>
                </div>
              </div>
              <div className="space-y-2">
                <Accordion type="single" collapsible>
                  {result.results.map((item, index) => {
                    return (
                      <AccordionItem key={index} value={`item-${index}`} className="border-b">
                        <AccordionTrigger className="hover:no-underline">
                          <div className="flex items-start justify-between w-full">
                            <div className="flex items-start gap-3">
                              <ScoreIndicator score={item.score} />
                              <div className="space-y-1">
                                <div className="text-sm font-medium">{item.question}</div>
                                {item.context ? (
                                  <div className="text-xs text-muted-foreground">{item.context}</div>
                                ) : null}
                              </div>
                            </div>

                          </div>
                        </AccordionTrigger>
                        <AccordionContent>
                          <div className="ml-11">
                            <div className="bg-muted/50 rounded-lg p-4 border">
                              <div className="flex items-center justify-between mb-2">
                                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Reasoning</div>
                                <Sheet>
                                  <SheetTrigger asChild>
                                   <Button variant="outline" size="sm" aria-label="References" className="inline-flex items-center gap-1" onClick={() => setOpenRefsItemIndex(index)}>
                                      <FileText className="size-3" />
                                      References
                                    </Button>
                                  </SheetTrigger>
                                  <SheetContent side="right" className="w-[504px] sm:w-[624px] sm:max-w-[624px]">
                                    <SheetHeader>
                                      <SheetTitle>References</SheetTitle>
                                    </SheetHeader>
                                    <div className="mt-4">
                                      <Tabs defaultValue={item.sources?.query_paths?.length ? "query" : "pages"}>
                                        <TabsList>
                                          <TabsTrigger value="query">Query Paths</TabsTrigger>
                                          <TabsTrigger value="pages">Retrieved Pages</TabsTrigger>
                                        </TabsList>
                                        <TabsContent value="query">
                                          <div className="mt-2">
                                            {refsQueryLoading ? (
                                              <div className="flex flex-col gap-3 p-2">
                                                {Array(3).fill(0).map((_, j) => (
                                                  <Skeleton key={j} className="h-6 w-full" />
                                                ))}
                                              </div>
                                            ) : refsQueryTree ? (
                                              <div className="p-2 rounded border overflow-auto max-h-[70vh]">
                                                <TreeNodeViewInline node={refsQueryTree} depth={0} />
                                              </div>
                                            ) : (
                                              <p className="p-2 text-sm text-muted-foreground">No query paths</p>
                                            )}
                                          </div>
                                        </TabsContent>
                                        <TabsContent value="pages">
                                          <div className="mt-2">
                                            {refsPagesLoading ? (
                                              <div className="p-2">
                                                <Skeleton className="h-64 w-full" />
                                              </div>
                                            ) : Object.keys(refsPagesImages).length ? (
                                              <div className="grid grid-cols-1 gap-3">
                                                {Object.entries(refsPagesImages).map(([key, src]) => (
                                                  <div key={key} className="p-2">
                                                    {src ? (
                                                      <div className="flex w-full items-start justify-center">
                                                        <img
                                                          src={src}
                                                          alt={key}
                                                          className="block w-full h-auto object-contain rounded border cursor-zoom-in"
                                                          onClick={() => { setPreviewPageSrc(src); setPreviewZoom(1.5) }}
                                                        />
                                                      </div>
                                                    ) : (
                                                      <div className="flex h-64 w-full items-center justify-center rounded border bg-muted text-muted-foreground">
                                                        Image unavailable
                                                      </div>
                                                    )}
                                                  </div>
                                                ))}
                                              </div>
                                            ) : (
                                              <p className="p-2 text-sm text-muted-foreground">No retrieved pages</p>
                                            )}
                                          </div>
                                        </TabsContent>
                                      </Tabs>
                                    </div>
                                    <Dialog open={previewPageSrc !== null} onOpenChange={(o) => { if (!o) setPreviewPageSrc(null) }}>
                                      <DialogContent className="w-[96vw] max-w-[96vw] h-[95vh] p-0 overflow-hidden flex flex-col">
                                        <DialogHeader className="relative border-b p-3">
                                          <DialogTitle className="w-full text-center">Page Preview</DialogTitle>
                                        </DialogHeader>
                                        <div className="relative h-[calc(95vh-48px)] bg-background">
                                          <div className="absolute inset-x-0 top-3 z-10 px-3">
                                            <div className="ml-auto flex items-center gap-2">
                                              <span className="px-2 py-1 rounded border bg-background text-xs font-mono tabular-nums">
                                                {Math.round(previewZoom * 100)}%
                                              </span>
                                              <Button size="icon" variant="secondary" onClick={() => setPreviewZoom((z) => Math.max(1, Number((z - 0.1).toFixed(2))))}>
                                                <ZoomOut />
                                              </Button>
                                              <Button size="icon" variant="secondary" onClick={() => setPreviewZoom((z) => Math.min(5, Number((z + 0.1).toFixed(2))))}>
                                                <ZoomIn />
                                              </Button>
                                              <Button size="icon" variant="secondary" onClick={() => { setPreviewZoom(1); setPreviewPan({ x: 0, y: 0 }) }}>
                                                <RefreshCcw />
                                              </Button>
                                            </div>
                                          </div>
                                          <div
                                            ref={previewContainerRef}
                                            className={`flex h-full w-full items-center justify-center overflow-hidden p-2 ${isDragging ? "cursor-grabbing" : "cursor-grab"}`}
                                            onPointerDown={(e) => {
                                              if (!previewPageSrc) return
                                              setIsDragging(true)
                                              dragStartRef.current = { x: e.clientX, y: e.clientY }
                                              dragPanStartRef.current = { ...previewPan }
                                                ; (e.target as HTMLElement).setPointerCapture?.(e.pointerId)
                                            }}
                                            onPointerMove={(e) => {
                                              if (!isDragging || !dragStartRef.current) return
                                              const dx = e.clientX - dragStartRef.current.x
                                              const dy = e.clientY - dragStartRef.current.y
                                              const nx = dragPanStartRef.current.x + dx
                                              const ny = dragPanStartRef.current.y + dy
                                              clampPan(nx, ny)
                                            }}
                                            onPointerUp={() => {
                                              setIsDragging(false)
                                              dragStartRef.current = null
                                            }}
                                            onPointerLeave={() => {
                                              setIsDragging(false)
                                              dragStartRef.current = null
                                            }}
                                            onWheel={(e) => {
                                              e.preventDefault()
                                              e.stopPropagation()
                                              setPreviewZoom((z) => {
                                                const factor = e.deltaY < 0 ? 1.1 : 0.9
                                                const nz = Number((z * factor).toFixed(2))
                                                const clamped = Math.max(1, Math.min(5, nz))
                                                return clamped
                                              })
                                            }}
                                          >
                                            {previewPageSrc ? (
                                              (() => {
                                                const computed = previewFitWidth > 0
                                                const widthPx = computed ? Math.round(previewFitWidth * previewZoom) : undefined
                                                return (
                                                  <img
                                                    ref={previewImgRef}
                                                    src={previewPageSrc}
                                                    alt="Preview"
                                                    className={computed ? "block h-auto max-w-none flex-shrink-0 select-none" : "block h-auto max-w-full max-h-full object-contain select-none"}
                                                    style={computed ? { width: `${widthPx}px`, transform: `translate(${previewPan.x}px, ${previewPan.y}px)`, willChange: "transform" } : undefined}
                                                    onLoad={recomputePreviewFit}
                                                    draggable={false}
                                                  />
                                                )
                                              })()
                                            ) : (
                                              <Skeleton className="h-64 w-full" />
                                            )}
                                          </div>
                                        </div>
                                      </DialogContent>
                                    </Dialog>
                                  </SheetContent>
                                </Sheet>
                              </div>
                              <p className="text-sm leading-relaxed">{item.reason || "No reasoning provided"}</p>
                            </div>
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    )
                  })}
                </Accordion>
              </div>
            </div>
          )}
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}

function TreeNodeViewInline({ node, depth }: { node: any; depth: number }) {
  const hasChildren = Array.isArray(node?.children) && node.children.length > 0
  const typeLabel = node?.node_type === "L1" ? "Hierarchy" : node?.node_type === "L2" ? "Topic" : node?.node_type === "L3" ? "Detail" : node?.node_type
  return (
    <div className="mb-3">
      <div className="flex items-start gap-2">
        <div className="mt-1 h-[10px] w-[10px] shrink-0 rounded-full bg-muted" />
        <div className="flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-medium">{node?.title ?? "Untitled"}</span>
            {typeLabel ? <span className="rounded bg-muted px-2 py-0.5 text-xs">{typeLabel}</span> : null}
            {Array.isArray(node?.pages) && node.pages.length > 0 ? (
              <span className="rounded bg-muted px-2 py-0.5 text-xs">p. {node.pages.join(", ")}</span>
            ) : null}
          </div>
          {node?.summary ? (
            <p className="text-sm text-muted-foreground">{node.summary}</p>
          ) : null}
        </div>
      </div>
      {hasChildren ? (
        <div className="ml-6 mt-2 border-l pl-4">
          {node.children!.map((child: any, i: number) => (
            <TreeNodeViewInline key={i} node={child} depth={depth + 1} />
          ))}
        </div>
      ) : null}
    </div>
  )
}
