"use client"

import * as React from "react"
import { useParams, useRouter } from "next/navigation"
import { ArrowLeft, FileText, AlertCircle, Timer, ArrowLeftRight } from "lucide-react"
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
                                    <Button variant="outline" size="sm" aria-label="References" className="inline-flex items-center gap-1">
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
                                          {Array.isArray(item.sources?.query_paths) && item.sources.query_paths.length ? (
                                            <ul className="mt-2 space-y-2 text-sm">
                                              {item.sources.query_paths.map((qp, i) => (
                                                <li key={i} className="rounded border p-2">
                                                  <div className="font-medium">{qp.path}</div>
                                                  <div className="text-xs text-muted-foreground">Depth {qp.depth}</div>
                                                </li>
                                              ))}
                                            </ul>
                                          ) : (
                                            <p className="mt-2 text-sm text-muted-foreground">No query paths</p>
                                          )}
                                        </TabsContent>
                                        <TabsContent value="pages">
                                          {Array.isArray(item.sources?.retrieved_pages) && item.sources.retrieved_pages.length ? (
                                            <ul className="mt-2 space-y-2 text-sm">
                                              {item.sources.retrieved_pages.map((pg, i) => (
                                                <li key={i} className="rounded border p-2 truncate">{pg}</li>
                                              ))}
                                            </ul>
                                          ) : (
                                            <p className="mt-2 text-sm text-muted-foreground">No retrieved pages</p>
                                          )}
                                        </TabsContent>
                                      </Tabs>
                                    </div>
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
