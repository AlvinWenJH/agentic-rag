"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { FileText, Download, Eye, PlayCircle, CheckCircle2, XCircle, ClipboardList, Search, Trash } from "lucide-react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter, DialogClose } from "@/components/ui/dialog"
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination"
import { getBackendUrl } from "@/lib/env"
import { parseUtcDate } from "@/lib/utils"
import { toast } from "sonner"
import { Spinner } from "@/components/ui/spinner"

type AnalysisItem = {
  question: string
  context?: string
  order: number
}

type Analysis = {
  id: string
  title: string
  description: string
  items: AnalysisItem[]
  status: string
  created_at: string
  updated_at: string
}

type DocumentItem = {
  id: string
  title: string
  filename: string
  status: string
}

type AnalysisResultItem = {
  question: string
  pass: boolean
  reason: string
  context?: string
  sources: {
    query_paths?: string[]
    retrieved_pages?: string[]
    [key: string]: unknown
  }
}

type AnalysisResult = {
  id: string
  document_id: string
  document_title: string
  status: string
  results: AnalysisResultItem[]
  total_items: number
  completed_items: number
  created_at: string
  updated_at: string
  score_total?: number
  score_max?: number
  score_percentage?: number
}

function formatDateTime(dt?: string | null) {
  if (!dt) return "—"
  const d = parseUtcDate(dt)
  if (!d) return dt as string
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  })
}

export default function AnalysisDetail({ analysisId }: { analysisId: string }) {
  const router = useRouter()
  const backendUrl = getBackendUrl()
  const [loading, setLoading] = React.useState(true)
  const [analysis, setAnalysis] = React.useState<Analysis | null>(null)

  // Document analysis state
  const [documents, setDocuments] = React.useState<DocumentItem[]>([])
  const [analysisResults, setAnalysisResults] = React.useState<AnalysisResult[]>([])
  const [loadingResults, setLoadingResults] = React.useState(true)
  const [openDocumentDialog, setOpenDocumentDialog] = React.useState(false)
  const [openDeleteDialog, setOpenDeleteDialog] = React.useState(false)
  const [deleteTarget, setDeleteTarget] = React.useState<AnalysisResult | null>(null)
  const [deleting, setDeleting] = React.useState(false)
  const [selectedDoc, setSelectedDoc] = React.useState<string | null>(null)
  const [running, setRunning] = React.useState(false)

  // Document search with pagination (server-side)
  const [documentSearch, setDocumentSearch] = React.useState("")
  const [documentSkip, setDocumentSkip] = React.useState(0)
  const [documentHasMore, setDocumentHasMore] = React.useState(true)
  const [loadingDocuments, setLoadingDocuments] = React.useState(false)
  const documentListRef = React.useRef<HTMLDivElement>(null)
  const DOCUMENTS_PER_PAGE = 10

  // Results server-side pagination and filtering
  const [resultsSearch, setResultsSearch] = React.useState("")
  const [debouncedResultsSearch, setDebouncedResultsSearch] = React.useState("")
  const [resultsPage, setResultsPage] = React.useState(1)
  const [totalResults, setTotalResults] = React.useState(0)
  const RESULTS_PER_PAGE = 10

  // View items dialog
  const [openItemsDialog, setOpenItemsDialog] = React.useState(false)

  // Load analysis
  React.useEffect(() => {
    async function fetchAnalysis() {
      try {
        const res = await fetch(`${backendUrl}/api/v1/analysis/${analysisId}`, {
          headers: { accept: "application/json" },
        })
        if (!res.ok) throw new Error(`Failed to fetch analysis`)
        const data = await res.json()
        setAnalysis(data)
      } catch (err) {
        console.error(err)
        toast.error("Failed to load analysis")
      } finally {
        setLoading(false)
      }
    }
    fetchAnalysis()
  }, [analysisId, backendUrl])

  // Load documents with server-side search and pagination
  async function loadDocuments(reset = false) {
    if (loadingDocuments || (!documentHasMore && !reset)) return

    setLoadingDocuments(true)
    try {
      const skip = reset ? 0 : documentSkip
      const searchParam = documentSearch ? `&search=${encodeURIComponent(documentSearch)}` : ''
      const res = await fetch(
        `${backendUrl}/api/v1/documents/?skip=${skip}&limit=${DOCUMENTS_PER_PAGE}${searchParam}`,
        { headers: { accept: "application/json" } }
      )
      const data = await res.json().catch(() => ({ documents: [] }))
      const newDocs = Array.isArray(data?.documents)
        ? data.documents.filter((d: DocumentItem) => d.status === "completed")
        : []

      if (reset) {
        setDocuments(newDocs)
        setDocumentSkip(DOCUMENTS_PER_PAGE)
      } else {
        setDocuments(prev => [...prev, ...newDocs])
        setDocumentSkip(skip + DOCUMENTS_PER_PAGE)
      }

      setDocumentHasMore(newDocs.length === DOCUMENTS_PER_PAGE)
    } catch (err) {
      console.error(err)
      if (reset) setDocuments([])
    } finally {
      setLoadingDocuments(false)
    }
  }

  // Load initial documents when dialog opens
  React.useEffect(() => {
    if (openDocumentDialog && documents.length === 0) {
      loadDocuments(true)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openDocumentDialog])

  // Reset and search when search query changes
  React.useEffect(() => {
    if (openDocumentDialog) {
      const timer = setTimeout(() => {
        setDocumentSkip(0)
        setDocumentHasMore(true)
        loadDocuments(true)
      }, 300)
      return () => clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documentSearch])

  // Scroll detection for infinite scroll
  function handleDocumentScroll() {
    const el = documentListRef.current
    if (!el) return

    const { scrollTop, scrollHeight, clientHeight } = el
    if (scrollTop + clientHeight >= scrollHeight - 50 && documentHasMore && !loadingDocuments) {
      loadDocuments(false)
    }
  }

  // Debounce results search
  React.useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedResultsSearch(resultsSearch)
      setResultsPage(1)
    }, 300)
    return () => clearTimeout(timer)
  }, [resultsSearch])

  // Load analysis results with server-side pagination
  React.useEffect(() => {
    async function fetchResults() {
      setLoadingResults(true)
      try {
        const skip = (resultsPage - 1) * RESULTS_PER_PAGE
        const searchParam = debouncedResultsSearch ? `&search=${encodeURIComponent(debouncedResultsSearch)}` : ''
        const res = await fetch(
          `${backendUrl}/api/v1/analysis/${analysisId}/documents?skip=${skip}&limit=${RESULTS_PER_PAGE}${searchParam}`,
          { headers: { accept: "application/json" } }
        )
        if (!res.ok) throw new Error("Failed to fetch analysis results")
        const data = await res.json()
        setAnalysisResults(data.results || [])
        setTotalResults(data.total || 0)
      } catch (err) {
        console.error(err)
        toast.error("Failed to load analysis results")
        setAnalysisResults([])
        setTotalResults(0)
      } finally {
        setLoadingResults(false)
      }
    }
    if (analysis) {
      fetchResults()
    }
  }, [analysis, analysisId, backendUrl, resultsPage, debouncedResultsSearch])

  async function handleRunAnalysis() {
    if (!selectedDoc || !analysis) return
    setRunning(true)
    try {
      let userId: string | null = null
      try {
        const raw = typeof window !== "undefined" ? localStorage.getItem("auth_user") : null
        const obj = raw ? JSON.parse(raw) : null
        userId = (obj?.user_id ?? obj?.id ?? null) as string | null
      } catch {}

      const res = await fetch(`${backendUrl}/api/v1/analysis/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
        },
        body: JSON.stringify({
          document_id: selectedDoc,
          analysis_id: analysisId,
          user_id: userId,
        }),
      })
      if (!res.ok) throw new Error(`Failed to start analysis`)
      const data = await res.json()

      toast.success("Analysis started", {
        description: "Processing in background...",
      })

      setOpenDocumentDialog(false)
      setSelectedDoc(null)

      // Add optimistic result to list
      const doc = documents.find(d => d.id === selectedDoc)
      if (doc) {
        setAnalysisResults(prev => [{
          id: data.id,
          document_id: selectedDoc,
          document_title: doc.title || doc.filename,
          status: "processing",
          results: [],
          total_items: analysis.items.length,
          completed_items: 0,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          score_total: 0,
          score_max: analysis.items.length * 3,
          score_percentage: 0,
        }, ...prev])
        setTotalResults(prev => prev + 1)
      }

      // Start polling
      pollAnalysisResult(data.id)
    } catch (err) {
      console.error(err)
      toast.error("Failed to start analysis")
    } finally {
      setRunning(false)
    }
  }

  async function pollAnalysisResult(resultId: string) {
    const maxAttempts = 60
    let attempts = 0

    const interval = setInterval(async () => {
      attempts++
      try {
        const res = await fetch(`${backendUrl}/api/v1/analysis/results/${resultId}`, {
          headers: { accept: "application/json" },
        })
        if (!res.ok) throw new Error("Failed to fetch result")
        const result = await res.json()

        if (result.status === "completed" || result.status === "failed") {
          clearInterval(interval)
          setAnalysisResults(prev => prev.map(r => r.id === result.id ? result : r))

          if (result.status === "completed") {
            toast.success("Analysis completed")
          } else {
            toast.error("Analysis failed")
          }
        } else {
          // Update progress
          setAnalysisResults(prev => prev.map(r => r.id === result.id ? result : r))
        }
      } catch (err) {
        console.error(err)
      }

      if (attempts >= maxAttempts) {
        clearInterval(interval)
      }
    }, 5000)
  }

  function handleExport() {
    toast.info("Export functionality coming soon")
  }

  function handleViewResult(result: AnalysisResult) {
    router.push(`/analysis/${analysisId}/result/${result.document_id}`)
  }

  function confirmDeleteResult(result: AnalysisResult) {
    setDeleteTarget(result)
    setOpenDeleteDialog(true)
  }

  async function performDeleteResult() {
    if (!deleteTarget) return
    setDeleting(true)
    try {
      const res = await fetch(
        `${backendUrl}/api/v1/analysis/${analysisId}/result/${deleteTarget.document_id}`,
        {
          method: "DELETE",
          headers: { accept: "application/json" },
        }
      )
      if (!res.ok) throw new Error(`Delete failed (${res.status})`)

      // Optimistically update list
      setAnalysisResults((prev) => prev.filter((r) => r.id !== deleteTarget.id))
      setTotalResults(prev => Math.max(0, prev - 1))
      setOpenDeleteDialog(false)
      setDeleteTarget(null)
      toast.success("Analysis result deleted")
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err)
      toast.error(message || "Unable to delete analysis result")
    } finally {
      setDeleting(false)
    }
  }

  const totalPages = Math.ceil(totalResults / RESULTS_PER_PAGE)

  if (loading) {
    return (
      <div className="flex flex-col gap-4">
        <Skeleton className="h-48 w-full" />
        <Skeleton className="h-96 w-full" />
      </div>
    )
  }

  if (!analysis) {
    return (
      <Card>
        <CardContent className="py-12 text-center">
          <p className="text-muted-foreground">Analysis not found</p>
        </CardContent>
      </Card>
    )
  }

  const completedCount = analysisResults.filter(r => r.status === "completed").length
  const inProgressCount = analysisResults.filter(r => r.status === "processing" || r.status === "pending").length

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Top Section (30%) - Overview */}
      <div className="grid gap-4 lg:grid-cols-3">
        {/* Left: Actions */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ClipboardList className="size-5 text-muted-foreground" />
              Quick Actions
            </CardTitle>
            <CardDescription className="sr-only">Actions for this analysis</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <Dialog open={openDocumentDialog} onOpenChange={setOpenDocumentDialog}>
              <DialogTrigger asChild>
                <Button className="bg-foreground text-background hover:opacity-95 px-3">
                  <div className="grid grid-cols-[24px_1fr_24px] items-center w-full">
                    <PlayCircle className="size-4" />
                    <span className="justify-self-center">Run Analysis</span>
                    <span className="size-4" aria-hidden="true" />
                  </div>
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-lg">
                <DialogHeader>
                  <DialogTitle>Select Document</DialogTitle>
                  <DialogDescription>Choose a document to analyze</DialogDescription>
                </DialogHeader>

                {/* Search Input */}
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
                  <Input
                    placeholder="Search documents..."
                    value={documentSearch}
                    onChange={(e) => setDocumentSearch(e.target.value)}
                    className="pl-9"
                  />
                </div>

                {/* Document List with Infinite Scroll */}
                <div
                  ref={documentListRef}
                  onScroll={handleDocumentScroll}
                  className="max-h-[400px] overflow-y-auto space-y-2 pr-2"
                >
                  {documents.length === 0 && !loadingDocuments ? (
                    <p className="text-sm text-muted-foreground text-center py-8">
                      {documentSearch ? "No documents found" : "No completed documents available"}
                    </p>
                  ) : (
                    <>
                      {documents.map((doc) => (
                        <div
                          key={doc.id}
                          className={`p-3 border rounded-lg cursor-pointer transition-colors ${selectedDoc === doc.id ? "bg-accent border-accent-foreground" : "hover:bg-muted"
                            }`}
                          onClick={() => setSelectedDoc(doc.id)}
                        >
                          <div className="flex items-center gap-2">
                            <FileText className="size-4 flex-shrink-0" />
                            <div className="flex-1 min-w-0">
                              <p className="font-medium text-sm truncate">{doc.title || doc.filename}</p>
                              <p className="text-xs text-muted-foreground truncate">{doc.filename}</p>
                            </div>
                          </div>
                        </div>
                      ))}
                      {loadingDocuments && (
                        <div className="flex items-center justify-center py-4">
                          <Spinner className="size-5" />
                        </div>
                      )}
                    </>
                  )}
                </div>

                <DialogFooter>
                  <Button onClick={handleRunAnalysis} disabled={!selectedDoc || running}>
                    {running ? <Spinner className="size-4 mr-2" /> : null}
                    Run
                  </Button>
                  <DialogClose asChild>
                    <Button variant="outline">Cancel</Button>
                  </DialogClose>
                </DialogFooter>
              </DialogContent>
            </Dialog>
            <Button
              variant="outline"
              onClick={() => setOpenItemsDialog(true)}
              className="px-3"
            >
              <div className="grid grid-cols-[24px_1fr_24px] items-center w-full">
                <Eye className="size-4" />
                <span className="justify-self-center">View Analysis Items</span>
                <span className="size-4" aria-hidden="true" />
              </div>
            </Button>
            <Button
              variant="outline"
              onClick={handleExport}
              className="px-3"
            >
              <div className="grid grid-cols-[24px_1fr_24px] items-center w-full">
                <Download className="size-4" />
                <span className="justify-self-center">Export</span>
                <span className="size-4" aria-hidden="true" />
              </div>
            </Button>
          </CardContent>
        </Card>

        {/* Right: Details */}
        <Card className="lg:col-span-2 gap-0">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ClipboardList className="size-5 text-muted-foreground" />
              {analysis.title}
            </CardTitle>
            <CardDescription>
              {analysis.description && analysis.description.length > 100 ? (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="cursor-help">
                      {analysis.description.slice(0, 100)}...
                    </span>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-md">
                    <p>{analysis.description}</p>
                  </TooltipContent>
                </Tooltip>
              ) : (
                analysis.description || "No description"
              )}
            </CardDescription>
          </CardHeader>
          <CardContent className="py-4">
            <div className="grid gap-4 md:grid-cols-2 mb-4">
              <div>
                <div className="text-sm text-muted-foreground">Created</div>
                <div className="mt-1 text-sm">{formatDateTime(analysis.created_at)}</div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground">Last Updated</div>
                <div className="mt-1 text-sm">{formatDateTime(analysis.updated_at)}</div>
              </div>
            </div>
            <div className="my-3 border-b" />
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="rounded-lg border p-4 text-center">
                <div className="text-3xl font-bold tracking-tight">{analysis.items.length}</div>
                <div className="mt-1 text-xs text-muted-foreground">Analysis Items</div>
              </div>
              <div className="rounded-lg border p-4 text-center">
                <div className="text-3xl font-bold tracking-tight">{completedCount}</div>
                <div className="mt-1 text-xs text-muted-foreground">Documents Analyzed</div>
              </div>
              <div className="rounded-lg border p-4 text-center">
                <div className="text-3xl font-bold tracking-tight">{inProgressCount}</div>
                <div className="mt-1 text-xs text-muted-foreground">In Progress</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Bottom Section (70%) - Documents Table */}
      <Card className="flex-1">
        <CardHeader className="border-b">
          <CardTitle>Analyzed Documents</CardTitle>
          <CardDescription>{formatNumber(totalResults)} documents</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mb-4 flex items-center gap-2">
            <div className="relative w-full max-w-md">
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
              <Input
                placeholder="Search results..."
                value={resultsSearch}
                onChange={(e) => setResultsSearch(e.target.value)}
                className="pl-8"
              />
            </div>
          </div>

          <div className="pt-0">
            {loadingResults ? (
              <div className="flex flex-col gap-2">
                {Array(3).fill(0).map((_, i) => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            ) : analysisResults.length > 0 ? (
              <div className="space-y-4">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-muted-foreground border-b">
                        <th className="py-3 px-2 text-left font-medium">Document</th>
                        <th className="py-3 px-2 text-left font-medium">Status</th>
                        <th className="py-3 px-2 text-left font-medium">Progress</th>
                      <th className="py-3 px-2 text-left font-medium">Score</th>
                      <th className="py-3 px-2 text-left font-medium">Last Updated</th>
                        <th className="py-3 px-2 text-right font-medium">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analysisResults.length > 0 ? (
                        analysisResults.map((result) => (
                          <tr key={result.id} className="border-b last:border-0">
                            <td className="py-3 px-2">
                              <div className="flex items-center gap-2">
                                <FileText className="size-4 text-muted-foreground" />
                                <span className="font-medium truncate max-w-[22ch]">{result.document_title}</span>
                              </div>
                            </td>
                          <td className="py-3 px-2">
                            <div className="flex items-center gap-2">
                              {result.status === "completed" && <CheckCircle2 className="size-4 text-green-600" />}
                              {result.status === "failed" && <XCircle className="size-4 text-red-600" />}
                              {(result.status === "processing" || result.status === "pending") && <Spinner className="size-4" />}
                              <span className="capitalize">{result.status}</span>
                            </div>
                          </td>
                            <td className="py-3 px-2">
                              <span className="text-muted-foreground">
                                {result.completed_items} / {result.total_items} items
                              </span>
                            </td>
                          <td className="py-3 px-2">
                            <span className="text-muted-foreground">
                              {formatNumber(result.score_total ?? 0)} / {formatNumber(result.score_max ?? 0)}
                              {typeof result.score_percentage === "number" ? ` (${Math.round(result.score_percentage)}%)` : ""}
                            </span>
                          </td>
                          <td className="py-3 px-2">{formatTime(result.updated_at)}</td>
                            <td className="py-3 px-2 text-right">
                              <div className="flex items-center justify-end gap-2">
                                <Button
                                  variant="ghost"
                                  size="icon-sm"
                                  disabled={result.status !== "completed"}
                                  onClick={() => handleViewResult(result)}
                                  aria-label="View"
                                >
                                  <Eye className="size-4" />
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="icon-sm"
                                  onClick={() => confirmDeleteResult(result)}
                                  aria-label="Delete"
                                >
                                  <Trash className="size-4" />
                                </Button>
                              </div>
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={5} className="py-8 text-center text-muted-foreground">
                            {`No results found matching ${resultsSearch}`}
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>

                {/* Pagination */}
                {totalPages > 1 && (
                  <Pagination>
                    <PaginationContent>
                      <PaginationItem>
                        <PaginationPrevious
                          onClick={() => setResultsPage(p => Math.max(1, p - 1))}
                          className={resultsPage === 1 ? "pointer-events-none opacity-50" : "cursor-pointer"}
                          size="default"
                        />
                      </PaginationItem>

                      {Array.from({ length: totalPages }, (_, i) => i + 1).map(page => {
                        // Show first, last, current, and adjacent pages
                        if (
                          page === 1 ||
                          page === totalPages ||
                          (page >= resultsPage - 1 && page <= resultsPage + 1)
                        ) {
                          return (
                            <PaginationItem key={page}>
                              <PaginationLink
                                isActive={page === resultsPage}
                                onClick={() => setResultsPage(page)}
                                className="cursor-pointer"
                                size="icon"
                              >
                                {page}
                              </PaginationLink>
                            </PaginationItem>
                          )
                        }

                        // Show ellipsis
                        if (
                          (page === resultsPage - 2 && page > 1) ||
                          (page === resultsPage + 2 && page < totalPages)
                        ) {
                          return (
                            <PaginationItem key={page}>
                              <PaginationEllipsis />
                            </PaginationItem>
                          )
                        }

                        return null
                      })}

                      <PaginationItem>
                        <PaginationNext
                          onClick={() => setResultsPage(p => Math.min(totalPages, p + 1))}
                          className={resultsPage === totalPages ? "pointer-events-none opacity-50" : "cursor-pointer"}
                          size="default"
                        />
                      </PaginationItem>
                    </PaginationContent>
                  </Pagination>
                )}
              </div>
            ) : (
              <div className="text-center py-12">
                <FileText className="size-12 mx-auto text-muted-foreground mb-3" />
                <p className="text-muted-foreground mb-4">No documents analyzed yet</p>
                <Button onClick={() => setOpenDocumentDialog(true)}>
                  <PlayCircle className="size-4 mr-2" />
                  Run First Analysis
                </Button>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Delete confirmation dialog */}
      <Dialog open={openDeleteDialog} onOpenChange={setOpenDeleteDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Delete analysis result?</DialogTitle>
            <DialogDescription>
              This will permanently delete this document’s analysis result.
            </DialogDescription>
          </DialogHeader>
          <div className="text-sm">
            {deleteTarget ? (
              <>
                <p className="font-medium">{deleteTarget.document_title}</p>
                <p className="text-muted-foreground mt-1">Are you sure you want to continue?</p>
              </>
            ) : null}
          </div>
          <DialogFooter>
            <Button
              onClick={performDeleteResult}
              disabled={deleting}
              className="bg-destructive text-destructive-foreground hover:opacity-95"
            >
              {deleting ? "Deleting..." : "Delete"}
            </Button>
            <DialogClose asChild>
              <Button type="button" variant="outline">Cancel</Button>
            </DialogClose>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* View Items Sheet */}
      <Sheet open={openItemsDialog} onOpenChange={setOpenItemsDialog}>
        <SheetContent className="w-[400px] sm:w-[540px] overflow-y-auto">
          <SheetHeader>
            <SheetTitle>Analysis Items</SheetTitle>
            <SheetDescription>{analysis.items.length} questions</SheetDescription>
          </SheetHeader>
          <div className="mt-6 space-y-3">
            {analysis.items.map((item, index) => (
              <div key={index} className="border rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-sm font-semibold text-primary">
                    {index + 1}
                  </div>
                  <div className="flex-1">
                    <p className="font-medium">{item.question}</p>
                    {item.context && (
                      <p className="text-sm text-muted-foreground mt-1">{item.context}</p>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </SheetContent>
      </Sheet>
    </div>
  )
}

function formatTime(value?: string): string {
  if (!value) return "—"
  const d = parseUtcDate(value)
  if (!d) return value!
  return d.toLocaleDateString()
}

function formatNumber(n: number): string {
  try { return new Intl.NumberFormat().format(n) } catch { return String(n) }
}
