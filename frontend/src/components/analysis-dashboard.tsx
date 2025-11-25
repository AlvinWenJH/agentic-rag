"use client"

import * as React from "react"
import Link from "next/link"
import { Plus, Eye, Trash, Search, ClipboardList, ListChecks, Timer } from "lucide-react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Empty, EmptyHeader, EmptyTitle, EmptyDescription, EmptyContent, EmptyMedia } from "@/components/ui/empty"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter, DialogClose } from "@/components/ui/dialog"
import { Pagination, PaginationContent, PaginationItem, PaginationLink, PaginationNext, PaginationPrevious } from "@/components/ui/pagination"
import { getBackendUrl } from "@/lib/env"
import { parseUtcDate } from "@/lib/utils"

type AnalysisItem = {
  id: string
  title: string
  description: string
  items: { question: string; context?: string; order: number }[]
  status: string
  tags: string[]
  user_id?: string
  created_at: string
  updated_at: string
}

type AnalysisStats = {
  total_documents: number
  total_input_token_usage: number
  total_output_token_usage: number
  total_analysis_time: number
}

export default function AnalysisDashboard() {
  const [loading, setLoading] = React.useState(true)
  const [analyses, setAnalyses] = React.useState<AnalysisItem[]>([])
  const [totalAnalyses, setTotalAnalyses] = React.useState(0)
  const [query, setQuery] = React.useState("")
  const [debouncedQuery, setDebouncedQuery] = React.useState("")
  const [stats, setStats] = React.useState<AnalysisStats | null>(null)

  const [currentPage, setCurrentPage] = React.useState(1)
  const itemsPerPage = 10

  const [openDelete, setOpenDelete] = React.useState(false)
  const [deleteTarget, setDeleteTarget] = React.useState<AnalysisItem | null>(null)
  const [deleting, setDeleting] = React.useState(false)
  const backendUrl = getBackendUrl()

  // Debounce query
  React.useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(query)
      setCurrentPage(1) // Reset to page 1 on search
    }, 300)
    return () => clearTimeout(timer)
  }, [query])

  // Load analyses with server-side pagination
  React.useEffect(() => {
    async function fetchData() {
      setLoading(true)
      try {
        const skip = (currentPage - 1) * itemsPerPage
        const searchParam = debouncedQuery ? `&search=${encodeURIComponent(debouncedQuery)}` : ''
        const res = await fetch(
          `${backendUrl}/api/v1/analysis/?skip=${skip}&limit=${itemsPerPage}${searchParam}`,
          { headers: { accept: "application/json" } }
        )
        const data = await res.json().catch(() => ({}))
        setAnalyses(Array.isArray(data?.analyses) ? data.analyses : [])
        setTotalAnalyses(data?.total ?? 0)
      } catch {
        setAnalyses([])
        setTotalAnalyses(0)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [backendUrl, currentPage, debouncedQuery])

  React.useEffect(() => {
    async function fetchStats() {
      try {
        const res = await fetch(`${backendUrl}/api/v1/analysis/results/stats`, { headers: { accept: "application/json" } })
        const data = await res.json().catch(() => ({}))
        if (typeof data?.total_documents === "number") {
          setStats(data as AnalysisStats)
        } else {
          setStats(null)
        }
      } catch {
        setStats(null)
      }
    }
    fetchStats()
  }, [backendUrl])

  async function handleCreateAnalysis() {
    // Navigate to draft page instead of creating immediately
    window.location.href = "/analysis/draft"
  }

  function confirmDelete(analysis: AnalysisItem) {
    setDeleteTarget(analysis)
    setOpenDelete(true)
  }

  async function performDelete() {
    if (!deleteTarget) return
    setDeleting(true)
    try {
      const res = await fetch(`${backendUrl}/api/v1/analysis/${deleteTarget.id}`, {
        method: "DELETE",
        headers: { accept: "application/json" },
      })
      if (!res.ok) throw new Error(`Delete failed (${res.status})`)

      // Optimistically update list
      setAnalyses((prev) => prev.filter((a) => a.id !== deleteTarget.id))
      setTotalAnalyses(prev => Math.max(0, prev - 1))
      setOpenDelete(false)
      setDeleteTarget(null)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err)
      alert(message || "Unable to delete analysis")
    } finally {
      setDeleting(false)
    }
  }

  // Stats - Note: These are now only accurate for the current page if we don't have a separate stats API
  // Ideally we should have a stats API for analysis too, but for now we'll just show counts based on what we have or remove them if inaccurate.
  // Given the previous implementation calculated these from ALL analyses, and now we only have a page, these will be wrong.
  // However, the user didn't ask for a stats API. I will keep the total count correct from the API response.
  // The status breakdown cards will be inaccurate if I only use the current page.
  // I will remove the status breakdown cards for now or just show total, as fetching ALL just for stats defeats the purpose of pagination.
  // Actually, looking at the previous code, it was calculating stats from `analyses` which contained ALL items.
  // Since I can't easily get stats without fetching all, and I shouldn't fetch all, I will simplify the metrics to just show Total.
  // Or I could fetch stats separately if an endpoint existed. It doesn't seem to exist in `analysis.py` based on my previous read.
  // I'll stick to showing Total Analyses for now to be safe and accurate.

  const total = totalAnalyses
  const totalItems = analyses.reduce((sum, a) => sum + (a.items?.length ?? 0), 0)
  const totalPages = Math.ceil(total / itemsPerPage)

  return (
    <div className="flex flex-1 flex-col gap-4">
      {/* Metrics */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Total Analyses"
          value={formatNumber(total)}
          description="All analysis projects"
          icon={<ClipboardList className="size-5" />}
        />
        <MetricCard
          title="Total Items"
          value={formatNumber(totalItems)}
          description="Analysis items on this page"
          icon={<ListChecks className="size-5" />}
        />
        <MetricCard
          title="Documents Analyzed"
          value={formatNumber(stats?.total_documents ?? 0)}
          description="Across all analyses"
          icon={<ClipboardList className="size-5" />}
        />
        <MetricCard
          title="Total Analysis Time"
          value={formatDuration(stats?.total_analysis_time ?? 0)}
          description="Accumulated processing time"
          icon={<Timer className="size-5" />}
        />
        {/* 
          Other metrics removed as they cannot be accurately calculated with server-side pagination 
          without a dedicated stats endpoint.
        */}
      </div>

      {/* Analyses table */}
      <Card>
        <CardHeader className="border-b">
          <CardTitle>All Analyses</CardTitle>
          <CardDescription>{formatNumber(total)} analyses found</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mb-4 flex items-center gap-2">
            <div className="relative w-full max-w-md">
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
              <Input
                placeholder="Search analyses..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="pl-8"
              />
            </div>
            <Button onClick={handleCreateAnalysis}>
              <Plus className="size-4 mr-2" />
              Create Analysis
            </Button>

            {/* Delete confirmation dialog */}
            <Dialog open={openDelete} onOpenChange={setOpenDelete}>
              <DialogContent className="sm:max-w-md">
                <DialogHeader>
                  <DialogTitle>Delete analysis?</DialogTitle>
                  <DialogDescription>
                    This will permanently delete the analysis project.
                  </DialogDescription>
                </DialogHeader>
                <div className="text-sm">
                  {deleteTarget ? (
                    <>
                      <p className="font-medium">{deleteTarget.title}</p>
                      <p className="text-muted-foreground mt-1">Are you sure you want to continue?</p>
                    </>
                  ) : null}
                </div>
                <DialogFooter>
                  <Button onClick={performDelete} disabled={deleting} className="bg-destructive text-destructive-foreground hover:opacity-95">
                    {deleting ? "Deleting..." : "Delete"}
                  </Button>
                  <DialogClose asChild>
                    <Button type="button" variant="outline">Cancel</Button>
                  </DialogClose>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>

          {loading ? (
            <div className="flex flex-col gap-2">
              {Array.from(Array(6).keys()).map((i) => (
                <Skeleton key={i} className="h-9 w-full" />
              ))}
            </div>
          ) : analyses.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-muted-foreground border-b">
                    <th className="py-3 px-2 text-left font-medium">Title</th>
                    <th className="py-3 px-2 text-left font-medium">Description</th>
                    <th className="py-3 px-2 text-left font-medium">Items</th>
                    <th className="py-3 px-2 text-left font-medium">Last Updated</th>
                    <th className="py-3 px-2 text-right font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {analyses.map((analysis) => (
                    <tr key={analysis.id} className="border-b last:border-0">
                      <td className="py-3 px-2">
                        <div className="flex items-center gap-2">
                          <ClipboardList className="size-4 text-muted-foreground" />
                          <span className="font-medium truncate max-w-[22ch]">
                            {analysis.title}
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-2">
                        <span className="truncate max-w-[40ch] block">
                          {analysis.description || "—"}
                        </span>
                      </td>
                      <td className="py-3 px-2">
                        {formatNumber(analysis.items?.length ?? 0)}
                      </td>
                      <td className="py-3 px-2">
                        {formatTime(analysis.updated_at)}
                      </td>
                      <td className="py-3 px-2 text-right">
                        <div className="flex items-center justify-end gap-2 text-muted-foreground">
                          <Link href={`/analysis/${analysis.id}`} prefetch>
                            <Button variant="ghost" size="icon-sm" aria-label="View">
                              <Eye className="size-4" />
                            </Button>
                          </Link>
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            aria-label="Delete"
                            onClick={() => confirmDelete(analysis)}
                          >
                            <Trash className="size-4" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* Pagination Controls */}
              {totalPages > 1 && (
                <div className="flex items-center justify-center pt-4">
                  <Pagination>
                    <PaginationContent>
                      <PaginationItem>
                        <PaginationPrevious
                          size="default"
                          onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                          className={currentPage === 1 ? "pointer-events-none opacity-50" : "cursor-pointer"}
                        />
                      </PaginationItem>

                      {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => {
                        // Show first, last, current, and adjacent pages
                        if (
                          page === 1 ||
                          page === totalPages ||
                          (page >= currentPage - 1 && page <= currentPage + 1)
                        ) {
                          return (
                            <PaginationItem key={page}>
                              <PaginationLink
                                size="default"
                                onClick={() => setCurrentPage(page)}
                                isActive={currentPage === page}
                                className="cursor-pointer"
                              >
                                {page}
                              </PaginationLink>
                            </PaginationItem>
                          )
                        }

                        // Show ellipsis
                        if (
                          (page === currentPage - 2 && page > 1) ||
                          (page === currentPage + 2 && page < totalPages)
                        ) {
                          return (
                            <PaginationItem key={page}>
                              <span className="flex h-9 w-9 items-center justify-center">...</span>
                            </PaginationItem>
                          )
                        }

                        return null
                      })}

                      <PaginationItem>
                        <PaginationNext
                          size="default"
                          onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                          className={currentPage === totalPages ? "pointer-events-none opacity-50" : "cursor-pointer"}
                        />
                      </PaginationItem>
                    </PaginationContent>
                  </Pagination>
                </div>
              )}
            </div>
          ) : (
            <Empty>
              <EmptyHeader>
                <EmptyMedia variant="icon"><ClipboardList className="size-6" /></EmptyMedia>
                <EmptyTitle>No analyses</EmptyTitle>
                <EmptyDescription>Try adjusting your search or create an analysis.</EmptyDescription>
              </EmptyHeader>
              <EmptyContent>
                <Button onClick={handleCreateAnalysis}>
                  <Plus className="size-4 mr-2" />
                  Create Analysis
                </Button>
              </EmptyContent>
            </Empty>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function MetricCard({ title, value, description, icon }: { title: string; value: string; description: string; icon: React.ReactNode }) {
  return (
    <Card className="py-3">
      <CardHeader className="border-b !pb-2 gap-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-m font-medium text-foreground">{title}</CardTitle>
          <div className="rounded-md bg-muted p-2 text-muted-foreground">
            {icon}
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="text-xl font-semibold">{value}</div>
        <CardDescription className="mt-1">{description}</CardDescription>
      </CardContent>
    </Card>
  )
}

function formatNumber(n: number): string {
  try { return new Intl.NumberFormat().format(n) } catch { return String(n) }
}

function formatTime(value?: string): string {
  if (!value) return "—"
  const d = parseUtcDate(value)
  if (!d) return value!
  return d.toLocaleDateString()
}

function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m ${sec}s`
  return `${sec}s`
}
