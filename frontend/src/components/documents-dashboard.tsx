"use client"

import * as React from "react"
import Link from "next/link"
import { FileText, Upload, CircleCheckBig, Eye, Download, Trash, Search, HardDrive } from "lucide-react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Empty, EmptyHeader, EmptyTitle, EmptyDescription, EmptyContent, EmptyMedia } from "@/components/ui/empty"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter, DialogClose } from "@/components/ui/dialog"
import { Pagination, PaginationContent, PaginationItem, PaginationLink, PaginationNext, PaginationPrevious } from "@/components/ui/pagination"
import { getBackendUrl } from "@/lib/env"
import { Spinner } from "@/components/ui/spinner"
import { toast } from "sonner"

type DocumentItem = {
  id: string
  title?: string
  filename?: string
  content_type?: string
  file_size?: number
  status?: string
  document_type?: string
  description?: string
  tags?: string[]
  metadata?: any
  page_count?: number | null
  image_count?: number | null
  processing_time?: number | null
  error_message?: string | null
  created_at?: string
  updated_at?: string
}

type Stats = {
  total_documents?: number
  documents_by_status?: Record<string, number>
  documents_by_type?: Record<string, number>
  total_file_size?: number
}

export default function DocumentsDashboard() {
  const [loadingStats, setLoadingStats] = React.useState(true)
  const [stats, setStats] = React.useState<Stats | null>(null)

  const [loadingDocs, setLoadingDocs] = React.useState(true)
  const [docs, setDocs] = React.useState<DocumentItem[]>([])
  const [totalDocs, setTotalDocs] = React.useState(0)
  const [query, setQuery] = React.useState("")
  const [debouncedQuery, setDebouncedQuery] = React.useState("")
  
  const [currentPage, setCurrentPage] = React.useState(1)
  const itemsPerPage = 10

  const [openUpload, setOpenUpload] = React.useState(false)
  const [uploading, setUploading] = React.useState(false)
  const [uploadError, setUploadError] = React.useState<string | null>(null)
  const [uploadTitle, setUploadTitle] = React.useState("")
  const [uploadDescription, setUploadDescription] = React.useState("")
  const [uploadFile, setUploadFile] = React.useState<File | null>(null)
  const backendUrl = getBackendUrl()
  // Delete dialog state
  const [openDelete, setOpenDelete] = React.useState(false)
  const [deleteTarget, setDeleteTarget] = React.useState<DocumentItem | null>(null)
  const [deleting, setDeleting] = React.useState(false)

  // Debounce query
  React.useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(query)
      setCurrentPage(1) // Reset to page 1 on search
    }, 300)
    return () => clearTimeout(timer)
  }, [query])

  // Load stats
  React.useEffect(() => {
    async function fetchStats() {
      try {
        const res = await fetch(`${backendUrl}/api/v1/documents/stats`, { headers: { accept: "application/json" } })
        const data = await res.json().catch(() => ({}))
        setStats(data ?? null)
      } catch (_) {
        setStats(null)
      } finally {
        setLoadingStats(false)
      }
    }
    fetchStats()
  }, [backendUrl])

  // Load documents with server-side pagination
  React.useEffect(() => {
    async function fetchDocs() {
      setLoadingDocs(true)
      try {
        const skip = (currentPage - 1) * itemsPerPage
        const searchParam = debouncedQuery ? `&search=${encodeURIComponent(debouncedQuery)}` : ''
        const res = await fetch(
          `${backendUrl}/api/v1/documents/?skip=${skip}&limit=${itemsPerPage}${searchParam}`, 
          { headers: { accept: "application/json" } }
        )
        const data = await res.json().catch(() => ({}))
        setDocs(Array.isArray(data?.documents) ? data.documents : [])
        setTotalDocs(data?.total ?? 0)
      } catch (_) {
        setDocs([])
        setTotalDocs(0)
      } finally {
        setLoadingDocs(false)
      }
    }
    fetchDocs()
  }, [backendUrl, currentPage, debouncedQuery])

  React.useEffect(() => {
    const ids = Array.isArray(docs) ? docs.map((d) => d.id).filter(Boolean) : []
    if (ids.length === 0) return

    function buildWsUrl(channels: string[]): string {
      const origin = backendUrl
      try {
        const base = new URL(origin)
        const proto = base.protocol === "https:" ? "wss" : "ws"
        const qs = new URLSearchParams({ channels: channels.join(",") })
        return `${proto}://${base.host}/api/v1/ws/status?${qs.toString()}`
      } catch {
        const proto = origin.startsWith("https") ? "wss" : "ws"
        const host = origin.replace(/^https?:\/\//, "")
        const qs = new URLSearchParams({ channels: channels.join(",") })
        return `${proto}://${host}/api/v1/ws/status?${qs.toString()}`
      }
    }

    function normalizeStatus(v?: string): string | undefined {
      if (!v) return v
      const s = String(v).toLowerCase()
      if (s.includes("documentstatus.")) {
        return s.split(".").pop()
      }
      return s
    }

    const channels = Array.from(new Set(ids.map((id) => `status:document:${id}`)))
    const wsUrl = buildWsUrl(channels)
    const ws = new WebSocket(wsUrl)

    ws.onmessage = async (evt) => {
      let payload: any = null
      try {
        payload = JSON.parse(evt.data)
      } catch {
        payload = null
      }
      if (!payload) return
      const rid = payload?.resource_id ?? payload?.document_id
      if (!rid) return
      const status = normalizeStatus(payload?.status)

      setDocs((prev) => {
        const next = prev.map((d) => {
          if (d.id !== rid) return d
          return {
            ...d,
            status: status ?? d.status,
            updated_at: new Date().toISOString(),
            error_message: payload?.error ?? d.error_message,
          }
        })
        return next
      })

      if (payload?.type === "processing_completed" || payload?.type === "processing_failed") {
        if (payload?.type === "processing_completed") {
          try {
            const name = (() => {
              const d = docs.find((x) => x.id === rid)
              return d?.filename || d?.title || rid
            })()
            toast.success("Document processing complete", {
              description: name,
              action: {
                label: "View",
                onClick: () => {
                  try { window.location.href = `/documents/${encodeURIComponent(rid)}` } catch { }
                },
              },
            })
          } catch { }
        }
        try {
          const sRes = await fetch(`${backendUrl}/api/v1/documents/stats`, { headers: { accept: "application/json" } })
          const sJson = await sRes.json().catch(() => ({}))
          setStats(sJson ?? null)
        } catch { }
      }
    }

    ws.onopen = () => { }
    ws.onerror = () => { }
    ws.onclose = () => { }

    return () => {
      try { ws.close() } catch { }
    }
  }, [backendUrl, docs.map((d) => d.id).join(",")])

  async function downloadDocument(doc: DocumentItem) {
    try {
      const name = doc.filename || doc.title || `${doc.id}.bin`
      const res = await fetch(`${backendUrl}/api/v1/documents/${doc.id}/download`, {
        headers: { accept: "application/octet-stream" },
      })
      if (!res.ok) throw new Error(`Download failed (${res.status})`)
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = name
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error(err)
      alert((err as any)?.message ?? "Unable to download document")
    }
  }

  function confirmDelete(doc: DocumentItem) {
    setDeleteTarget(doc)
    setOpenDelete(true)
  }

  async function performDelete() {
    if (!deleteTarget) return
    setDeleting(true)
    try {
      const res = await fetch(`${backendUrl}/api/v1/documents/${deleteTarget.id}`, {
        method: "DELETE",
        headers: { accept: "application/json" },
      })
      if (!res.ok) {
        let message = `Delete failed (${res.status})`
        try {
          const data = await res.json().catch(() => ({}))
          if (data?.detail) message = Array.isArray(data.detail) ? data.detail[0]?.msg ?? message : data.detail
          if (data?.message) message = data.message
        } catch (_) { }
        throw new Error(message)
      }
      // Optimistically update list
      setDocs((prev) => prev.filter((d) => d.id !== deleteTarget.id))
      setTotalDocs(prev => Math.max(0, prev - 1))
      setOpenDelete(false)
      setDeleteTarget(null)
    } catch (err: any) {
      alert(err?.message ?? "Unable to delete document")
    } finally {
      setDeleting(false)
    }
  }

  const uploaded = stats?.documents_by_status?.uploaded ?? 0
  const completed = stats?.documents_by_status?.completed ?? 0
  const total = stats?.total_documents ?? totalDocs
  const totalSize = stats?.total_file_size ?? sumBytes(docs)

  async function handleUpload(e?: React.FormEvent) {
    if (e) e.preventDefault()
    setUploadError(null)
    if (!uploadFile) {
      setUploadError("Please select a file to upload.")
      return
    }
    setUploading(true)
    try {
      const backendUrl = getBackendUrl()
      // Derive user_id from cached auth_user or explicit localStorage keys
      let userId: string | null = null
      try {
        const raw = typeof window !== "undefined" ? localStorage.getItem("auth_user") : null
        const obj = raw ? JSON.parse(raw) : null
        userId = (obj?.user_id ?? obj?.id ?? obj?.user?.id ?? obj?.user?.user_id ?? null) as string | null
        // Fallback: some backends store username as id
        if (!userId) {
          const cached = typeof window !== "undefined" ? localStorage.getItem("username") : null
          if (cached) userId = String(cached)
        }
      } catch (_) { }
      if (!userId) {
        setUploadError("Missing user_id. Please login first.")
        setUploading(false)
        return
      }

      const qs = new URLSearchParams()
      if (uploadTitle) qs.append("title", uploadTitle)
      if (uploadDescription) qs.append("description", uploadDescription)
      qs.append("user_id", String(userId))

      const formData = new FormData()
      formData.append("file", uploadFile)

      const res = await fetch(`${backendUrl}/api/v1/documents/upload?${qs.toString()}`, {
        method: "POST",
        headers: { accept: "application/json" },
        body: formData,
      })

      if (!res.ok) {
        let message = `Upload failed (${res.status})`
        try {
          const data = await res.json().catch(() => ({}))
          if (data?.detail) message = Array.isArray(data.detail) ? data.detail[0]?.msg ?? message : data.detail
          if (data?.message) message = data.message
        } catch (_) { }
        throw new Error(message)
      }

      // Try to refresh stats and list for up-to-date view
      try {
        const sRes = await fetch(`${backendUrl}/api/v1/documents/stats`, { headers: { accept: "application/json" } })
        const sJson = await sRes.json().catch(() => ({}))
        setStats(sJson ?? null)
        
        // Refresh current page
        const skip = (currentPage - 1) * itemsPerPage
        const searchParam = debouncedQuery ? `&search=${encodeURIComponent(debouncedQuery)}` : ''
        const dRes = await fetch(
          `${backendUrl}/api/v1/documents/?skip=${skip}&limit=${itemsPerPage}${searchParam}`, 
          { headers: { accept: "application/json" } }
        )
        const dJson = await dRes.json().catch(() => ({}))
        setDocs(Array.isArray(dJson?.documents) ? dJson.documents : [])
        setTotalDocs(dJson?.total ?? 0)
      } catch (_) {
        // ignore refresh errors
      }

      // Reset form and close
      setUploadFile(null)
      setUploadTitle("")
      setUploadDescription("")
      setOpenUpload(false)
    } catch (err: any) {
      setUploadError(err?.message ?? "Unexpected error during upload")
    } finally {
      setUploading(false)
    }
  }
  
  const totalPages = Math.ceil(totalDocs / itemsPerPage)

  return (
    <div className="flex flex-1 flex-col gap-4">
      {/* Metrics */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Total Documents"
          value={formatNumber(total)}
          description="Active document records"
          icon={<FileText className="size-5" />}
        />
        <MetricCard
          title="Uploaded"
          value={formatNumber(uploaded)}
          description="Awaiting processing"
          icon={<Upload className="size-5" />}
        />
        <MetricCard
          title="Completed"
          value={formatNumber(completed)}
          description="Processed and ready"
          icon={<CircleCheckBig className="size-5" />}
        />
        <MetricCard
          title="Total Size"
          value={formatBytes(totalSize)}
          description="Aggregate file size"
          icon={<HardDrive className="size-5" />}
        />
      </div>


      {/* Documents table */}
      <Card>
        <CardHeader className="border-b">
          <CardTitle>All Documents</CardTitle>
          <CardDescription>{formatNumber(totalDocs)} documents found</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mb-4 flex items-center gap-2">
            <div className="relative w-full max-w-md">
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
              <Input
                placeholder="Search documents..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="pl-8"
              />
            </div>
            <Dialog open={openUpload} onOpenChange={setOpenUpload}>
              <DialogTrigger asChild>
                <Button>Upload Document</Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-md">
                <DialogHeader>
                  <DialogTitle>Upload Document</DialogTitle>
                  <DialogDescription>Upload a single file with optional title and description.</DialogDescription>
                </DialogHeader>
                <form onSubmit={handleUpload} className="flex flex-col gap-4">
                  <div>
                    <label className="text-sm font-medium">Title</label>
                    <Input value={uploadTitle} onChange={(e) => setUploadTitle(e.target.value)} placeholder="Optional title" />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Description</label>
                    <Input value={uploadDescription} onChange={(e) => setUploadDescription(e.target.value)} placeholder="Optional description" />
                  </div>
                  <div>
                    <label className="text-sm font-medium">File</label>
                    <Input type="file" accept="application/pdf,image/*,.txt,.doc,.docx" onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)} />
                    <p className="mt-1 text-xs text-muted-foreground">Single file only.</p>
                  </div>
                  {uploadError ? <p className="text-sm text-destructive" role="alert">{uploadError}</p> : null}
                  <DialogFooter>
                    <Button type="submit" disabled={uploading}>{uploading ? "Uploading..." : "Submit"}</Button>
                    <DialogClose asChild>
                      <Button type="button" variant="outline">Cancel</Button>
                    </DialogClose>
                  </DialogFooter>
                </form>
              </DialogContent>
            </Dialog>
            {/* Delete confirmation dialog */}
            <Dialog open={openDelete} onOpenChange={setOpenDelete}>
              <DialogContent className="sm:max-w-md">
                <DialogHeader>
                  <DialogTitle>Delete document?</DialogTitle>
                  <DialogDescription>
                    This will mark the document as deleted and remove it from the list.
                  </DialogDescription>
                </DialogHeader>
                <div className="text-sm">
                  {deleteTarget ? (
                    <>
                      <p className="font-medium">{deleteTarget.filename || deleteTarget.title || deleteTarget.id}</p>
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

          {loadingDocs ? (
            <div className="flex flex-col gap-2">
              {Array(6).fill(0).map((_, i) => (
                <Skeleton key={i} className="h-9 w-full" />
              ))}
            </div>
          ) : docs.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-muted-foreground border-b">
                    <th className="py-3 px-2 text-left font-medium">Name</th>
                    <th className="py-3 px-2 text-left font-medium">Description</th>
                    <th className="py-3 px-2 text-left font-medium">Pages</th>
                    <th className="py-3 px-2 text-left font-medium">Status</th>
                    <th className="py-3 px-2 text-left font-medium">Last Updated</th>
                    <th className="py-3 px-2 text-right font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {docs.map((doc) => (
                    <tr key={doc.id} className="border-b last:border-0">
                      <td className="py-3 px-2">
                        <div className="flex items-center gap-2">
                          <FileText className="size-4 text-muted-foreground" />
                          <span className="font-medium truncate max-w-[22ch]">
                            {doc.filename || doc.title || doc.id}
                          </span>
                        </div>
                        {doc.title && (
                          <div className="text-xs text-muted-foreground truncate max-w-[30ch]">{doc.title}</div>
                        )}
                      </td>
                      <td className="py-3 px-2">
                        <span className="truncate max-w-[40ch] block">
                          {doc.description || "—"}
                        </span>
                      </td>
                      <td className="py-3 px-2">
                        {formatNumber(resolvePageCount(doc))}
                      </td>
                      <td className="py-3 px-2">
                        <span className="rounded bg-muted px-2 py-0.5 text-xs capitalize inline-flex items-center gap-1">
                          {((doc.status ?? "").toLowerCase() === "processing") ? (
                            <Spinner className="size-3" />
                          ) : (
                            <StatusIcon status={doc.status} />
                          )}
                          {doc.status ?? "unknown"}
                        </span>
                      </td>
                      <td className="py-3 px-2">
                        {formatTime(doc.updated_at)}
                      </td>
                      <td className="py-3 px-2 text-right">
                        <div className="flex items-center justify-end gap-2 text-muted-foreground">
                          {doc.status === "completed" ? (
                            <Link href={`/documents/${doc.id}`} prefetch>
                              <Button variant="ghost" size="icon-sm" aria-label="View"><Eye className="size-4" /></Button>
                            </Link>
                          ) : (
                            <Button variant="ghost" size="icon-sm" aria-label="View" disabled>
                              <Eye className="size-4" />
                            </Button>
                          )}
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            aria-label="Download"
                            onClick={() => downloadDocument(doc)}
                          >
                            <Download className="size-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            aria-label="Delete"
                            onClick={() => confirmDelete(doc)}
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
                        <PaginationPrevious size="default"
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
                              <PaginationLink size="default"
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
                        <PaginationNext size="default"
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
                <EmptyMedia variant="icon"><FileText className="size-6" /></EmptyMedia>
                <EmptyTitle>No documents</EmptyTitle>
                <EmptyDescription>Try adjusting your search or upload a document.</EmptyDescription>
              </EmptyHeader>
              <EmptyContent>
                <Link href="#" prefetch>
                  <Button>Upload Document</Button>
                </Link>
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

function StatusIcon({ status }: { status?: string }) {
  const s = (status ?? "").toLowerCase()
  let color = "text-muted-foreground"
  let Icon: React.ComponentType<React.ComponentProps<"svg">> = FileText

  switch (s) {
    case "completed":
      Icon = CircleCheckBig
      color = "text-green-600"
      break
    case "uploaded":
      Icon = Upload
      color = "text-blue-600"
      break
    default:
      Icon = FileText
      color = "text-muted-foreground"
  }
  return <Icon className={`size-4 ${color}`} />
}

function sumBytes(items: DocumentItem[]): number {
  return items.reduce((acc, d) => acc + (Number(d.file_size) || Number(d.metadata?.file_size) || 0), 0)
}

function resolvePageCount(d: DocumentItem): number {
  return Number(
    d.page_count ?? d.metadata?.page_count ?? 0
  ) || 0
}

function formatNumber(n: number): string {
  try { return new Intl.NumberFormat().format(n) } catch { return String(n) }
}

function formatBytes(bytes?: number): string {
  const b = Number(bytes || 0)
  if (b < 1024) return `${b} B`
  const units = ["KB", "MB", "GB", "TB"]
  let u = -1
  let v = b
  while (v >= 1024 && u < units.length - 1) { v /= 1024; u++ }
  return `${v.toFixed(1)} ${units[u]}`
}

function formatTime(value?: string): string {
  if (!value) return "—"
  const normalized = value.replace(/(\.\d{3})\d+$/, "$1").replace(/\.\d+$/, (m) => (m.length > 4 ? m.slice(0, 4) : m))
  const d = new Date(normalized)
  if (isNaN(d.getTime())) return value
  return d.toLocaleDateString()
}