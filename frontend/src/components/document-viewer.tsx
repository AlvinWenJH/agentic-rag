"use client"

import * as React from "react"
import { useEffect, useMemo, useRef, useState } from "react"
import { useParams, useSearchParams } from "next/navigation"
import { FileText, HardDrive, Files, ListTree, ZoomIn, ZoomOut, RefreshCcw, Download, GitBranch, LibraryBig } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardAction } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { getBackendUrl } from "@/lib/env"
import { parseUtcDate } from "@/lib/utils"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { ReactFlow, Background, Controls, Handle, Position, useReactFlow, type Node, type Edge } from "@xyflow/react"
import "@xyflow/react/dist/style.css"

type DocDetail = {
  id: string
  title?: string
  filename?: string
  description?: string
  file_size?: number
  page_count?: number | null
  metadata?: { page_count?: number }
  created_at?: string
  updated_at?: string
}

type TreeNode = {
  title?: string
  summary?: string
  children?: TreeNode[]
  pages?: number[]
  node_type?: string
}

type TreeResponse = {
  node_counts?: { total?: number; L1?: number; L2?: number; L3?: number }
  tree_data?: TreeNode
}

function filterTreeByPage(node: TreeNode, page: number): TreeNode | null {
  const selfMatches = Array.isArray(node.pages) && node.pages.includes(page)
  const children = Array.isArray(node.children) ? node.children : []
  const filteredChildren: TreeNode[] = []
  for (const child of children) {
    const filtered = filterTreeByPage(child, page)
    if (filtered) filteredChildren.push(filtered)
  }
  if (selfMatches || filteredChildren.length) {
    return { ...node, children: filteredChildren }
  }
  return null
}

function formatBytes(bytes?: number | null) {
  const b = typeof bytes === "number" ? bytes : 0
  if (b < 1024) return `${b} B`
  const u = ["KB", "MB", "GB", "TB"]
  let i = -1
  let v = b
  do { v /= 1024; i++ } while (v >= 1024 && i < u.length - 1)
  return `${v.toFixed(1)} ${u[i]}`
}

function formatNumber(n?: number | null) {
  const v = typeof n === "number" ? n : 0
  return Intl.NumberFormat().format(v)
}

function resolvePageCount(doc?: DocDetail | null): number {
  if (!doc) return 0
  return (doc.page_count ?? doc.metadata?.page_count ?? 0) || 0
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

function Badge({ children }: { children: React.ReactNode }) {
  return <span className="rounded bg-muted px-2 py-0.5 text-xs">{children}</span>
}

export default function DocumentViewer({ docId }: { docId?: string }) {
  const backendUrl = getBackendUrl()
  const params = useParams() as Record<string, string | string[]>
  const searchParams = useSearchParams()
  const routeId = typeof params?.id === "string" ? params.id : Array.isArray(params?.id) ? params.id[0] : undefined
  const queryId = searchParams?.get("id") ?? undefined
  const resolvedId = docId ?? routeId ?? queryId
  const [doc, setDoc] = useState<DocDetail | null>(null)
  const [tree, setTree] = useState<TreeResponse | null>(null)
  const [loadingMeta, setLoadingMeta] = useState(true)
  const [loadingTree, setLoadingTree] = useState(true)
  const [errorMeta, setErrorMeta] = useState<string | null>(null)
  const [errorTree, setErrorTree] = useState<string | null>(null)
  const [conceptTab, setConceptTab] = useState<"tree" | "graph">("tree")

  const pageCount = useMemo(() => resolvePageCount(doc), [doc])

  // Images pagination state
  const [loadedUntil, setLoadedUntil] = useState<number>(0)
  const [images, setImages] = useState<Record<number, string>>({})
  const containerRef = useRef<HTMLDivElement>(null)
  const batchSize = 10
  const [containerMinHeight, setContainerMinHeight] = useState<number | undefined>(undefined)
  const [previewPage, setPreviewPage] = useState<number | null>(null)
  const [previewZoom, setPreviewZoom] = useState<number>(1.5)
  const previewContainerRef = useRef<HTMLDivElement>(null)
  const previewImgRef = useRef<HTMLImageElement>(null)
  const [previewFitWidth, setPreviewFitWidth] = useState<number>(0)
  const [previewPan, setPreviewPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState<boolean>(false)
  const dragStartRef = useRef<{ x: number; y: number } | null>(null)
  const dragPanStartRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 })
  const previewAspectRef = useRef<number | null>(null)
  const [selectedPage, setSelectedPage] = useState<number | null>(() => {
    const sp = searchParams?.get("page")
    const n = sp ? parseInt(sp, 10) : NaN
    return Number.isFinite(n) && n >= 1 ? n : 1
  })
  const [selectedPageSource, setSelectedPageSource] = useState<"input" | "scroll" | "init" | null>(
    () => (selectedPage ? "init" : null)
  )
  const filteredTree = useMemo(() => {
    const td = tree?.tree_data
    if (!td) return null
    if (!selectedPage) return td
    return filterTreeByPage(td, selectedPage)
  }, [tree, selectedPage])

  useEffect(() => {
    if (!resolvedId) {
      setErrorMeta("Missing document id")
      setErrorTree("Missing document id")
      setLoadingMeta(false)
      setLoadingTree(false)
      return
    }
    async function fetchMeta() {
      setLoadingMeta(true)
      setErrorMeta(null)
      try {
        const res = await fetch(`${backendUrl}/api/v1/documents/${resolvedId}`, { headers: { accept: "application/json" } })
        const json = await res.json().catch(() => ({}))
        const item: DocDetail = Array.isArray(json?.documents) ? json.documents?.[0] : json
        setDoc(item ?? null)
      } catch (err) {
        setErrorMeta("Failed to load document details")
      } finally {
        setLoadingMeta(false)
      }
    }

    async function fetchTree() {
      setLoadingTree(true)
      setErrorTree(null)
      try {
        const res = await fetch(`${backendUrl}/api/v1/documents/${resolvedId}/tree`, { headers: { accept: "application/json" } })
        const json = await res.json().catch(() => ({}))
        setTree(json ?? null)
      } catch (err) {
        setErrorTree("Failed to load document tree")
      } finally {
        setLoadingTree(false)
      }
    }

    fetchMeta()
    fetchTree()
  }, [backendUrl, resolvedId])

  useEffect(() => {
    if (pageCount > 0 && loadedUntil === 0) {
      loadBatch(1)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pageCount])

  function imageUrl(page: number) {
    const base = `${backendUrl}/api/v1/documents/${resolvedId}`
    // New API endpoint that returns JSON: { "page_base64": "..." }
    return `${base}/page/${page}`
  }

  async function loadPageImage(page: number) {
    const url = imageUrl(page)
    try {
      const res = await fetch(url, { headers: { accept: "application/json" } })
      if (!res.ok) throw new Error(`Failed to fetch page ${page}`)
      const json = await res.json().catch(() => null)
      const b64: string | undefined = json?.page_base64
      if (!b64) {
        setImages((prev) => ({ ...prev, [page]: "" }))
        return
      }
      const src = b64.startsWith("data:image") ? b64 : `data:image/png;base64,${b64}`
      setImages((prev) => ({ ...prev, [page]: src }))
    } catch (_) {
      // if failed, mark placeholder
      setImages((prev) => ({ ...prev, [page]: "" }))
    }
  }

  // Compute fit-to-viewport width for full-screen preview so 100% shows whole page without scrollbars
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
      // clamp pan based on new bounds
      clampPan(previewPan.x, previewPan.y)
    } catch (_) {
      // ignore
    }
  }

  useEffect(() => {
    if (previewPage !== null) {
      // set default zoom to 150% when opening a page
      setPreviewZoom(1.5)
      setPreviewPan({ x: 0, y: 0 })
      // slight delay to ensure DOM has measured sizes
      const t = setTimeout(() => recomputePreviewFit(), 0)
      const onResize = () => recomputePreviewFit()
      window.addEventListener("resize", onResize)
      return () => { clearTimeout(t); window.removeEventListener("resize", onResize) }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewPage])

  // Helper: clamp pan to within the visible bounds
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

  // Re-clamp pan whenever zoom changes or fit width updates
  useEffect(() => {
    clampPan(previewPan.x, previewPan.y)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewZoom, previewFitWidth])

  async function loadBatch(start: number) {
    const end = Math.min(pageCount, start + batchSize - 1)
    if (end < start) return
    const pages = Array.from({ length: end - start + 1 }, (_, i) => start + i)
    await Promise.all(pages.map((p) => loadPageImage(p)))
    setLoadedUntil(end)
  }

  async function ensureLoadedUpTo(target: number) {
    const t = Math.min(pageCount, Math.max(1, target))
    if (t <= loadedUntil) return
    const pages = Array.from({ length: t - loadedUntil }, (_, i) => loadedUntil + 1 + i)
    await Promise.all(pages.map((p) => loadPageImage(p)))
    setLoadedUntil(t)
  }

  function onImageLoad(e: React.SyntheticEvent<HTMLImageElement>) {
    const img = e.currentTarget
    // Use the rendered image height to ensure the scroll container
    // can display at least one full page without cropping.
    const h = img.clientHeight
    if (h > 0) setContainerMinHeight(h + 16) // account for padding around the image
  }

  useEffect(() => {
    if (previewPage) {
      void ensureLoadedUpTo(previewPage)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewPage])

  function onScroll() {
    const el = containerRef.current
    if (!el) return
    if (el.scrollTop + el.clientHeight >= el.scrollHeight - 80) {
      if (loadedUntil < pageCount) {
        void loadBatch(loadedUntil + 1)
      }
    }

    // Determine which page is nearest to the top of the container
    const nodes = el.querySelectorAll('[id^="page-"]')
    if (nodes.length) {
      const containerTop = el.getBoundingClientRect().top
      let nearest = selectedPage ?? 1
      let minDist = Number.POSITIVE_INFINITY
      nodes.forEach((node) => {
        const rect = (node as HTMLElement).getBoundingClientRect()
        const dist = Math.abs(rect.top - containerTop)
        if (dist < minDist) {
          minDist = dist
          const id = (node as HTMLElement).id
          const num = parseInt(id.replace("page-", ""), 10)
          if (Number.isFinite(num)) nearest = num
        }
      })
      if (nearest !== selectedPage) {
        setSelectedPage(nearest)
        setSelectedPageSource("scroll")
      }
    }
  }

  useEffect(() => {
    // Only auto-scroll when the page selection comes from input or initial query
    if (!selectedPage || pageCount === 0 || (selectedPageSource !== "input" && selectedPageSource !== "init")) return
    void (async () => {
      await ensureLoadedUpTo(selectedPage)
      const el = document.getElementById(`page-${selectedPage}`)
      const cont = containerRef.current
      if (el && cont) {
        // Scroll the container itself to the target page, so the internal
        // scroll state remains intact and the scrollbar doesn't disappear.
        const top = (el as HTMLElement).offsetTop - (cont as HTMLElement).offsetTop
        cont.scrollTo({ top, behavior: "smooth" })
      }
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPage, pageCount, selectedPageSource])

  function download() {
    if (!resolvedId) return alert("Missing document id")
    const name = doc?.filename || doc?.title || `${resolvedId}.pdf`
    fetch(`${backendUrl}/api/v1/documents/${resolvedId}/download`, { headers: { accept: "application/octet-stream" } })
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status))
        return r.blob()
      })
      .then((blob) => {
        const url = URL.createObjectURL(blob)
        const a = document.createElement("a")
        a.href = url
        a.download = name ?? `${resolvedId}.bin`
        document.body.appendChild(a)
        a.click()
        a.remove()
        URL.revokeObjectURL(url)
      })
      .catch(() => alert("Download failed"))
  }

  const l1 = tree?.node_counts?.L1 ?? 0
  const l2 = tree?.node_counts?.L2 ?? 0
  const l3 = tree?.node_counts?.L3 ?? 0
  const totalNodes = (l1 + l2 + l3) || (tree?.node_counts?.total ?? 0)

  function SmallMetric({ value, label }: { value: React.ReactNode; label: string }) {
    return (
      <div className="rounded-lg border p-4 text-center">
        <div className="text-3xl font-bold tracking-tight">{value}</div>
        <div className="mt-1 text-xs text-muted-foreground">{label}</div>
      </div>
    )
  }

  return (
    <div className="flex flex-1 flex-col gap-4">
      {/* Top row: Quick Actions + Overview */}
      <div className="grid gap-4 lg:grid-cols-3">
        {/* Quick Actions */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <LibraryBig className="size-5 text-muted-foreground" />
              Quick Actions
            </CardTitle>
            <CardDescription className="sr-only">Actions for this document</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <Button onClick={download} className="bg-foreground text-background hover:opacity-95 px-3">
              <div className="grid grid-cols-[24px_1fr_24px] items-center w-full">
                <Download className="size-4" />
                <span className="justify-self-center">Download Document</span>
                <span className="size-4" aria-hidden="true" />
              </div>
            </Button>
            <a
              href={`${backendUrl}/api/v1/documents/${resolvedId}`}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-md border px-3 py-2 text-sm hover:bg-muted grid grid-cols-[24px_1fr_24px] items-center"
            >
              <FileText className="size-4" />
              <span className="justify-self-center">View Details (JSON)</span>
              <span className="size-4" aria-hidden="true" />
            </a>
            <a
              href={`${backendUrl}/api/v1/documents/${resolvedId}/tree`}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-md border px-3 py-2 text-sm hover:bg-muted grid grid-cols-[24px_1fr_24px] items-center"
            >
              <GitBranch className="size-4" />
              <span className="justify-self-center">View Tree (JSON)</span>
              <span className="size-4" aria-hidden="true" />
            </a>
            <a
              href={`${backendUrl}/api/v1/documents/${resolvedId}/download`}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-md border px-3 py-2 text-sm hover:bg-muted grid grid-cols-[24px_1fr_24px] items-center"
            >
              <HardDrive className="size-4" />
              <span className="justify-self-center">Show in MinIO</span>
              <span className="size-4" aria-hidden="true" />
            </a>
          </CardContent>
        </Card>

        {/* Document Overview */}
        <Card className="lg:col-span-2 gap-0">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="size-5 text-muted-foreground" />
              {doc?.title || doc?.filename || resolvedId || "Unknown Document"}
            </CardTitle>
            <CardDescription>
              {doc?.description || "No description"}
            </CardDescription>
          </CardHeader>
          <CardContent className="py-4">
            <div className="mt-2 mb-4 flex flex-wrap items-center gap-2">
              <Badge>Hierarchy {formatNumber(l1)}</Badge>
              <Badge>Topic {formatNumber(l2)}</Badge>
              <Badge>Detail {formatNumber(l3)}</Badge>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <div className="text-sm text-muted-foreground">Created</div>
                <div className="mt-1 text-sm">{formatDateTime(doc?.created_at)}</div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground">Last Updated</div>
                <div className="mt-1 text-sm">{formatDateTime(doc?.updated_at)}</div>
              </div>
            </div>
            <div className="my-3 border-b" />
            <div className="grid gap-4 sm:grid-cols-3">
              <SmallMetric value={formatNumber(pageCount)} label="Pages" />
              <SmallMetric value={formatBytes(doc?.file_size || 0)} label="Storage Size" />
              <SmallMetric value={formatNumber(totalNodes)} label="Total Nodes" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main two-column layout */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Left: images */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Files className="size-5 text-muted-foreground" /> Pages ({formatNumber(pageCount)})
            </CardTitle>
            <CardAction>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  min={1}
                  max={pageCount || 1}
                  placeholder="Filter by page"
                  value={selectedPage ?? ""}
                  onChange={(e) => {
                    const v = e.target.value
                    if (!v) { setSelectedPage(null); setSelectedPageSource("input"); return }
                    const n = parseInt(v, 10)
                    if (Number.isFinite(n)) { setSelectedPage(Math.min(Math.max(1, n), pageCount || 1)); setSelectedPageSource("input") }
                  }}
                  className="w-28"
                />
                <Button variant="ghost" onClick={() => { setSelectedPage(1); setSelectedPageSource("init") }}>Reset</Button>
              </div>
            </CardAction>
          </CardHeader>
          <CardContent>
            <div
              ref={containerRef}
              onScroll={onScroll}
              className="overflow-y-auto rounded h-[85vh]"
            >
              {loadedUntil === 0 ? (
                <div className="p-4">
                  <Skeleton className="h-64 w-full" />
                </div>
              ) : (
                <div className="flex flex-col">
                  {Array.from({ length: loadedUntil }, (_, i) => i + 1).map((p) => (
                    <div key={p} id={`page-${p}`} className="p-2">
                      {images[p] ? (
                        images[p].length ? (
                          <div className="flex w-full items-start justify-center">
                            <img
                              src={images[p]}
                              alt={`Page ${p}`}
                              className="block w-full h-auto object-contain rounded border cursor-zoom-in"
                              loading="lazy"
                              onLoad={onImageLoad}
                              onClick={() => { setPreviewPage(p); setPreviewZoom(1) }}
                            />
                          </div>
                        ) : (
                          <div className="flex h-64 w-full items-center justify-center rounded border bg-muted text-muted-foreground">
                            Image unavailable for page {p}
                          </div>
                        )
                      ) : (
                        <Skeleton className="h-64 w-full" />
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Right: tree */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ListTree className="size-5 text-muted-foreground" /> Concept Tree
            </CardTitle>
            <CardAction>
              <Tabs value={conceptTab} onValueChange={(v) => setConceptTab((v as "tree" | "graph") ?? "tree")}>
                <TabsList>
                  <TabsTrigger value="tree">Tree</TabsTrigger>
                  <TabsTrigger value="graph">Graph</TabsTrigger>
                </TabsList>
              </Tabs>
            </CardAction>
          </CardHeader>
          <CardContent>
            {conceptTab === "tree" ? (
              <div
                className="overflow-y-auto rounded border h-[85vh]"
              >
                {loadingTree ? (
                  <div className="flex flex-col gap-3 p-2">
                    {Array(3).fill(0).map((_, i) => (
                      <Skeleton key={i} className="h-6 w-full" />
                    ))}
                  </div>
                ) : selectedPage ? (
                  filteredTree ? (
                    <div className="p-2">
                      <TreeNodeView node={filteredTree} depth={0} />
                    </div>
                  ) : (
                    <p className="p-2 text-sm text-muted-foreground">No nodes for page {selectedPage}</p>
                  )
                ) : tree?.tree_data ? (
                  <div className="p-2">
                    <TreeNodeView node={tree.tree_data} depth={0} />
                  </div>
                ) : errorTree ? (
                  <p className="p-2 text-sm text-destructive">{errorTree}</p>
                ) : (
                  <p className="p-2 text-sm text-muted-foreground">No tree available</p>
                )}
              </div>
            ) : (
              <div
                className="rounded border w-full"
                style={{ height: containerMinHeight ? containerMinHeight : 600 }}
              >
                <ConceptGraph
                  root={selectedPage ? filteredTree ?? null : tree?.tree_data ?? null}
                  loading={loadingTree}
                  error={errorTree}
                />
              </div>
            )}
          </CardContent>
        </Card>

        {/* Full-screen page preview */}
        <Dialog open={previewPage !== null} onOpenChange={(o) => { if (!o) setPreviewPage(null) }}>
          <DialogContent className="w-[96vw] max-w-[96vw] h-[95vh] p-0 overflow-hidden flex flex-col">
            <DialogHeader className="relative border-b p-3">
              <DialogTitle className="w-full text-center">Page {previewPage ?? ""}</DialogTitle>
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
                  if (!previewPage) return
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
                  // Zoom with mouse wheel. Scroll up to zoom in, down to zoom out.
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
                {previewPage && images[previewPage] ? (
                  (() => {
                    const computed = previewFitWidth > 0
                    const widthPx = computed ? Math.round(previewFitWidth * previewZoom) : undefined
                    return (
                      <img
                        ref={previewImgRef}
                        src={images[previewPage]}
                        alt={`Page ${previewPage}`}
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
      </div>
    </div>
  )
}

function TreeNodeView({ node, depth }: { node: TreeNode; depth: number }) {
  const hasChildren = Array.isArray(node.children) && node.children.length > 0
  const typeLabel = node.node_type === "L1"
    ? "Hierarchy"
    : node.node_type === "L2"
      ? "Topic"
      : node.node_type === "L3"
        ? "Detail"
        : node.node_type
  return (
    <div className="mb-3">
      <div className="flex items-start gap-2">
        <div className="mt-1 h-[10px] w-[10px] shrink-0 rounded-full bg-muted" />
        <div className="flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-medium">{node.title ?? "Untitled"}</span>
            {typeLabel ? <Badge>{typeLabel}</Badge> : null}
            {Array.isArray(node.pages) && node.pages.length > 0 ? (
              <Badge>p. {node.pages.join(", ")}</Badge>
            ) : null}
          </div>
          {node.summary ? (
            <p className="text-sm text-muted-foreground">{node.summary}</p>
          ) : null}
        </div>
      </div>
      {hasChildren ? (
        <div className="ml-6 mt-2 border-l pl-4">
          {node.children!.map((child, i) => (
            <TreeNodeView key={i} node={child} depth={depth + 1} />
          ))}
        </div>
      ) : null}
    </div>
  )
}

// Graph view using React Flow
function ConceptGraph({ root, loading, error }: { root: TreeNode | null; loading: boolean; error: string | null }) {
  const xGap = 280
  const yGap = 130

  function flattenToGraph(rootNode: TreeNode): { nodes: Node[]; edges: Edge[] } {
    const nodes: Node[] = []
    const edges: Edge[] = []
    const depthCounts = new Map<number, number>()

    let idCounter = 0
    function nextId() { idCounter++; return `n-${idCounter}` }

    function visit(node: TreeNode, depth: number, parentId?: string): string {
      const rowIndex = depthCounts.get(depth) ?? 0
      depthCounts.set(depth, rowIndex + 1)
      const id = nextId()
      const typeLabel = node.node_type === "L1" ? "Hierarchy" : node.node_type === "L2" ? "Topic" : node.node_type === "L3" ? "Detail" : node.node_type ?? "Node"
      const label = node.title ?? "Untitled"

      nodes.push({
        id,
        type: "concept",
        position: { x: depth * xGap, y: rowIndex * yGap },
        data: {
          label,
          typeLabel,
          summary: node.summary ?? "",
          pages: Array.isArray(node.pages) ? node.pages : [],
        },
      } as Node)

      if (parentId) {
        edges.push({ id: `e-${parentId}-${id}`, source: parentId, target: id, animated: true })
      }

      const children = Array.isArray(node.children) ? node.children : []
      for (const child of children) {
        visit(child, depth + 1, id)
      }
      return id
    }

    visit(rootNode, 0)
    return { nodes, edges }
  }

  const graph = React.useMemo(() => {
    if (!root) return { nodes: [], edges: [] }
    return flattenToGraph(root)
  }, [root])

  const nodeTypes = React.useMemo(() => ({ concept: ConceptNode }), [])


  if (loading) {
    return (
      <div className="flex flex-col gap-3 p-2">
        {Array(3).fill(0).map((_, i) => (
          <Skeleton key={i} className="h-6 w-full" />
        ))}
      </div>
    )
  }
  if (error) return <p className="p-2 text-sm text-destructive">{error}</p>
  if (!root) return <p className="p-2 text-sm text-muted-foreground">No graph available</p>

  // Helper component to call fitView after ReactFlow mounts (inside provider)
  function FitViewOnInit() {
    const rf = useReactFlow()
    React.useEffect(() => {
      try { rf.fitView({ padding: 0.2 }) } catch { }
    }, [rf])
    return null
  }

  if (graph.nodes.length === 0) {
    return <p className="p-2 text-sm text-muted-foreground">No nodes to render</p>
  }

  return (
    <ReactFlow
      nodes={graph.nodes}
      edges={graph.edges}
      fitView
      nodeTypes={nodeTypes}
      minZoom={0.2}
      proOptions={{ hideAttribution: true }}
      style={{ width: "100%", height: "100%" }}
    >
      <FitViewOnInit />
      <Background />
      <Controls />
    </ReactFlow>
  )
}

function ConceptNode({ data }: { data: { label: string; typeLabel: string; summary?: string; pages?: number[] } }) {
  const title = data.label ?? "Untitled"
  const displayLabel = title.length > 28 ? `${title.slice(0, 28)}…` : title
  return (
    <div className="rounded-md border bg-background shadow-xs">
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="px-3 py-2">
            <div className="text-sm font-medium" title={title}>{displayLabel}</div>
            <div className="mt-0.5 text-xs text-muted-foreground">{data.typeLabel}</div>
          </div>
        </TooltipTrigger>
        <TooltipContent sideOffset={6}>
          <div className="max-w-xs">
            <p className="text-xs font-medium">{title}</p>
            {data.summary ? <p className="text-xs leading-snug">{data.summary}</p> : <p className="text-xs text-muted-foreground">No summary</p>}
            {Array.isArray(data.pages) && data.pages.length ? (
              <p className="mt-1 text-xs text-muted-foreground">p. {data.pages.join(", ")}</p>
            ) : null}
          </div>
        </TooltipContent>
      </Tooltip>
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
    </div>
  )
}