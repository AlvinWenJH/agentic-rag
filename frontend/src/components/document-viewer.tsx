"use client"

import * as React from "react"
import { useEffect, useMemo, useRef, useState } from "react"
import { useParams, useSearchParams } from "next/navigation"
import { FileText, HardDrive, CircleCheckBig, Download, GitBranch, LibraryBig } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardAction } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { getBackendUrl } from "@/lib/env"

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
  const d = new Date(dt)
  if (isNaN(d.getTime())) return dt
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

  const pageCount = useMemo(() => resolvePageCount(doc), [doc])

  // Images pagination state
  const [loadedUntil, setLoadedUntil] = useState<number>(0)
  const [images, setImages] = useState<Record<number, string>>({})
  const containerRef = useRef<HTMLDivElement>(null)
  const batchSize = 10
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

  function candidates(page: number) {
    const base = `${backendUrl}/api/v1/documents/${resolvedId}`
    return [
      `${base}/pages/${page}/image`,
      `${base}/image?page=${page}`,
      `${base}/page/${page}/image`,
    ]
  }

  async function loadPageImage(page: number) {
    for (const url of candidates(page)) {
      try {
        const res = await fetch(url, { headers: { accept: "image/png,image/jpeg" } })
        if (!res.ok) continue
        const blob = await res.blob()
        const objUrl = URL.createObjectURL(blob)
        setImages((prev) => ({ ...prev, [page]: objUrl }))
        return
      } catch (_) {
        // try next
      }
    }
    // if all failed, mark placeholder
    setImages((prev) => ({ ...prev, [page]: "" }))
  }

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
              <CircleCheckBig className="size-5 text-muted-foreground" /> Pages ({formatNumber(pageCount)})
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
              className="max-h-[70vh] overflow-y-auto rounded border"
            >
              {loadedUntil === 0 && pageCount > 0 ? (
                <div className="p-4">
                  <Skeleton className="h-64 w-full" />
                </div>
              ) : (
                <div className="flex flex-col">
                  {Array.from({ length: loadedUntil }, (_, i) => i + 1).map((p) => (
                    <div key={p} id={`page-${p}`} className="p-2">
                      {images[p] ? (
                        images[p].length ? (
                          <img
                            src={images[p]}
                            alt={`Page ${p}`}
                            className="w-full rounded border"
                            loading="lazy"
                          />
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
            <CardTitle>Document Tree</CardTitle>
          </CardHeader>
          <CardContent>
            {loadingTree ? (
              <div className="flex flex-col gap-3">
                {Array(3).fill(0).map((_, i) => (
                  <Skeleton key={i} className="h-6 w-full" />
                ))}
              </div>
            ) : selectedPage ? (
              filteredTree ? (
                <TreeNodeView node={filteredTree} depth={0} />
              ) : (
                <p className="text-sm text-muted-foreground">No nodes for page {selectedPage}</p>
              )
            ) : tree?.tree_data ? (
              <TreeNodeView node={tree.tree_data} depth={0} />
            ) : errorTree ? (
              <p className="text-sm text-destructive">{errorTree}</p>
            ) : (
              <p className="text-sm text-muted-foreground">No tree available</p>
            )}
          </CardContent>
        </Card>
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