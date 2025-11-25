"use client"

import * as React from "react"
import Link from "next/link"
import { FileText, MessageSquare, CircleCheckBig, Clock, Upload, XCircle, CheckCircle, AlertTriangle, ClipboardList } from "lucide-react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Empty, EmptyHeader, EmptyTitle, EmptyDescription, EmptyMedia, EmptyContent } from "@/components/ui/empty"
import { Skeleton } from "@/components/ui/skeleton"
import { getBackendUrl } from "@/lib/env"
import { parseUtcDate } from "@/lib/utils"

export default function HomeDashboard() {
    return (
        <div className="flex flex-1 flex-col gap-6">
            {/* Top descriptive cards */}
            <div className="grid gap-4 sm:grid-cols-3">
                <TopCard
                    href="/documents"
                    title="Documents"
                    icon={FileText}
                    description="Upload and analyze documents to extract key topics"
                />
                <TopCard
                    href="/query"
                    title="Chat"
                    icon={MessageSquare}
                    description="Ask questions about your documents in agentic pattern"
                />
                <TopCard
                    href="/analysis"
                    title="Analysis"
                    icon={ClipboardList}
                    description="View and manage your analysis"
                />
            </div>

            {/* Recent sections */}
            <div className="grid gap-4 sm:grid-cols-2">
                <RecentDocumentsCard />
                <RecentConversationsCard />
            </div>
            <div className="grid gap-4 sm:grid-cols-1">
                <RecentAnalysisCard />
            </div>
        </div>
    )
}

function TopCard({
    href,
    title,
    description,
    icon: Icon,
}: {
    href: string
    title: string
    description: string
    icon: React.ComponentType<{ className?: string }>
}) {
    return (
        <Link href={href} className="group" prefetch>
            <Card className="h-full transition-colors hover:border-primary/40">
                <CardHeader className="space-y-2">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="rounded-md bg-muted p-2 text-muted-foreground">
                                <Icon className="size-5" />
                            </div>
                            <CardTitle>{title}</CardTitle>
                        </div>
                        <span className="text-sm text-muted-foreground">Explore →</span>
                    </div>
                    <CardDescription>{description}</CardDescription>
                </CardHeader>
            </Card>
        </Link>
    )
}

function RecentDocumentsCard() {
  const [loading, setLoading] = React.useState(true)
  const [items, setItems] = React.useState<Array<{ id: string; filename?: string; title?: string; status?: string; updated_at?: string }>>([])
    React.useEffect(() => {
        const backendUrl = getBackendUrl()
        async function fetchDocs() {
            try {
                const res = await fetch(`${backendUrl}/api/v1/documents/?skip=0&limit=4`, {
                    headers: { accept: "application/json" },
                })
                if (!res.ok) throw new Error(String(res.status))
                const data = await res.json().catch(() => ({}))
                setItems(Array.isArray(data?.documents) ? data.documents : [])
            } catch (_) {
                setItems([])
            } finally {
                setLoading(false)
            }
        }
        fetchDocs()
    }, [])

    const hasItems = items.length > 0

    return (
        <Card>
            <CardHeader>
                <CardTitle>Recent Documents</CardTitle>
                <CardDescription>Your most recently uploaded or updated documents</CardDescription>
            </CardHeader>
            <CardContent>
                {loading ? (
                    <div className="flex flex-col gap-4">
                        {Array(3).fill(0).map((_, i) => (
                            <Skeleton key={i} className="h-6 w-full" />
                        ))}
                    </div>
                ) : hasItems ? (
                    <div className="flex flex-col gap-4">
                        {items.map((doc) => (
                            <Link key={doc.id} href={`/documents/${encodeURIComponent(doc.id)}`} prefetch>
                                <Button variant="outline" className="flex w-full items-center justify-between gap-3 py-6">
                                    <span className="flex min-w-0 items-center gap-3">
                                        <FileText className="size-5 shrink-0 text-muted-foreground" />
                                        <span className="min-w-0 flex flex-col items-start text-left">
                                            <span className="truncate">{doc?.filename ?? doc?.title ?? doc?.id}</span>
                                            <span className="mt-1 text-xs leading-snug text-muted-foreground">{formatTimeAgo(doc?.updated_at)}</span>
                                        </span>
                                    </span>
                                    <span className="ml-auto flex items-center gap-1 rounded bg-muted px-2 py-0.5 text-xs capitalize">
                                        <StatusIcon status={doc?.status} />
                                        {doc?.status ?? "unknown"}
                                    </span>
                                </Button>
                            </Link>
                        ))}
                    </div>
                ) : (
                    <Empty>
                        <EmptyHeader>
                            <EmptyMedia variant="icon"><FileText className="size-6" /></EmptyMedia>
                            <EmptyTitle>No documents</EmptyTitle>
                            <EmptyDescription>Get started by uploading your first document.</EmptyDescription>
                        </EmptyHeader>
                        <EmptyContent>
                            <Link href="/documents" prefetch>
                                <Button>Upload Document</Button>
                            </Link>
                        </EmptyContent>
                    </Empty>
                )}
            </CardContent>
        </Card>
    )
}

function RecentConversationsCard() {
  const [loading, setLoading] = React.useState(true)
  const [items, setItems] = React.useState<Array<{ id: string; document_id: string; title?: string; updated_at?: string }>>([])
    React.useEffect(() => {
        const backendUrl = getBackendUrl()
        async function fetchConversations() {
            try {
                const res = await fetch(`${backendUrl}/api/v1/conversations/recent?limit=4`, {
                    headers: { accept: "application/json" },
                })
                if (!res.ok) throw new Error(String(res.status))
                const data = await res.json().catch(() => ({}))
                setItems(Array.isArray(data?.conversations) ? data.conversations : [])
            } catch (_) {
                setItems([])
            } finally {
                setLoading(false)
            }
        }
        fetchConversations()
    }, [])

    const hasItems = items.length > 0

    return (
        <Card>
            <CardHeader>
                <CardTitle>Recent Conversations</CardTitle>
                <CardDescription>Your most recent chat sessions</CardDescription>
            </CardHeader>
            <CardContent>
                {loading ? (
                    <div className="flex flex-col gap-4">
                        {Array(3).fill(0).map((_, i) => (
                            <Skeleton key={i} className="h-6 w-full" />
                        ))}
                    </div>
                ) : hasItems ? (
                    <div className="flex flex-col gap-4">
                        {items.map((conv) => (
                            <Link key={conv.id} href={`/query?documentId=${conv.document_id}&conversationId=${conv.id}`} prefetch>
                                <Button variant="outline" className="flex w-full items-center justify-between gap-3 py-6">
                                    <span className="flex min-w-0 items-center gap-3">
                                        <MessageSquare className="size-5 shrink-0 text-muted-foreground" />
                                        <span className="min-w-0 flex flex-col items-start text-left">
                                            <span className="truncate">{conv?.title ?? "Untitled Conversation"}</span>
                                            <span className="mt-1 text-xs leading-snug text-muted-foreground">{formatTimeAgo(conv?.updated_at)}</span>
                                        </span>
                                    </span>
                                </Button>
                            </Link>
                        ))}
                    </div>
                ) : (
                    <Empty>
                        <EmptyHeader>
                            <EmptyMedia variant="icon"><MessageSquare className="size-6" /></EmptyMedia>
                            <EmptyTitle>No conversations</EmptyTitle>
                            <EmptyDescription>Get started by starting your first chat.</EmptyDescription>
                        </EmptyHeader>
                        <EmptyContent>
                            <Link href="/query" prefetch>
                                <Button>Start Chat</Button>
                            </Link>
                        </EmptyContent>
                    </Empty>
                )}
            </CardContent>
        </Card>
    )
}

function RecentAnalysisCard() {
  const [loading, setLoading] = React.useState(true)
  const [items, setItems] = React.useState<Array<{ id: string; analysis_id: string; document_id: string; document_title?: string; analysis_title?: string; completed_items?: number; total_items?: number; updated_at?: string; status?: string }>>([])
    React.useEffect(() => {
        const backendUrl = getBackendUrl()
        async function fetchAnalysisResults() {
            try {
                const res = await fetch(`${backendUrl}/api/v1/analysis/recent?limit=4`, {
                    headers: { accept: "application/json" },
                })
                if (!res.ok) throw new Error(String(res.status))
                const data = await res.json().catch(() => ({}))
                setItems(Array.isArray(data?.results) ? data.results : [])
            } catch (_) {
                setItems([])
            } finally {
                setLoading(false)
            }
        }
        fetchAnalysisResults()
    }, [])

    const hasItems = items.length > 0

    return (
        <Card className="min-w-0">
            <CardHeader>
                <CardTitle>Recent Analysis Results</CardTitle>
                <CardDescription>Your most recently run analysis</CardDescription>
            </CardHeader>
            <CardContent>
                {loading ? (
                    <div className="flex flex-col gap-4">
                        {Array(4).fill(0).map((_, i) => (
                            <Skeleton key={i} className="h-20 w-full" />
                        ))}
                    </div>
                ) : hasItems ? (
                    <div className="flex flex-col gap-4">
                        {items.map((result) => (
                            <Link
                                key={result.id}
                                href={`/analysis/${encodeURIComponent(result.analysis_id)}/result/${encodeURIComponent(result.document_id)}`}
                                prefetch
                                className="block overflow-hidden min-w-0"
                            >
                                <Button variant="outline" className="flex h-auto w-full max-w-full min-w-0 flex-col items-start gap-3 p-4 text-left md:flex-row md:items-center md:justify-between md:gap-4">
                                    <div className="flex min-w-0 flex-1 items-center gap-3 w-full">
                                        <FileText className="size-5 shrink-0 text-muted-foreground" />
                                        <div className="flex min-w-0 flex-1 flex-col gap-1">
                                            <span className="truncate text-sm font-medium">{result?.document_title ?? "Unknown Document"}</span>
                                            <span className="truncate text-xs text-muted-foreground">{result?.analysis_title ?? "Analysis"}</span>
                                        </div>
                                    </div>
                                    <div className="flex w-full flex-wrap items-center gap-2 md:w-auto md:shrink-0 md:gap-4">
                                        {result?.completed_items != null && result?.total_items != null ? (
                                            <span className="text-xs text-muted-foreground">
                                                {result.completed_items}/{result.total_items} items
                                            </span>
                                        ) : null}
                                        <span className="text-xs text-muted-foreground">{formatTimeAgo(result?.updated_at)}</span>
                                        <AnalysisStatusBadge status={result?.status} />
                                    </div>
                                </Button>
                            </Link>
                        ))}
                    </div>
                ) : (
                    <Empty>
                        <EmptyHeader>
                            <EmptyMedia variant="icon"><CheckCircle className="size-6" /></EmptyMedia>
                            <EmptyTitle>No analysis results</EmptyTitle>
                            <EmptyDescription>Run analysis on your documents to see results here.</EmptyDescription>
                        </EmptyHeader>
                        <EmptyContent>
                            <Link href="/analysis" prefetch>
                                <Button>Go to Analysis</Button>
                            </Link>
                        </EmptyContent>
                    </Empty>
                )}
            </CardContent>
        </Card>
    )
}

function AnalysisStatusBadge({ status }: { status?: string }) {
    const s = (status ?? "").toLowerCase()
    let color = "bg-muted text-muted-foreground"
    let Icon: React.ComponentType<React.ComponentProps<"svg">> = Clock

    switch (s) {
        case "completed":
            Icon = CheckCircle
            color = "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
            break
        case "processing":
        case "in_progress":
        case "pending":
            Icon = Clock
            color = "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400"
            break
        case "failed":
        case "error":
            Icon = AlertTriangle
            color = "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"
            break
    }
    return (
        <span className={`inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs capitalize ${color}`}>
            <Icon className="size-3" />
            {status ?? "unknown"}
        </span>
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
        case "processing":
        case "in_progress":
            Icon = Clock
            color = "text-amber-600"
            break
        case "uploaded":
            Icon = Upload
            color = "text-blue-600"
            break
        case "failed":
        case "error":
            Icon = XCircle
            color = "text-red-600"
            break
    }
    return <Icon className={`size-4 ${color}`} />
}

function formatTimeAgo(value?: string) {
    if (!value) return "—"
    const d = parseUtcDate(value)
    if (!d) return value

    const now = new Date()
    const diffMs = now.getTime() - d.getTime()
    const absMs = Math.abs(diffMs)

    const minutes = Math.floor(absMs / (60 * 1000))
    const hours = Math.floor(absMs / (60 * 60 * 1000))
    const days = Math.floor(absMs / (24 * 60 * 60 * 1000))
    const months = Math.floor(absMs / (30 * 24 * 60 * 60 * 1000))
    const years = Math.floor(absMs / (365 * 24 * 60 * 60 * 1000))

    const suffix = diffMs >= 0 ? "ago" : "from now"
    if (minutes < 1) return `just now`
    if (minutes < 60) return `${minutes} ${minutes === 1 ? "min" : "mins"} ${suffix}`
    if (hours < 24) return `${hours} ${hours === 1 ? "hour" : "hours"} ${suffix}`
    if (days < 30) return `${days} ${days === 1 ? "day" : "days"} ${suffix}`
    if (months < 12) return `${months} ${months === 1 ? "month" : "months"} ${suffix}`
    return `${years} ${years === 1 ? "year" : "years"} ${suffix}`
}