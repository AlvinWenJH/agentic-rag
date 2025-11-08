"use client"

import * as React from "react"
import Link from "next/link"
import { FileText, Folder, MessageSquare } from "lucide-react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { getBackendUrl } from "@/lib/env"

type Metric = { label: string; value: number | string }

function toNumberOrZero(val: unknown): number {
    const num = typeof val === "number" ? val : Number(val)
    return Number.isFinite(num) ? num : 0
}

function bytesToMB(bytes: unknown, digits = 2): string {
    const num = toNumberOrZero(bytes)
    const mb = num / (1024 * 1024)
    return mb.toFixed(digits)
}

function extractDocumentMetrics(data: any): Metric[] {
    // Matches the provided /documents/stats response shape
    const totalDocs = toNumberOrZero(data?.total_documents)
    const completed = toNumberOrZero(data?.documents_by_status?.completed)
    const storageMB = bytesToMB(data?.total_file_size)
    return [
        { label: "Total Documents", value: totalDocs },
        { label: "Completed", value: completed },
        { label: "Storage (MB)", value: storageMB },
    ]
}

function extractCollectionMetrics(data: any): Metric[] {
    return [
        { label: "Collections", value: toNumberOrZero(data?.collections ?? data?.count ?? data?.total) },
        { label: "Documents Linked", value: toNumberOrZero(data?.documents_linked ?? data?.documents ?? data?.docs ?? data?.items) },
        { label: "Concepts", value: toNumberOrZero(data?.concepts ?? data?.tags ?? data?.nodes) },
    ]
}

function extractChatMetrics(data: any): Metric[] {
    return [
        { label: "Conversations", value: toNumberOrZero(data?.conversations ?? data?.count ?? data?.total) },
        { label: "Messages", value: toNumberOrZero(data?.messages ?? data?.msgs) },
        { label: "Avg Response (ms)", value: toNumberOrZero(data?.avg_response_time_ms ?? data?.avg_ms) },
    ]
}

export default function HomeDashboard() {
    const [loading, setLoading] = React.useState({ documents: true, collections: true, chat: true })
    const [docsStats, setDocsStats] = React.useState<any>({})
    const [collectionsStats, setCollectionsStats] = React.useState<any>({})
    const [chatStats, setChatStats] = React.useState<any>({})

    React.useEffect(() => {
        const backendUrl = getBackendUrl()

        async function fetchStats(kind: "documents" | "collections" | "chat") {
            try {
                const res = await fetch(`${backendUrl}/api/v1/${kind}/stats`, {
                    headers: { accept: "application/json" },
                })
                if (!res.ok) throw new Error(String(res.status))
                const data = await res.json().catch(() => ({}))
                if (kind === "documents") setDocsStats(data)
                if (kind === "collections") setCollectionsStats(data)
                if (kind === "chat") setChatStats(data)
            } catch {
                // Fallback to zeros by leaving stats as empty objects
            } finally {
                setLoading((prev) => ({ ...prev, [kind]: false }))
            }
        }

        fetchStats("documents")
        fetchStats("collections")
        fetchStats("chat")
    }, [])

    const docMetrics = extractDocumentMetrics(docsStats)
    const colMetrics = extractCollectionMetrics(collectionsStats)
    const chatMetrics = extractChatMetrics(chatStats)

    return (
        <div className="flex flex-1 flex-col gap-6">
            {/* Top descriptive cards */}
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                <TopCard
                    href="/documents"
                    title="Documents"
                    icon={FileText}
                    description="Analyze key topics to form concepts tree and visual elements"
                />
                <TopCard
                    href="/collection"
                    title="Collections"
                    icon={Folder}
                    description="Organize documents into sets for retrieval source"
                />
                <TopCard
                    href="/query"
                    title="Chat"
                    icon={MessageSquare}
                    description="Ask questions about your documents in agentic pattern"
                />
            </div>

            <Separator className="my-2" />

            {/* Metrics overview cards */}
            <div className="grid gap-4 sm:grid-cols-2">
                <MetricCard
                    title="Documents Overview"
                    description="Stay on top of your uploaded files."
                    loading={loading.documents}
                    metrics={docMetrics}
                />
                <MetricCard
                    title="Collections Overview"
                    description="Structure documents into reusable sets."
                    loading={loading.collections}
                    metrics={colMetrics}
                />
                <MetricCard
                    title="Chat Overview"
                    description="Recent activity and responsiveness."
                    loading={loading.chat}
                    metrics={chatMetrics}
                />
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

function MetricCard({
    title,
    description,
    loading,
    metrics,
}: {
    title: string
    description: string
    loading: boolean
    metrics: Metric[]
}) {
    return (
        <Card>
            <CardHeader>
                <CardTitle>{title}</CardTitle>
                <CardDescription>{description}</CardDescription>
            </CardHeader>
            <CardContent>
                {loading ? (
                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                        {(metrics.length ? Array(metrics.length).fill(0) : Array(3).fill(0)).map((_, i) => (
                            <Skeleton key={i} className="h-16 w-full" />
                        ))}
                    </div>
                ) : (
                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                        {metrics.map((m) => (
                            <StatTile key={m.label} label={m.label} value={m.value} />
                        ))}
                    </div>
                )}
            </CardContent>
        </Card>
    )
}

function StatTile({ label, value }: { label: string; value: number | string }) {
    return (
        <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-sm text-muted-foreground">{label}</div>
            <div className="mt-1 text-3xl font-semibold tabular-nums tracking-tight">{value}</div>
        </div>
    )
}