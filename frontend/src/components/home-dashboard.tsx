"use client"

import * as React from "react"
import Link from "next/link"
import { FileText, Folder, MessageSquare, CircleCheckBig, Clock, Upload, XCircle } from "lucide-react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Button } from "@/components/ui/button"
import { Empty, EmptyHeader, EmptyTitle, EmptyDescription, EmptyMedia, EmptyContent } from "@/components/ui/empty"
import { Skeleton } from "@/components/ui/skeleton"
import { getBackendUrl } from "@/lib/env"

export default function HomeDashboard() {

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

            {/* Recent sections (empty state until backend is ready) */}
            <div className="grid gap-4 sm:grid-cols-2">
                <RecentDocumentsCard />
                <RecentEmptyCard
                    title="Recent Collections"
                    description="Your most recently created or updated collections"
                    ctaLabel="Create Collection"
                    href="/collection"
                    icon={<Folder className="size-6" />}
                    emptyTitle="No collections"
                    emptyDescription="Get started by creating your first collection."
                />
                <RecentEmptyCard
                    title="Recent Conversations"
                    description="Your most recent chat sessions"
                    ctaLabel="Start Chat"
                    href="/query"
                    icon={<MessageSquare className="size-6" />}
                    emptyTitle="No conversations"
                    emptyDescription="Get started by starting your first chat."
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

function RecentEmptyCard({
    title,
    description,
    href,
    ctaLabel,
    icon,
    emptyTitle,
    emptyDescription,
}: {
    title: string
    description: string
    href: string
    ctaLabel: string
    icon: React.ReactNode
    emptyTitle: string
    emptyDescription: string
}) {
    return (
        <Card>
            <CardHeader>
                <CardTitle>{title}</CardTitle>
                <CardDescription>{description}</CardDescription>
            </CardHeader>
            <CardContent>
                <Empty>
                    <EmptyHeader>
                        <EmptyMedia variant="icon">{icon}</EmptyMedia>
                        <EmptyTitle>{emptyTitle}</EmptyTitle>
                        <EmptyDescription>{emptyDescription}</EmptyDescription>
                    </EmptyHeader>
                    <EmptyContent>
                        <Link href={href} prefetch>
                            <Button>
                                {ctaLabel}
                            </Button>
                        </Link>
                    </EmptyContent>
                </Empty>
            </CardContent>
        </Card>
    )
}

function RecentDocumentsCard() {
    const [loading, setLoading] = React.useState(true)
    const [items, setItems] = React.useState<any[]>([])
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
                            <Button key={doc.id} variant="outline" className="flex w-full items-center justify-between gap-3 py-6">
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
    // Normalize fractional seconds to at most 3 digits for Date parsing
    const normalized = value.replace(/(\.\d{3})\d+$/, "$1").replace(/\.\d+$/, (m) => (m.length > 4 ? m.slice(0, 4) : m))
    const tryDate = (s: string) => {
        const d = new Date(s)
        return isNaN(d.getTime()) ? undefined : d
    }
    const d = tryDate(normalized) ?? tryDate(value.replace(/\.\d+$/, ""))
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