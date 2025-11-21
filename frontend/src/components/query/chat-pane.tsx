"use client"

import * as React from "react"
import { useSearchParams, useRouter } from "next/navigation"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { getBackendUrl } from "@/lib/env"
import { Settings, Hash, BarChart3, Cpu, Database, Send as SendIcon, Bot, FileText, ZoomOut, ZoomIn, RefreshCcw, MessageSquare, Plus, Trash2 } from "lucide-react"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { Spinner } from "@/components/ui/spinner"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Skeleton } from "@/components/ui/skeleton"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"

type DocumentItem = {
  id: string
  title?: string
  filename?: string
  description?: string
  status?: string
}

type ChatMessage = {
  role: "user" | "assistant" | "system"
  content: string
  meta?: {
    usage?: any
    references?: any
  }
}

type Conversation = {
  id: string
  document_id: string
  user_id?: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

export default function ChatPane() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const backendUrl = getBackendUrl()
  const [mode, setMode] = React.useState<"document">("document")
  const [docs, setDocs] = React.useState<DocumentItem[]>([])
  const [loadingDocs, setLoadingDocs] = React.useState(true)
  const [selected, setSelected] = React.useState<DocumentItem | null>(null)
  const [selectedDetail, setSelectedDetail] = React.useState<{ id: string; title?: string; description?: string; file_size?: number } | null>(null)
  const [docQuery, setDocQuery] = React.useState("")
  const [messages, setMessages] = React.useState<ChatMessage[]>([])
  const [input, setInput] = React.useState("")
  const [isStreaming, setIsStreaming] = React.useState(false)
  const scrollRef = React.useRef<HTMLDivElement | null>(null)
  const controllerRef = React.useRef<AbortController | null>(null)
  const assistantIndexRef = React.useRef<number | null>(null)
  const [openRefsIndex, setOpenRefsIndex] = React.useState<number | null>(null)
  const [refsTab, setRefsTab] = React.useState<"query" | "pages">("query")
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
  
  // Conversation history state
  const [conversations, setConversations] = React.useState<Conversation[]>([])
  const [loadingConversations, setLoadingConversations] = React.useState(false)
  const [activeConversation, setActiveConversation] = React.useState<Conversation | null>(null)

  React.useEffect(() => {
    let mounted = true
    async function fetchDocs() {
      try {
        const res = await fetch(`${backendUrl}/api/v1/documents/?skip=0&limit=20`, {
          headers: { accept: "application/json" },
        })
        const json = await res.json().catch(() => ({}))
        if (!mounted) return
        const list = Array.isArray(json?.documents) ? json.documents : []
        setDocs(list)
        setLoadingDocs(false)
      } catch {
        if (!mounted) return
        setDocs([])
        setLoadingDocs(false)
      }
    }
    fetchDocs()
    return () => {
      mounted = false
    }
  }, [backendUrl])

  const loadSelectedDetail = React.useCallback(async () => {
    if (!selected?.id) { setSelectedDetail(null); return }
    try {
      const res = await fetch(`${backendUrl}/api/v1/documents/${encodeURIComponent(selected.id)}`, {
        headers: { accept: "application/json" },
      })
      const data = await res.json().catch(() => ({}))
      setSelectedDetail({ id: selected.id, title: data?.title ?? data?.filename ?? selected.title ?? selected.filename ?? selected.id, description: data?.description, file_size: Number(data?.file_size ?? data?.metadata?.file_size ?? 0) })
    } catch {
      setSelectedDetail({ id: selected.id, title: selected.title ?? selected.filename ?? selected.id, description: undefined, file_size: undefined })
    }
  }, [backendUrl, selected?.id])

  React.useEffect(() => {
    setSelectedDetail(null)
    if (selected?.id) loadSelectedDetail()
    // Clear active conversation and messages when switching documents
    // Only clear if the active conversation doesn't belong to the new document
    if (activeConversation?.document_id !== selected?.id) {
      setActiveConversation(null)
      setMessages([])
    }
  }, [loadSelectedDetail, selected?.id])

  // Handle URL parameters for deep linking
  React.useEffect(() => {
    const docId = searchParams.get("documentId")
    const convId = searchParams.get("conversationId")

    if (docId && (!selected || selected.id !== docId)) {
      // Find the document in the list or create a temporary one
      const doc = docs.find(d => d.id === docId)
      if (doc) {
        setSelected(doc)
      } else {
        // If not in the initial list, we might need to fetch it or set a placeholder
        // For now, let's set a placeholder and let loadSelectedDetail fetch details
        setSelected({ id: docId })
      }
    }

    if (convId && docId) {
      // If we have a conversation ID, try to load it
      // We need to wait for conversations to load first, or fetch it directly
      const fetchConversation = async () => {
        try {
          const res = await fetch(`${backendUrl}/api/v1/conversations/${convId}`, {
            headers: { accept: "application/json" },
          })
          const data = await res.json().catch(() => null)
          if (data) {
             const convo: Conversation = {
              id: data.id || data._id,
              document_id: data.document_id,
              title: data.title,
              created_at: data.created_at,
              updated_at: data.updated_at,
              message_count: data.messages?.length || 0
             }
             setActiveConversation(convo)
             // Load messages
             if (data.messages) {
                const loadedMessages: ChatMessage[] = data.messages.map((msg: any) => ({
                  role: msg.role,
                  content: msg.content,
                  meta: msg.meta,
                }))
                setMessages(loadedMessages)
             }
          }
        } catch (err) {
          console.error("Failed to load conversation from URL", err)
        }
      }
      fetchConversation()
    }
  }, [searchParams, backendUrl, docs])

  React.useEffect(() => {
    if (!scrollRef.current) return
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [messages])

  function labelFor(doc: DocumentItem): string {
    return doc.title || doc.filename || doc.id
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

  React.useEffect(() => {
    if (openRefsIndex == null) {
      setRefsQueryTree(null)
      setRefsPagesImages({})
      setRefsQueryLoading(false)
      setRefsPagesLoading(false)
      setPreviewPageSrc(null)
      return
    }
    const refs = messages[openRefsIndex]?.meta?.references || {}
    const qps: Array<{ document_id: string; path: string; depth?: number }> = Array.isArray(refs?.query_paths) ? refs.query_paths : []
    const rps: Array<{ document_id: string; page: number }> = Array.isArray(refs?.retrieved_pages) ? refs.retrieved_pages : []
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
            const entries = await Promise.all(rps.map(async (p) => {
              const u = `${backendUrl}/api/v1/documents/${p.document_id}/page/${p.page}`
              const res = await fetch(u, { headers: { accept: "application/json" } })
              const json = await res.json().catch(() => null)
              const b64: string | undefined = json?.page_base64
              const src = b64 ? (b64.startsWith("data:image") ? b64 : `data:image/png;base64,${b64}`) : ""
              return [`${p.document_id}:${p.page}`, src] as const
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
  }, [openRefsIndex, backendUrl, messages])

  // Load conversations when document is selected
  React.useEffect(() => {
    if (!selected?.id) {
      setConversations([])
      setActiveConversation(null)
      return
    }
    loadConversations()
  }, [selected?.id, backendUrl])

  async function loadConversations() {
    if (!selected?.id) return
    setLoadingConversations(true)
    try {
      const res = await fetch(`${backendUrl}/api/v1/conversations/?limit=50&skip=0`, {
        headers: { accept: "application/json" },
      })
      const data = await res.json().catch(() => ({ conversations: [] }))
      const convos = Array.isArray(data.conversations) ? data.conversations.filter((c: Conversation) => c.document_id === selected.id) : []
      setConversations(convos)
    } catch {
      setConversations([])
    } finally {
      setLoadingConversations(false)
    }
  }

  async function createNewConversation() {
    if (!selected?.id) return
    try {
      const res = await fetch(`${backendUrl}/api/v1/conversations/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_id: selected.id }),
      })
      const data = await res.json().catch(() => null)
      if (data?.conversation_id) {
        const newConvo: Conversation = {
          id: data.conversation_id,
          document_id: selected.id,
          title: data.title || "New Conversation",
          created_at: data.created_at,
          updated_at: data.created_at,
          message_count: 0,
        }
        setActiveConversation(newConvo)
        setMessages([])
        await loadConversations()
        router.push(`/query?documentId=${selected.id}&conversationId=${newConvo.id}`)
      }
    } catch (err) {
      console.error("Failed to create conversation", err)
    }
  }

  async function loadConversationMessages(conversationId: string) {
    try {
      const res = await fetch(`${backendUrl}/api/v1/conversations/${conversationId}`, {
        headers: { accept: "application/json" },
      })
      const data = await res.json().catch(() => null)
      if (data?.messages) {
        const loadedMessages: ChatMessage[] = data.messages.map((msg: any) => ({
          role: msg.role,
          content: msg.content,
          meta: msg.meta,
        }))
        setMessages(loadedMessages)
      }
    } catch (err) {
      console.error("Failed to load conversation messages", err)
    }
  }

  async function selectConversation(convo: Conversation) {
    setActiveConversation(convo)
    await loadConversationMessages(convo.id)
    router.push(`/query?documentId=${convo.document_id}&conversationId=${convo.id}`)
  }

  async function deleteConversationById(conversationId: string) {
    try {
      await fetch(`${backendUrl}/api/v1/conversations/${conversationId}`, {
        method: "DELETE",
      })
      await loadConversations()
      if (activeConversation?.id === conversationId) {
        setActiveConversation(null)
        setMessages([])
        router.push(`/query?documentId=${selected?.id}`)
      }
    } catch (err) {
      console.error("Failed to delete conversation", err)
    }
  }

  async function handleSubmit(e?: React.FormEvent) {
    if (e) e.preventDefault()
    if (!selected || !input.trim() || isStreaming) return
    const text = input.trim()
    setInput("")
    setMessages((prev) => [...prev, { role: "user", content: text }])
    setIsStreaming(true)
    assistantIndexRef.current = null
    
    // Create conversation if none exists
    let convId = activeConversation?.id
    if (!convId) {
      try {
        const res = await fetch(`${backendUrl}/api/v1/conversations/`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ document_id: selected.id }),
        })
        const data = await res.json().catch(() => null)
        if (data?.conversation_id) {
          convId = data.conversation_id
          const newConvo: Conversation = {
            id: data.conversation_id,
            document_id: selected.id,
            title: data.title || "New Conversation",
            created_at: data.created_at,
            updated_at: data.created_at,
            message_count: 0,
          }
          setActiveConversation(newConvo)
          router.push(`/query?documentId=${selected.id}&conversationId=${newConvo.id}`)
        }
      } catch (err) {
        console.error("Failed to create conversation", err)
      }
    }
    
    const url = `${backendUrl}/api/v1/query/document/${selected.id}`
    const controller = new AbortController()
    controllerRef.current = controller

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: {
          accept: "text/event-stream",
          "content-type": "application/json",
        },
        body: JSON.stringify({ query: text, conversation_id: convId }),
        signal: controller.signal,
      })
      const reader = res.body?.getReader()
      const decoder = new TextDecoder()
      let buffer = ""

      while (reader) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split("\n\n")
        buffer = parts.pop() || ""
        for (const part of parts) {
          const line = part.split("\n").find((l) => l.startsWith("data:")) || ""
          const jsonStr = line.replace(/^data:\s*/, "")
          if (!jsonStr) continue
          let evt: any
          try {
            evt = JSON.parse(jsonStr)
          } catch {
            continue
          }
          if (evt?.type === "start") {
            continue
          }
          if (evt?.type === "text_delta") {
            const chunk = String(evt?.content || "")
            setMessages((prev) => {
              if (assistantIndexRef.current == null) {
                const nextIndex = prev.length
                assistantIndexRef.current = nextIndex
                return [...prev, { role: "assistant", content: chunk }]
              }
              const i = assistantIndexRef.current!
              const next = [...prev]
              const existing = next[i]
              next[i] = { role: "assistant", content: (existing?.content || "") + chunk, meta: existing?.meta }
              return next
            })
            continue
          }
          if (evt?.type === "tool_call") {
            const raw = String(evt?.content || "")
            const name = raw.replace(/^Calling\s+/, "").replace(/_/g, " ")
            const text = `Calling ${name.charAt(0).toUpperCase()}${name.slice(1)}`
            setMessages((prev) => [...prev, { role: "system", content: text }])
            continue
          }
          if (evt?.type === "final_result") {
            const chunk = String(evt?.content || "")
            const usage = evt?.usage
            const references = evt?.references
            setMessages((prev) => {
              if (assistantIndexRef.current == null) {
                const nextIndex = prev.length
                assistantIndexRef.current = nextIndex
                return [...prev, { role: "assistant", content: chunk, meta: { usage, references } }]
              }
              const i = assistantIndexRef.current!
              const next = [...prev]
              const existing = next[i]
              next[i] = { role: "assistant", content: chunk || existing?.content || "", meta: { usage, references } }
              return next
            })
            continue
          }
          if (evt?.type === "error") {
            const errMsg = String(evt?.error || "Unexpected error")
            setMessages((prev) => {
              if (assistantIndexRef.current == null) {
                const nextIndex = prev.length
                assistantIndexRef.current = nextIndex
                return [...prev, { role: "assistant", content: errMsg }]
              }
              const i = assistantIndexRef.current!
              const next = [...prev]
              const existing = next[i]
              next[i] = { role: "assistant", content: (existing?.content || "") + "\n" + errMsg, meta: existing?.meta }
              return next
            })
            continue
          }
        }
      }
    } catch {
    } finally {
      setIsStreaming(false)
      controllerRef.current = null
      // Reload conversations to update message count
      if (convId) {
        await loadConversations()
      }
    }
  }

  function cancelStream() {
    if (controllerRef.current) controllerRef.current.abort()
    setIsStreaming(false)
  }

  return (
    <div className="flex flex-1 gap-0">
      {/* Conversations Sidebar */}
      <div className="w-80 border-r flex flex-col bg-background">
        <div className="p-3 border-b">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Conversations</h2>
            <Button size="sm" onClick={createNewConversation} disabled={!selected}>
              <Plus className="size-4 mr-1" />
              New Chat
            </Button>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto">
          {loadingConversations ? (
            <div className="p-4 space-y-2">
              {Array(5).fill(0).map((_, i) => (
                <Skeleton key={i} className="h-16 w-full" />
              ))}
            </div>
          ) : conversations.length === 0 ? (
            <div className="p-4 text-center text-sm text-muted-foreground">
              No conversations yet.
              <br />
              Start a new chat!
            </div>
          ) : (
            <div className="p-2">
              {conversations.map((convo) => (
                <div
                  key={convo.id}
                  className={`group p-3 rounded-lg mb-1 cursor-pointer transition-colors ${
                    activeConversation?.id === convo.id
                      ? "bg-accent border border-primary"
                      : "hover:bg-accent/50"
                  }`}
                  onClick={() => selectConversation(convo)}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-sm truncate">{convo.title}</div>
                      <div className="text-xs text-muted-foreground mt-1">
                        {new Date(convo.updated_at).toLocaleDateString()}
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-8 shrink-0 opacity-0 group-hover:opacity-100 hover:opacity-100"
                      onClick={(e) => {
                        e.stopPropagation()
                        deleteConversationById(convo.id)
                      }}
                    >
                      <Trash2 className="size-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col">
        <div className="flex-1 min-h-0 p-4">
        <div ref={scrollRef} className="h-full overflow-y-auto flex flex-col gap-3">
          {messages.length === 0 ? (
            <div className="text-sm text-muted-foreground">No messages yet. Ask a question using the input below.</div>
          ) : (
            messages.map((m, i) => (
              <div key={i} className="flex flex-col items-start gap-2">
                {m.role === "assistant" ? (
                  <div className="flex items-start gap-2">
                    <Avatar className="size-6">
                      <AvatarFallback>
                        <Bot className="size-4" />
                      </AvatarFallback>
                    </Avatar>
                    <div className="max-w-[75vw]">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                    </div>
                  </div>
                ) : m.role === "user" ? (
                  <div className="self-end max-w-[75vw] rounded-lg bg-primary text-primary-foreground px-3 py-2">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                  </div>
                ) : (
                  <div className="self-start text-xs text-muted-foreground">
                    {m.content}
                  </div>
                )}
                {m.role === "assistant" && m.meta?.usage ? (
                  <div className="self-start flex items-center gap-2 text-xs text-muted-foreground">
                    <span className="inline-flex items-center gap-1 rounded bg-muted px-2 py-0.5">
                      <Hash className="size-3" /> In {Number(m.meta.usage?.input_tokens || 0)}
                    </span>
                    <span className="inline-flex items-center gap-1 rounded bg-muted px-2 py-0.5">
                      <Hash className="size-3" /> Out {Number(m.meta.usage?.output_tokens || 0)}
                    </span>
                    <span className="inline-flex items-center gap-1 rounded bg-muted px-2 py-0.5">
                      <BarChart3 className="size-3" /> Req {Number(m.meta.usage?.requests || 0)}
                    </span>
                    <span className="inline-flex items-center gap-1 rounded bg-muted px-2 py-0.5">
                      <Cpu className="size-3" /> Tools {Number(m.meta.usage?.tool_calls || 0)}
                    </span>
                    {Number(m.meta.usage?.cache_read_tokens || 0) > 0 ? (
                      <span className="inline-flex items-center gap-1 rounded bg-muted px-2 py-0.5">
                        <Database className="size-3" /> Cache {Number(m.meta.usage?.cache_read_tokens || 0)}
                      </span>
                    ) : null}
                    {m.meta?.references && (Array.isArray(m.meta.references.query_paths) && m.meta.references.query_paths.length > 0 || Array.isArray(m.meta.references.retrieved_pages) && m.meta.references.retrieved_pages.length > 0) ? (
                      <Sheet open={openRefsIndex === i} onOpenChange={(o) => setOpenRefsIndex(o ? i : null)}>
                        <SheetTrigger asChild>
                          <Button variant="ghost" size="sm" aria-label="References" className="inline-flex items-center gap-1">
                            <FileText className="size-3" />
                            References
                          </Button>
                        </SheetTrigger>
                        <SheetContent side="right" className="w-[504px] sm:w-[624px] sm:max-w-[624px]">
                          <SheetHeader>
                            <SheetTitle>References</SheetTitle>
                          </SheetHeader>
                          <div className="mt-4">
                            <Tabs value={refsTab} onValueChange={(v) => setRefsTab((v as "query" | "pages") ?? "query")}>
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
                    ) : null}
                  </div>
                ) : null}
              </div>
            ))
          )}
          </div>
        </div>

        <div className="border-t bg-background p-4">
        <form onSubmit={handleSubmit} className="flex items-center gap-2">
          <DropdownMenu>
            {selected ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" size="icon" aria-label="Selected Document" onMouseEnter={() => { if (!selectedDetail) loadSelectedDetail() }}>
                      <FileText className="size-4" />
                    </Button>
                  </DropdownMenuTrigger>
                </TooltipTrigger>
                <TooltipContent className="max-w-[320px]">
                  <div className="flex items-start gap-2">
                    <FileText className="size-4" />
                    <div className="min-w-0">
                      <div className="font-medium truncate">{selectedDetail?.title || labelFor(selected)}</div>
                      <div className="mt-1 text-xs text-balance text-background">{selectedDetail?.description || "No description"}</div>
                      <div className="mt-1 text-xs">{selectedDetail?.file_size !== undefined ? `Size ${formatBytes(selectedDetail?.file_size)}` : null}</div>
                    </div>
                  </div>
                </TooltipContent>
              </Tooltip>
            ) : (
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="icon" aria-label="Settings">
                  <Settings className="size-4" />
                </Button>
              </DropdownMenuTrigger>
            )}
            <DropdownMenuContent className="w-80">
              <DropdownMenuLabel>Document</DropdownMenuLabel>
              <div className="p-2">
                <Input placeholder="Search document..." value={docQuery} onChange={(e) => setDocQuery(e.target.value)} />
              </div>
              <DropdownMenuSeparator />
              {loadingDocs ? (
                <DropdownMenuItem disabled>Loading documents...</DropdownMenuItem>
              ) : docs.length === 0 ? (
                <DropdownMenuItem disabled>No documents</DropdownMenuItem>
              ) : (
                (() => {
                  const q = docQuery.trim().toLowerCase()
                  const filtered = q ? docs.filter((d) => (labelFor(d) || "").toLowerCase().includes(q)) : docs
                  if (filtered.length === 0) return <DropdownMenuItem disabled>No results</DropdownMenuItem>
                  return (
                    <div className="max-h-64 overflow-auto">
                      {filtered.map((d) => (
                        <DropdownMenuItem key={d.id} onClick={() => { setSelected(d); setDocQuery(""); }}>
                          {labelFor(d)}
                        </DropdownMenuItem>
                      ))}
                    </div>
                  )
                })()
              )}
            </DropdownMenuContent>
          </DropdownMenu>
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={selected ? "Type your question..." : "Select a document from settings"}
            disabled={!selected || isStreaming}
            className="flex-1"
          />
          <Button type="submit" disabled={!selected || !input.trim() || isStreaming} size="icon" aria-label="Send">
            {isStreaming ? <Spinner /> : <SendIcon className="size-4" />}
          </Button>
          {isStreaming ? (
            <Button type="button" variant="outline" onClick={cancelStream}>Stop</Button>
          ) : null}
          </form>
        </div>
      </div>
    </div>
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
