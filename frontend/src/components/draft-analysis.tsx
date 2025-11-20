"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { Wand2, Plus, Trash2, Loader2, ArrowLeft } from "lucide-react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { getBackendUrl } from "@/lib/env"
import { toast } from "sonner"

type AnalysisItem = {
  question: string
  context?: string
  order: number
}

export default function DraftAnalysis() {
  const router = useRouter()
  const backendUrl = getBackendUrl()

  // Draft generation state
  const [draftText, setDraftText] = React.useState("")
  const [generating, setGenerating] = React.useState(false)

  // Form state
  const [title, setTitle] = React.useState("")
  const [description, setDescription] = React.useState("")
  const [items, setItems] = React.useState<AnalysisItem[]>([])
  const [submitting, setSubmitting] = React.useState(false)

  async function handleGenerateDraft() {
    if (!draftText.trim()) {
      toast.error("Please enter a description")
      return
    }
    setGenerating(true)
    try {
      const res = await fetch(`${backendUrl}/api/v1/analysis/draft`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
        },
        body: JSON.stringify({ text: draftText }),
      })
      if (!res.ok) throw new Error(`Failed to generate draft`)
      const data = await res.json()

      setTitle(data.title)
      setDescription(data.description)
      setItems(data.items)
      toast.success("Draft generated successfully")
      setDraftText("")
    } catch (err) {
      console.error(err)
      toast.error("Failed to generate draft")
    } finally {
      setGenerating(false)
    }
  }

  async function handleSubmit(e?: React.FormEvent) {
    if (e) e.preventDefault()

    if (!title.trim()) {
      toast.error("Please enter a title")
      return
    }

    if (items.length === 0) {
      toast.error("Please add at least one analysis item")
      return
    }

    setSubmitting(true)
    try {
      // Get user_id from localStorage
      let userId: string | null = null
      try {
        const raw = typeof window !== "undefined" ? localStorage.getItem("auth_user") : null
        const obj = raw ? JSON.parse(raw) : null
        userId = (obj?.user_id ?? obj?.id ?? null) as string | null
      } catch (_) {}

      const res = await fetch(`${backendUrl}/api/v1/analysis/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
        },
        body: JSON.stringify({
          title,
          description,
          items,
          tags: [],
          user_id: userId,
        }),
      })

      if (!res.ok) throw new Error(`Failed to create analysis`)

      const newAnalysis = await res.json()
      toast.success("Analysis created successfully")

      // Navigate back to analysis list
      router.push("/analysis")
    } catch (err) {
      console.error(err)
      toast.error("Failed to create analysis")
    } finally {
      setSubmitting(false)
    }
  }

  function addItem() {
    setItems([...items, { question: "", context: "", order: items.length + 1 }])
  }

  function removeItem(index: number) {
    setItems(items.filter((_, i) => i !== index).map((item, i) => ({ ...item, order: i + 1 })))
  }

  function updateItem(index: number, field: "question" | "context", value: string) {
    setItems(
      items.map((item, i) => (i === index ? { ...item, [field]: value } : item))
    )
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon" onClick={() => router.push("/analysis")}>
            <ArrowLeft className="size-4" />
          </Button>
          <h1 className="text-2xl font-semibold">Draft New Analysis</h1>
        </div>
        <div className="flex items-center gap-2">
          <Button type="button" variant="outline" onClick={() => router.push("/analysis")}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={submitting}>
            {submitting ? <Loader2 className="size-4 mr-2 animate-spin" /> : null}
            Create Analysis
          </Button>
        </div>
      </div>

      {/* Two Column Layout: 1/3 left, 2/3 right */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left Column - AI Generation (1/3 width) */}
        <div className="space-y-4 lg:col-span-1">
          <Card className="h-fit sticky top-4">
            <CardHeader>
              <CardTitle>Generate with AI</CardTitle>
              <CardDescription>
                Describe what you want to analyze, and AI will generate structured analysis items
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-2 block">Analysis Description</label>
                <textarea
                  value={draftText}
                  onChange={(e) => setDraftText(e.target.value)}
                  className="w-full min-h-[200px] p-3 rounded-md border border-input bg-background resize-none focus:outline-none focus:ring-2 focus:ring-ring"
                  placeholder="Example: 'I want to analyze financial reports for:&#10;- Revenue trends and growth patterns&#10;- Cost analysis and expense breakdown&#10;- Profit margins and operational efficiency'"
                />
              </div>
              <Button onClick={handleGenerateDraft} disabled={generating} className="w-full">
                {generating ? (
                  <Loader2 className="size-4 mr-2 animate-spin" />
                ) : (
                  <Wand2 className="size-4 mr-2" />
                )}
                Generate Draft
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Right Column - Form (2/3 width) */}
        <div className="space-y-4 lg:col-span-2">
          {/* Title and Description */}
          <Card>
            <CardHeader>
              <CardTitle>Analysis Details</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-2 block">Title *</label>
                <Input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Enter analysis title"
                  required
                />
              </div>
              <div>
                <label className="text-sm font-medium mb-2 block">Description</label>
                <Input
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Enter analysis description"
                />
              </div>
            </CardContent>
          </Card>

          {/* Analysis Items */}
          <Card>
            <CardHeader className="border-b">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Analysis Items *</CardTitle>
                  <CardDescription className="mt-1">
                    {items.length} {items.length === 1 ? 'item' : 'items'}
                  </CardDescription>
                </div>
                <Button type="button" onClick={addItem} size="sm">
                  <Plus className="size-4 mr-1" />
                  Add Item
                </Button>
              </div>
            </CardHeader>
            <CardContent className="pt-4">
              {items.length === 0 ? (
                <div className="text-center py-12 border-2 border-dashed rounded-lg">
                  <p className="text-sm text-muted-foreground mb-3">
                    No analysis items yet
                  </p>
                  <Button type="button" onClick={addItem} size="sm" variant="outline">
                    <Plus className="size-4 mr-1" />
                    Add First Item
                  </Button>
                </div>
              ) : (
                <div className="space-y-3">
                  {items.map((item, index) => (
                    <div key={index} className="border rounded-lg p-4 bg-muted/30">
                      <div className="flex items-start gap-3">
                        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-sm font-semibold text-primary">
                          {index + 1}
                        </div>
                        <div className="flex-1 space-y-3">
                          <div>
                            <label className="text-xs font-medium text-muted-foreground mb-1.5 block">
                              Question *
                            </label>
                            <Input
                              value={item.question}
                              onChange={(e) => updateItem(index, "question", e.target.value)}
                              placeholder="What question should this analysis answer?"
                              required
                              className="bg-background"
                            />
                          </div>
                          <div>
                            <label className="text-xs font-medium text-muted-foreground mb-1.5 block">
                              Context (Optional)
                            </label>
                            <Input
                              value={item.context || ""}
                              onChange={(e) => updateItem(index, "context", e.target.value)}
                              placeholder="Additional context or instructions"
                              className="bg-background"
                            />
                          </div>
                        </div>
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon-sm"
                          onClick={() => removeItem(index)}
                          className="flex-shrink-0"
                        >
                          <Trash2 className="size-4 text-destructive" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

