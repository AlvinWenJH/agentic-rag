"use client"

import { usePathname } from "next/navigation"

export default function BreadcrumbCurrentPage() {
  const pathname = usePathname() || "/"
  const segments = pathname.split("/").filter(Boolean)
  const last = segments[segments.length - 1] || "home"
  const prev = segments.length > 1 ? segments[segments.length - 2] : undefined

  // For dynamic document routes like /documents/[id], show "Documents"
  const title = prev === "documents"
    ? "Documents"
    : last
        .replace(/-/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase())

  return <span>{title}</span>
}