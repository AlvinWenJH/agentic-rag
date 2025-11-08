"use client"

import { usePathname } from "next/navigation"

export default function BreadcrumbCurrentPage() {
  const pathname = usePathname() || "/"
  const segments = pathname.split("/").filter(Boolean)
  const last = segments[segments.length - 1] || "home"
  const title = last
    .replace(/-/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())

  return <span>{title}</span>
}