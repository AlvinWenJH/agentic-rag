"use client"

import * as React from "react"
import { useTheme } from "next-themes"
import { AppSidebar } from "@/components/app-sidebar"
import BreadcrumbCurrentPage from "@/components/breadcrumb-current-page"
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"
import { Separator } from "@/components/ui/separator"
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export default function Page() {
  const { theme, setTheme, resolvedTheme } = useTheme()

  const current = theme || resolvedTheme || "system"

  const applyTheme = React.useCallback((t: string) => {
    setTheme(t)
  }, [setTheme])

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 border-b">
          <div className="flex items-center gap-2 px-3">
            <SidebarTrigger />
            <Separator orientation="vertical" className="mr-2 h-4" />
            <Breadcrumb>
              <BreadcrumbList>
                <BreadcrumbItem className="hidden md:block">
                  <BreadcrumbLink href="#">Archivist</BreadcrumbLink>
                </BreadcrumbItem>
                <BreadcrumbSeparator className="hidden md:block" />
                <BreadcrumbItem>
                  <BreadcrumbPage>
                    <BreadcrumbCurrentPage />
                  </BreadcrumbPage>
                </BreadcrumbItem>
              </BreadcrumbList>
            </Breadcrumb>
          </div>
        </header>
        <div className="flex flex-1 flex-col gap-4 p-4">
          <div className="grid gap-4 max-w-3xl">
            <Card>
              <CardHeader className="border-b">
                <CardTitle>Appearance</CardTitle>
                <CardDescription>Choose how Archivist looks on your device</CardDescription>
              </CardHeader>
              <CardContent className="pt-4">
                <div className="flex items-center gap-3">
                  <ThemeOption label="Light" value="light" active={current === "light"} onClick={applyTheme} />
                  <ThemeOption label="Dark" value="dark" active={current === "dark"} onClick={applyTheme} />
                  <ThemeOption label="System" value="system" active={current === "system"} onClick={applyTheme} />
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}

function ThemeOption({ label, value, active, onClick }: { label: string; value: string; active: boolean; onClick: (t: string) => void }) {
  return (
    <Button
      variant={active ? "default" : "outline"}
      onClick={() => onClick(value)}
      className={cn("min-w-24", active && "ring-2 ring-ring")}
    >
      {label}
    </Button>
  )
}