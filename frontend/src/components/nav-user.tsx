"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { ChevronsUpDown, LogOut, Settings, Book } from "lucide-react"

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar"

export function NavUser({
    user,
}: {
    user: {
        name: string
        email: string
        avatar: string
    }
}) {
    const { isMobile } = useSidebar()
    const router = useRouter()

    // Display profile values sourced from cache if available
    const [name, setName] = React.useState<string>(user.name)
    const [email, setEmail] = React.useState<string>(user.email)

    React.useEffect(() => {
      try {
        // Prefer explicit keys set on successful login
        const cachedUser = typeof window !== "undefined" ? localStorage.getItem("username") : null
        const cachedEmail = typeof window !== "undefined" ? localStorage.getItem("user_email") : null

        // Fallback to parsed auth_user if present
        const authRaw = typeof window !== "undefined" ? localStorage.getItem("auth_user") : null
        const authObj = authRaw ? JSON.parse(authRaw) : null
        const derivedName = cachedUser || authObj?.username || authObj?.user?.username
        const derivedEmail = cachedEmail || authObj?.email || authObj?.user?.email

        if (derivedName) setName(String(derivedName))
        if (derivedEmail) setEmail(String(derivedEmail))
      } catch (_) {
        // ignore parse errors
      }
    }, [])

    const onLogout = React.useCallback(() => {
      try {
        localStorage.removeItem("is_logged_in")
        localStorage.removeItem("username")
        localStorage.removeItem("user_email")
        localStorage.removeItem("auth_user")
      } catch (_) {
        // ignore storage errors
      }
      router.push("/login")
    }, [router])

    const initial = (name?.[0] ?? "U").toUpperCase()

    return (
        <SidebarMenu>
            <SidebarMenuItem>
                <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                        <SidebarMenuButton
                            size="lg"
                            className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
                        >
                            <Avatar className="h-8 w-8 rounded-lg">
                                <AvatarImage src={user.avatar} alt={name} />
                                <AvatarFallback className="rounded-lg">{initial}</AvatarFallback>
                            </Avatar>
                            <div className="grid flex-1 text-left text-sm leading-tight">
                                <span className="truncate font-medium">{name}</span>
                                <span className="truncate text-xs">{email}</span>
                            </div>
                            <ChevronsUpDown className="ml-auto size-4" />
                        </SidebarMenuButton>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent
                        className="w-(--radix-dropdown-menu-trigger-width) min-w-56 rounded-lg"
                        side={isMobile ? "bottom" : "right"}
                        align="end"
                        sideOffset={4}
                    >
                        <DropdownMenuLabel className="p-0 font-normal">
                            <div className="flex items-center gap-2 px-1 py-1.5 text-left text-sm">
                                <Avatar className="h-8 w-8 rounded-lg">
                                    <AvatarImage src={user.avatar} alt={name} />
                                    <AvatarFallback className="rounded-lg">{initial}</AvatarFallback>
                                </Avatar>
                                <div className="grid flex-1 text-left text-sm leading-tight">
                                    <span className="truncate font-medium">{name}</span>
                                    <span className="truncate text-xs">{email}</span>
                                </div>
                            </div>
                        </DropdownMenuLabel>
                        <DropdownMenuSeparator />
                        <DropdownMenuGroup>
                            <DropdownMenuItem>
                                <Settings />
                                Settings
                            </DropdownMenuItem>
                            <DropdownMenuItem
                                onSelect={() =>
                                    window.open(
                                        "https://github.com/AlvinWenJH/agentic-rag.git",
                                        "_blank",
                                        "noopener,noreferrer"
                                    )
                                }
                            >
                                <Book />
                                Documentation
                            </DropdownMenuItem>
                        </DropdownMenuGroup>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem onSelect={onLogout}>
                            <LogOut />
                            Log out
                        </DropdownMenuItem>
                    </DropdownMenuContent>
                </DropdownMenu>
            </SidebarMenuItem>
        </SidebarMenu>
    )
}
