"use client"

import * as React from "react"
import { GalleryVerticalEnd, Home as HomeIcon, FileText, Folder, Wrench, MessageSquare, ClipboardList, type LucideIcon } from "lucide-react"
import { usePathname } from "next/navigation"

import {
    Sidebar,
    SidebarContent,
    SidebarGroup,
    SidebarHeader,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
    SidebarMenuSub,
    SidebarMenuSubButton,
    SidebarMenuSubItem,
    SidebarFooter,
    SidebarRail,
} from "@/components/ui/sidebar"

import { NavUser } from "@/components/nav-user"

type NavUser = {
    name: string
    email: string
    avatar: string
}

type NavSubItem = {
    title: string
    url: string
    isActive?: boolean
    icon?: LucideIcon
}

type NavMainItem = {
    title: string
    url: string
    isActive?: boolean
    icon?: LucideIcon
    items?: NavSubItem[]
}

// This is sample data.
const data: { user: NavUser, navMain: NavMainItem[] } = {
    user: {
        name: "shadcn",
        email: "m@example.com",
        avatar: "/avatars/shadcn.jpg",
    },
    navMain: [
        { title: "Home", url: "/home", icon: HomeIcon },
        {
            title: "Resources",
            url: "#",
            icon: Folder,
            items: [
                {
                    title: "Documents",
                    url: "/documents",
                    icon: FileText,
                },

            ],
        },
        {
            title: "Application",
            url: "#",
            icon: Wrench,
            items: [
                {
                    title: "Chat",
                    url: "/query",
                    icon: MessageSquare,
                },
                {
                    title: "Document Analysis",
                    url: "/analysis",
                    icon: ClipboardList,
                }
            ],
        }
    ],
}

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
    const pathname = usePathname()
    return (
        <Sidebar {...props}>
            <SidebarHeader>
                <SidebarMenu>
                    <SidebarMenuItem>
                        <SidebarMenuButton size="lg" asChild>
                            <a href="#">
                                <div className="bg-sidebar-primary text-sidebar-primary-foreground flex aspect-square size-8 items-center justify-center rounded-lg">
                                    <GalleryVerticalEnd className="size-4" />
                                </div>
                                <div className="flex flex-col gap-0.5 leading-none">
                                    <span className="font-medium">Archivist</span>
                                    <span className="">v0.0.1</span>
                                </div>
                            </a>
                        </SidebarMenuButton>
                    </SidebarMenuItem>
                </SidebarMenu>
            </SidebarHeader>
            <SidebarContent>
                <SidebarGroup>
                    <SidebarMenu>
                        {data.navMain.map((item) => (
                            <SidebarMenuItem key={item.title}>
                                <SidebarMenuButton asChild>
                                    <a href={item.url} className="font-medium flex items-center gap-2">
                                        {item.icon ? <item.icon className="size-4" /> : null}
                                        <span>{item.title}</span>
                                    </a>
                                </SidebarMenuButton>
                                {item.items?.length ? (
                                    <SidebarMenuSub>
                                        {item.items.map((sub) => (
                                            <SidebarMenuSubItem key={sub.title}>
                                                <SidebarMenuSubButton asChild isActive={pathname === sub.url}>
                                                    <a href={sub.url} className="flex items-center gap-2">
                                                        {sub.icon ? <sub.icon className="size-4" /> : null}
                                                        <span>{sub.title}</span>
                                                    </a>
                                                </SidebarMenuSubButton>
                                            </SidebarMenuSubItem>
                                        ))}
                                    </SidebarMenuSub>
                                ) : null}
                            </SidebarMenuItem>
                        ))}
                    </SidebarMenu>
                </SidebarGroup>
            </SidebarContent>
            <SidebarFooter>
                <NavUser user={data.user} />
            </SidebarFooter>
            <SidebarRail />
        </Sidebar>
    )
}
