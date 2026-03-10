"use client"

import { cn } from "@/lib/utils"
import {
  LayoutDashboard,
  Upload,
  BarChart3,
  GitCompare,
  Settings,
  Database,
  TrendingUp
} from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"

const navigation = [
  { name: "Dashboard", href: "/", icon: LayoutDashboard },
  { name: "Subir Dataset", href: "/upload", icon: Upload },
  { name: "Metricas", href: "/metrics", icon: BarChart3 },
  { name: "Comparacion", href: "/comparison", icon: GitCompare },
  { name: "Predicciones", href: "/predictions", icon: TrendingUp },
]

export function Sidebar() {
  const pathname = usePathname()

  return (
    <aside className="fixed inset-y-0 left-0 z-50 flex w-64 flex-col bg-sidebar border-r border-sidebar-border">
      <div className="flex h-16 items-center gap-3 border-b border-sidebar-border px-6">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary">
          <Database className="h-5 w-5 text-primary-foreground" />
        </div>
        <div className="flex flex-col">
          <span className="text-sm font-semibold text-sidebar-foreground">FIRE UdeA</span>
          <span className="text-xs text-muted-foreground">ML Dashboard</span>
        </div>
      </div>

      <nav className="flex-1 space-y-1 p-4">
        <div className="mb-4">
          <p className="px-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Menu Principal
          </p>
        </div>
        {navigation.map((item) => {
          const isActive = pathname === item.href
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                isActive
                  ? "bg-sidebar-accent text-sidebar-primary"
                  : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
              )}
            >
              <item.icon className="h-5 w-5" />
              {item.name}
            </Link>
          )
        })}
      </nav>

      <div className="border-t border-sidebar-border p-4">
        <Link
          href="/settings"
          className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-sidebar-foreground hover:bg-sidebar-accent"
        >
          <Settings className="h-5 w-5" />
          Configuracion
        </Link>
      </div>
    </aside>
  )
}
