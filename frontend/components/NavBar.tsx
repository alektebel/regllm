"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { GitBranch, Network, Table2, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/diff", label: "Diff", icon: Sparkles },
  { href: "/trace", label: "Trace", icon: GitBranch },
  { href: "/tabular", label: "Tabular", icon: Table2 },
] as const;

export default function NavBar() {
  const pathname = usePathname();

  return (
    <nav className="flex items-center gap-1 px-4 py-2 border-b border-border bg-card/60 shrink-0">
      <Link href="/" className="text-sm font-semibold text-foreground mr-4 flex items-center gap-1.5">
        <Network className="h-4 w-4 text-primary" />
        RegLLM
      </Link>
      {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
        const active = pathname === href || pathname?.startsWith(href + "/");
        return (
          <Link
            key={href}
            href={href}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1 rounded text-xs transition-colors",
              active
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:text-foreground hover:bg-muted/40",
            )}
          >
            <Icon className="h-3.5 w-3.5" />
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
