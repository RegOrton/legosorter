"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function Header() {
    const pathname = usePathname();

    const isActive = (path: string) => {
        return pathname === path ? "bg-zinc-800 text-white" : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900";
    };

    return (
        <header className="flex h-16 items-center border-b border-zinc-800 px-6 bg-zinc-950/50 backdrop-blur-md sticky top-0 z-50">
            <div className="flex items-center gap-6">
                <div className="flex items-center gap-3">
                    <div className="h-6 w-6 rounded bg-gradient-to-br from-red-500 to-yellow-500" />
                    <h1 className="text-lg font-bold tracking-tight text-white">Lego Sorter <span className="text-zinc-500 font-normal">| Debug Console</span></h1>
                </div>

                <nav className="flex items-center gap-1 ml-6 pl-6 border-l border-zinc-800 h-8">
                    <Link
                        href="/"
                        className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${isActive("/")}`}
                    >
                        Dashboard
                    </Link>
                    <Link
                        href="/training"
                        className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${isActive("/training")}`}
                    >
                        Training Center
                    </Link>
                </nav>
            </div>

            <div className="ml-auto flex items-center gap-6">
                <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-zinc-900 border border-zinc-800">
                    <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
                    <span className="text-xs font-medium text-zinc-400">System Online</span>
                </div>
            </div>
        </header>
    );
}
