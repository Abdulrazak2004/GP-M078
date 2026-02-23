import { NavLink, Outlet } from 'react-router-dom';
import { Map, Activity, PenTool } from 'lucide-react';

const navItems = [
  { to: '/', icon: Map, label: 'Map' },
  { to: '/monitor', icon: Activity, label: 'Monitor' },
  { to: '/designer', icon: PenTool, label: 'Designer' },
];

export default function Layout() {
  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="h-11 bg-bg-surface border-b border-bg-border flex items-center px-4 shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-accent-cyan" />
          <span className="text-xs font-semibold tracking-widest uppercase text-text-secondary">
            Corrosion Intelligence
          </span>
        </div>
        <div className="flex-1 text-center">
          <span className="text-sm font-semibold tracking-tight text-text-primary" id="page-title">
            Operations Overview
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-accent-green" />
          <span className="text-xs text-text-muted">System Online</span>
        </div>
      </header>

      <div className="flex flex-1 min-h-0">
        {/* Sidebar */}
        <nav className="w-14 bg-bg-surface border-r border-bg-border flex flex-col items-center py-3 gap-1 shrink-0">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `w-10 h-10 flex items-center justify-center rounded-lg transition-colors ${
                  isActive
                    ? 'bg-accent-blue/15 text-accent-blue'
                    : 'text-text-muted hover:text-text-secondary hover:bg-bg-elevated'
                }`
              }
              title={label}
            >
              <Icon size={18} />
            </NavLink>
          ))}
        </nav>

        {/* Content */}
        <main className="flex-1 min-h-0 overflow-hidden">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
