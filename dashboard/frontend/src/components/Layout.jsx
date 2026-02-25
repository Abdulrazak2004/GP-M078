import { useState, useRef, useEffect } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { Map, Activity, PenTool, Settings } from 'lucide-react';
import { useUnits } from '../contexts/UnitContext';
import { UNIT_SYSTEMS } from '../utils/units';

const navItems = [
  { to: '/', icon: Map, label: 'Map' },
  { to: '/monitor', icon: Activity, label: 'Monitor' },
  { to: '/designer', icon: PenTool, label: 'Designer' },
];

function UnitSelector() {
  const { system, setSystem } = useUnits();
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const handler = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(o => !o)}
        className="w-7 h-7 flex items-center justify-center rounded hover:bg-bg-elevated text-text-muted hover:text-text-primary transition-colors"
        title="Unit System"
      >
        <Settings size={14} />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1 w-56 bg-bg-surface border border-bg-border rounded-lg shadow-xl z-[2000] animate-fade-in">
          <div className="px-3 py-2 border-b border-bg-border">
            <div className="text-[10px] uppercase tracking-widest text-text-muted">Unit System</div>
          </div>
          {Object.entries(UNIT_SYSTEMS).map(([key, cfg]) => (
            <button
              key={key}
              onClick={() => { setSystem(key); setOpen(false); }}
              className={`w-full text-left px-3 py-2 flex items-center gap-2 transition-colors ${
                system === key
                  ? 'bg-accent-blue/10 text-accent-blue'
                  : 'text-text-secondary hover:bg-bg-elevated hover:text-text-primary'
              }`}
            >
              <div className={`w-1.5 h-1.5 rounded-full shrink-0 ${system === key ? 'bg-accent-blue' : 'bg-transparent'}`} />
              <div>
                <div className="text-xs font-medium">{cfg.label}</div>
                <div className="text-[10px] text-text-muted">{cfg.desc}</div>
              </div>
            </button>
          ))}
          <div className="px-3 py-2 border-t border-bg-border">
            <div className="text-[10px] text-text-muted leading-relaxed">
              API/Field = Saudi Aramco standard<br/>
              SI = ISO/NORSOK metric convention
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

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
        <div className="flex items-center gap-3">
          <UnitSelector />
          <div className="w-px h-4 bg-bg-border" />
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-accent-green" />
            <span className="text-xs text-text-muted">System Online</span>
          </div>
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
