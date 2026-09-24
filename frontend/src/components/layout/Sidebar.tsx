import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Sparkles,
  UploadCloud,
  Compass,
  SmilePlus,
  BarChart3,
  X,
} from 'lucide-react';

const navItems = [
  { name: 'Dashboard', path: '/', icon: LayoutDashboard },
  { name: 'Analyze Feedback', path: '/analyze', icon: Sparkles },
  { name: 'Bulk Analysis', path: '/bulk', icon: UploadCloud },
  { name: 'Topic Explorer', path: '/topics', icon: Compass },
  { name: 'Sentiment Explorer', path: '/sentiment', icon: SmilePlus },
  { name: 'Model Performance', path: '/performance', icon: BarChart3 },
];

interface SidebarProps {
  mobileOpen?: boolean;
  onCloseMobile?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ mobileOpen, onCloseMobile }) => {
  return (
    <>
      {/* Mobile Backdrop */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-slate-900/60 z-40 md:hidden backdrop-blur-sm transition-opacity"
          onClick={onCloseMobile}
        />
      )}

      <aside
        className={`
          fixed md:static inset-y-0 left-0 z-50 md:z-auto
          w-64 bg-slate-900 text-slate-300 min-h-[calc(100vh-4rem)] flex flex-col justify-between p-4 shrink-0
          transform transition-transform duration-200 ease-in-out
          ${mobileOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
        `}
      >
        <div>
          <div className="flex items-center justify-between px-3 mb-3">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Navigation
            </span>
            {onCloseMobile && (
              <button
                type="button"
                onClick={onCloseMobile}
                className="md:hidden text-slate-400 hover:text-white p-1"
                aria-label="Close Sidebar"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
          <nav className="space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  end={item.path === '/'}
                  onClick={onCloseMobile}
                  className={({ isActive }) =>
                    `flex items-center space-x-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-sky-600 text-white shadow-sm'
                        : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                    }`
                  }
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  <span>{item.name}</span>
                </NavLink>
              );
            })}
          </nav>
        </div>

        <div className="bg-slate-800/80 rounded-xl p-3.5 border border-slate-700/50 text-xs text-slate-400 space-y-1.5 mt-6">
          <p className="font-semibold text-slate-200">System Architecture</p>
          <p>• FastAPI Backend (Port 8000)</p>
          <p>• Scikit-Learn LDA ($K=5$)</p>
          <p>• TF-IDF Multi-Class Sentiment</p>
          <p>• In-Memory / No Database</p>
        </div>
      </aside>
    </>
  );
};
