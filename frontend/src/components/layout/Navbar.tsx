import React from 'react';
import { GraduationCap, Menu } from 'lucide-react';

interface NavbarProps {
  onToggleMobileMenu?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ onToggleMobileMenu }) => {
  return (
    <header className="h-16 bg-white border-b border-slate-200 sticky top-0 z-30 px-4 sm:px-6 flex items-center justify-between shadow-sm">
      <div className="flex items-center space-x-3">
        {onToggleMobileMenu && (
          <button
            type="button"
            onClick={onToggleMobileMenu}
            className="md:hidden p-2 rounded-lg text-slate-600 hover:bg-slate-100 focus:outline-none"
            aria-label="Toggle Navigation Menu"
          >
            <Menu className="w-5 h-5" />
          </button>
        )}
        <div className="w-9 h-9 bg-sky-600 rounded-lg flex items-center justify-center text-white shadow-sm shadow-sky-200 shrink-0">
          <GraduationCap className="w-5 h-5" />
        </div>
        <div>
          <h1 className="text-sm sm:text-base font-bold text-slate-900 leading-tight">
            Student Feedback Intelligence System
          </h1>
          <p className="text-[11px] sm:text-xs text-slate-500 font-medium hidden xs:block">
            Sentiment Analysis & LDA Topic Modelling • Real-time NLP Analysis
          </p>
        </div>
      </div>
      <div className="flex items-center space-x-3 sm:space-x-4">
        <div className="hidden sm:flex items-center space-x-2 text-xs bg-emerald-50 text-emerald-700 px-3 py-1.5 rounded-full border border-emerald-200 font-medium">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span>FastAPI NLP Engine Active</span>
        </div>
        <div className="text-[11px] sm:text-xs font-mono text-slate-400 bg-slate-100 px-2.5 py-1 rounded-md">
          In-Memory NLP • No DB
        </div>
      </div>
    </header>
  );
};
