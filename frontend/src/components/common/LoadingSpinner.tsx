import React from 'react';

export const LoadingSpinner: React.FC<{ message?: string }> = ({ message = 'Processing NLP models...' }) => {
  return (
    <div className="flex flex-col items-center justify-center p-12 space-y-4">
      <div className="w-10 h-10 border-4 border-sky-200 border-t-sky-600 rounded-full animate-spin"></div>
      <p className="text-sm font-medium text-slate-600">{message}</p>
    </div>
  );
};
