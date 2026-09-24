import React, { useEffect, useState } from 'react';
import { fetchSentimentSummary, fetchModelPerformance } from '../services/api';
import type { SentimentSummaryResponse, ModelPerformanceResponse } from '../types';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { TrendingUp, Smile, Meh, Frown, Award } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell, PieChart, Pie } from 'recharts';

const SENTIMENT_COLORS = {
  positive: '#10b981', // emerald-500
  neutral: '#f59e0b',  // amber-500
  negative: '#ef4444', // rose-500
};

export const SentimentExplorerPage: React.FC = () => {
  const [sentimentData, setSentimentData] = useState<SentimentSummaryResponse | null>(null);
  const [performance, setPerformance] = useState<ModelPerformanceResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchSentimentSummary(), fetchModelPerformance()])
      .then(([sentRes, perfRes]) => {
        setSentimentData(sentRes);
        setPerformance(perfRes);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || 'Failed to load sentiment data');
        setLoading(false);
      });
  }, []);

  if (loading) return <LoadingSpinner message="Loading Sentiment Explorer Data from Backend..." />;
  if (error || !sentimentData) {
    return (
      <div className="bg-rose-50 border border-rose-200 text-rose-700 p-6 rounded-xl">
        <h3 className="font-bold">Error Loading Sentiment Data</h3>
        <p className="text-sm mt-1">{error || 'Unknown error occurred.'}</p>
      </div>
    );
  }

  const posFeatures = sentimentData.top_features.positive || [];
  const negFeatures = sentimentData.top_features.negative || [];

  const pieData = [
    { name: 'Positive', value: sentimentData.distribution.positive || 0, color: SENTIMENT_COLORS.positive },
    { name: 'Neutral', value: sentimentData.distribution.neutral || 0, color: SENTIMENT_COLORS.neutral },
    { name: 'Negative', value: sentimentData.distribution.negative || 0, color: SENTIMENT_COLORS.negative },
  ];

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-slate-800">Sentiment Explorer</h2>
        <p className="text-sm text-slate-500 mt-1">
          Explore student feedback sentiment distributions, top lexical predictors, and classifier metrics.
        </p>
      </div>

      {/* Distribution Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="bg-emerald-50/70 border border-emerald-200 rounded-xl p-5 flex items-start justify-between">
          <div>
            <p className="text-xs font-bold text-emerald-800 uppercase tracking-wider">Positive Evaluations</p>
            <h3 className="text-3xl font-extrabold text-emerald-900 mt-1.5">{sentimentData.percentages.positive || 0}%</h3>
            <p className="text-xs text-emerald-700 mt-1">
              {sentimentData.distribution.positive || 0} responses ({sentimentData.percentages.positive || 0}% of total)
            </p>
          </div>
          <div className="p-3 bg-emerald-100 text-emerald-700 rounded-xl">
            <Smile className="w-6 h-6" />
          </div>
        </div>

        <div className="bg-amber-50/70 border border-amber-200 rounded-xl p-5 flex items-start justify-between">
          <div>
            <p className="text-xs font-bold text-amber-800 uppercase tracking-wider">Neutral Evaluations</p>
            <h3 className="text-3xl font-extrabold text-amber-900 mt-1.5">{sentimentData.percentages.neutral || 0}%</h3>
            <p className="text-xs text-amber-700 mt-1">
              {sentimentData.distribution.neutral || 0} responses ({sentimentData.percentages.neutral || 0}% of total)
            </p>
          </div>
          <div className="p-3 bg-amber-100 text-amber-700 rounded-xl">
            <Meh className="w-6 h-6" />
          </div>
        </div>

        <div className="bg-rose-50/70 border border-rose-200 rounded-xl p-5 flex items-start justify-between">
          <div>
            <p className="text-xs font-bold text-rose-800 uppercase tracking-wider">Negative Evaluations</p>
            <h3 className="text-3xl font-extrabold text-rose-900 mt-1.5">{sentimentData.percentages.negative || 0}%</h3>
            <p className="text-xs text-rose-700 mt-1">
              {sentimentData.distribution.negative || 0} responses ({sentimentData.percentages.negative || 0}% of total)
            </p>
          </div>
          <div className="p-3 bg-rose-100 text-rose-700 rounded-xl">
            <Frown className="w-6 h-6" />
          </div>
        </div>
      </div>

      {/* Visual Sentiment Breakdown */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-base font-bold text-slate-800">Class Proportions & Volume</h3>
          <span className="text-xs text-slate-400">Total Analyzed: {sentimentData.total_analyzed} records</span>
        </div>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={45}
                outerRadius={75}
                paddingAngle={4}
                dataKey="value"
                label={({ name, percent }) => `${name}: ${((percent || 0) * 100).toFixed(0)}%`}
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#f8fafc', fontSize: '12px' }}
                itemStyle={{ color: '#f8fafc' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Feature Importance Indicators */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Top Positive Terms */}
        <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm space-y-4">
          <div className="flex items-center space-x-2 text-emerald-700">
            <TrendingUp className="w-5 h-5" />
            <h3 className="text-base font-bold text-slate-800">Top Positive NLP Features</h3>
          </div>
          <p className="text-xs text-slate-500">
            Terms with highest positive weighting in the TF-IDF feature vocabulary:
          </p>
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={posFeatures.slice(0, 8)} layout="vertical" margin={{ left: 10, right: 10 }}>
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis dataKey="term" type="category" width={90} tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#f8fafc', fontSize: '12px' }}
                  itemStyle={{ color: '#f8fafc' }}
                  formatter={(val: any) => [typeof val === 'number' ? val.toFixed(4) : val, 'Feature Weight']}
                />
                <Bar dataKey="coefficient" fill="#10b981" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top Negative Terms */}
        <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm space-y-4">
          <div className="flex items-center space-x-2 text-rose-700">
            <TrendingUp className="w-5 h-5 rotate-180" />
            <h3 className="text-base font-bold text-slate-800">Top Negative NLP Features</h3>
          </div>
          <p className="text-xs text-slate-500">
            Terms with highest negative weighting in the TF-IDF feature vocabulary:
          </p>
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={negFeatures.slice(0, 8)} layout="vertical" margin={{ left: 10, right: 10 }}>
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis dataKey="term" type="category" width={90} tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#f8fafc', fontSize: '12px' }}
                  itemStyle={{ color: '#f8fafc' }}
                  formatter={(val: any) => [typeof val === 'number' ? val.toFixed(4) : val, 'Feature Weight']}
                />
                <Bar dataKey="coefficient" fill="#ef4444" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Model Metrics Card */}
      {performance && (
        <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm space-y-4">
          <div className="flex items-center space-x-2">
            <Award className="w-5 h-5 text-sky-600" />
            <h3 className="text-base font-bold text-slate-800">Production Sentiment Classifier Metrics</h3>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-1">
            <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg text-center">
              <p className="text-[11px] text-slate-500 uppercase font-semibold">Active Model</p>
              <p className="text-sm font-bold text-slate-800 mt-1">{performance.best_model}</p>
            </div>
            <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg text-center">
              <p className="text-[11px] text-slate-500 uppercase font-semibold">Accuracy</p>
              <p className="text-sm font-bold text-emerald-700 mt-1">
                {(performance.models_benchmark[performance.best_model]?.Accuracy * 100).toFixed(1)}%
              </p>
            </div>
            <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg text-center">
              <p className="text-[11px] text-slate-500 uppercase font-semibold">Precision</p>
              <p className="text-sm font-bold text-sky-700 mt-1">
                {(performance.models_benchmark[performance.best_model]?.Precision * 100).toFixed(1)}%
              </p>
            </div>
            <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg text-center">
              <p className="text-[11px] text-slate-500 uppercase font-semibold">F1-Score (Weighted)</p>
              <p className="text-sm font-bold text-indigo-700 mt-1">
                {(performance.models_benchmark[performance.best_model]?.F1_Score * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
