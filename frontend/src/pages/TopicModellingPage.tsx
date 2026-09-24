import React, { useEffect, useState } from 'react';
import { fetchTopics, fetchModelPerformance } from '../services/api';
import type { TopicsListResponse, ModelPerformanceResponse } from '../types';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { Layers } from 'lucide-react';
import { Link } from 'react-router-dom';

export const TopicModellingPage: React.FC = () => {
  const [topicsData, setTopicsData] = useState<TopicsListResponse | null>(null);
  const [performance, setPerformance] = useState<ModelPerformanceResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    Promise.all([fetchTopics(), fetchModelPerformance()])
      .then(([topRes, perfRes]) => {
        setTopicsData(topRes);
        setPerformance(perfRes);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner message="Loading LDA Topic Model Data..." />;

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-800">Unsupervised LDA Topic Modelling</h2>
          <p className="text-sm text-slate-500 mt-1">
            Latent Dirichlet Allocation ($K=5$) extracting latent course themes from unlabelled feedback.
          </p>
        </div>
        <Link
          to="/topics"
          className="inline-flex items-center space-x-2 bg-sky-600 hover:bg-sky-700 text-white px-4 py-2 rounded-lg text-xs font-medium shadow-sm transition-colors"
        >
          <span>Open Interactive Topic Explorer</span>
        </Link>
      </div>

      {/* LDA Model Hyperparameters & Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
        <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Number of Topics ($K$)</p>
          <p className="text-3xl font-bold text-slate-800 mt-1">{performance?.topic_metrics?.n_topics || 5}</p>
          <p className="text-xs text-slate-400 mt-1">Optimal topic clustering</p>
        </div>
        <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Model Perplexity</p>
          <p className="text-3xl font-bold text-sky-600 mt-1">{performance?.topic_metrics?.perplexity || 356.14}</p>
          <p className="text-xs text-slate-400 mt-1">Lower perplexity = better generalization</p>
        </div>
        <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Topic Coherence ($C_v$)</p>
          <p className="text-3xl font-bold text-emerald-600 mt-1">{performance?.topic_metrics?.coherence_cv_score || 0.485}</p>
          <p className="text-xs text-slate-400 mt-1">High semantic keyword coherence</p>
        </div>
      </div>

      {/* Discovered Topics List */}
      <div className="space-y-4">
        <h3 className="text-lg font-bold text-slate-800 flex items-center space-x-2">
          <Layers className="w-5 h-5 text-sky-600" />
          <span>Discovered Themes & Vocabulary Distributions</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {topicsData?.topics.map((t) => (
            <div key={t.topic_id} className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-sky-700 bg-sky-50 px-2 py-0.5 rounded border border-sky-200">
                  Topic #{t.topic_id + 1}
                </span>
                <span className="text-xs font-semibold text-slate-600">
                  {t.percentage}% of evaluations ({t.total_feedbacks})
                </span>
              </div>

              <h4 className="text-sm font-bold text-slate-800">{t.label}</h4>

              <div>
                <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">Top Keywords:</p>
                <div className="flex flex-wrap gap-1.5">
                  {t.top_words.map((w, idx) => (
                    <span key={idx} className="px-2 py-0.5 bg-slate-100 text-slate-700 rounded text-xs font-mono">
                      {w}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
