import React, { useEffect, useState } from 'react';
import { fetchDashboardSummary, fetchModelPerformance } from '../services/api';
import type { DashboardSummaryResponse, ModelPerformanceResponse } from '../types';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { TrendingUp } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';

export const SentimentAnalysisPage: React.FC = () => {
  const [dashboard, setDashboard] = useState<DashboardSummaryResponse | null>(null);
  const [performance, setPerformance] = useState<ModelPerformanceResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    Promise.all([fetchDashboardSummary(), fetchModelPerformance()])
      .then(([dashRes, perfRes]) => {
        setDashboard(dashRes);
        setPerformance(perfRes);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner message="Loading Sentiment Engine Analysis..." />;

  const posFeatures = performance?.feature_importance?.positive || [];
  const negFeatures = performance?.feature_importance?.negative || [];

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-slate-800">Supervised Sentiment Analysis</h2>
        <p className="text-sm text-slate-500 mt-1">
          Multi-class text classification pipeline powered by TF-IDF N-Grams (1-2) and Calibrated Logistic Regression.
        </p>
      </div>

      {/* Summary Distribution */}
      {dashboard && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <div className="bg-emerald-50/70 border border-emerald-200 rounded-xl p-5">
            <h4 className="text-xs font-bold text-emerald-800 uppercase tracking-wider">Positive Feedback</h4>
            <p className="text-3xl font-extrabold text-emerald-900 mt-2">{dashboard.sentiment_percentages.positive}%</p>
            <p className="text-xs text-emerald-700 mt-1">{dashboard.sentiment_distribution.positive} students expressed strong satisfaction</p>
          </div>
          <div className="bg-amber-50/70 border border-amber-200 rounded-xl p-5">
            <h4 className="text-xs font-bold text-amber-800 uppercase tracking-wider">Neutral / Constructive</h4>
            <p className="text-3xl font-extrabold text-amber-900 mt-2">{dashboard.sentiment_percentages.neutral}%</p>
            <p className="text-xs text-amber-700 mt-1">{dashboard.sentiment_distribution.neutral} balanced responses with suggestions</p>
          </div>
          <div className="bg-rose-50/70 border border-rose-200 rounded-xl p-5">
            <h4 className="text-xs font-bold text-rose-800 uppercase tracking-wider">Negative / Issues</h4>
            <p className="text-3xl font-extrabold text-rose-900 mt-2">{dashboard.sentiment_percentages.negative}%</p>
            <p className="text-xs text-rose-700 mt-1">{dashboard.sentiment_distribution.negative} responses detailing pacing/slide problems</p>
          </div>
        </div>
      )}

      {/* Feature Importance Indicators */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Top Positive Predictors */}
        <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm space-y-4">
          <div className="flex items-center space-x-2 text-emerald-700">
            <TrendingUp className="w-5 h-5" />
            <h3 className="text-base font-bold text-slate-800">Top Positive NLP Features</h3>
          </div>
          <p className="text-xs text-slate-500">
            TF-IDF terms with highest positive coefficients in the classification model:
          </p>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={posFeatures.slice(0, 7)} layout="vertical" margin={{ left: 10, right: 10 }}>
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis dataKey="term" type="category" width={90} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="coefficient" fill="#10b981" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top Negative Predictors */}
        <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm space-y-4">
          <div className="flex items-center space-x-2 text-rose-700">
            <TrendingUp className="w-5 h-5 rotate-180" />
            <h3 className="text-base font-bold text-slate-800">Top Negative NLP Features</h3>
          </div>
          <p className="text-xs text-slate-500">
            TF-IDF terms with highest negative coefficients in the classification model:
          </p>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={negFeatures.slice(0, 7)} layout="vertical" margin={{ left: 10, right: 10 }}>
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis dataKey="term" type="category" width={90} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="coefficient" fill="#ef4444" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* NLP Methodology Details */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm space-y-3">
        <h3 className="text-base font-bold text-slate-800">Sentiment Classification Pipeline Details</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs text-slate-600 mt-2">
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200/60">
            <p className="font-semibold text-slate-800 mb-1">Feature Extraction</p>
            <p>1000 sublinear TF-IDF unigrams and bigrams extracted after WordNet lemmatization and negation-aware stopword removal.</p>
          </div>
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200/60">
            <p className="font-semibold text-slate-800 mb-1">Model Selection</p>
            <p>Trained using Logistic Regression with multinomial cross-entropy and evaluated against Naive Bayes and Linear SVM.</p>
          </div>
        </div>
      </div>
    </div>
  );
};
