import React, { useEffect, useState } from 'react';
import { fetchTopics } from '../services/api';
import type { TopicsListResponse } from '../types';
import { Badge } from '../components/common/Badge';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { MessageSquareQuote, Layers } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';

export const TopicExplorerPage: React.FC = () => {
  const [data, setData] = useState<TopicsListResponse | null>(null);
  const [selectedTopicId, setSelectedTopicId] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchTopics()
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner message="Loading Topic Explorer..." />;

  const topics = data?.topics || [];
  const selectedTopic = topics.find((t) => t.topic_id === selectedTopicId) || topics[0];

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-slate-800">Topic Theme Explorer</h2>
        <p className="text-sm text-slate-500 mt-1">
          Deep-dive into each discovered course topic: inspect keyword weights, sentiment distribution, and representative quotes.
        </p>
      </div>

      {/* Topic Tabs */}
      <div className="flex flex-wrap gap-2 border-b border-slate-200 pb-3">
        {topics.map((t) => (
          <button
            key={t.topic_id}
            type="button"
            onClick={() => setSelectedTopicId(t.topic_id)}
            className={`px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
              selectedTopicId === t.topic_id
                ? 'bg-sky-600 text-white shadow-sm'
                : 'bg-white text-slate-700 border border-slate-200 hover:bg-slate-50'
            }`}
          >
            Topic #{t.topic_id + 1}: {t.label}
          </button>
        ))}
      </div>

      {selectedTopic && (
        <div className="space-y-6">
          {/* Overview Card */}
          <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <span className="text-xs font-bold text-sky-700 bg-sky-50 px-2.5 py-1 rounded border border-sky-200">
                Topic #{selectedTopic.topic_id + 1}
              </span>
              <h3 className="text-xl font-bold text-slate-800 mt-2">{selectedTopic.label}</h3>
              <p className="text-xs text-slate-500 mt-0.5">
                Representing <span className="font-semibold text-slate-800">{selectedTopic.percentage}%</span> of all student feedback ({selectedTopic.total_feedbacks} comments)
              </p>
            </div>

            {/* Sentiment Breakdown for Topic */}
            <div className="flex gap-4 text-xs font-medium">
              <div className="text-center p-3 bg-emerald-50 border border-emerald-200 rounded-lg">
                <span className="text-emerald-800 font-bold block text-lg">{selectedTopic.sentiment_breakdown.positive || 0}</span>
                <span className="text-emerald-600">Positive</span>
              </div>
              <div className="text-center p-3 bg-amber-50 border border-amber-200 rounded-lg">
                <span className="text-amber-800 font-bold block text-lg">{selectedTopic.sentiment_breakdown.neutral || 0}</span>
                <span className="text-amber-600">Neutral</span>
              </div>
              <div className="text-center p-3 bg-rose-50 border border-rose-200 rounded-lg">
                <span className="text-rose-800 font-bold block text-lg">{selectedTopic.sentiment_breakdown.negative || 0}</span>
                <span className="text-rose-600">Negative</span>
              </div>
            </div>
          </div>

          {/* Grid: Word Weights & Representative Quotes */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Word Weights Chart */}
            <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm space-y-4">
              <div className="flex items-center space-x-2">
                <Layers className="w-5 h-5 text-sky-600" />
                <h4 className="text-sm font-bold text-slate-800">Top Keyword Term Weights in Topic</h4>
              </div>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={selectedTopic.word_weights} layout="vertical" margin={{ left: 15, right: 20 }}>
                    <XAxis type="number" tick={{ fontSize: 10 }} />
                    <YAxis dataKey="word" type="category" width={80} tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#f8fafc', fontSize: '12px' }}
                      itemStyle={{ color: '#f8fafc' }}
                      formatter={(val: any) => [typeof val === 'number' ? val.toFixed(4) : val, 'Topic Weight']}
                    />
                    <Bar dataKey="weight" fill="#0284c7" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Representative Quotes */}
            <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm space-y-4">
              <div className="flex items-center space-x-2">
                <MessageSquareQuote className="w-5 h-5 text-sky-600" />
                <h4 className="text-sm font-bold text-slate-800">Representative Student Evaluations</h4>
              </div>
              <div className="space-y-3">
                {selectedTopic.representative_samples.length > 0 ? (
                  selectedTopic.representative_samples.map((sample, idx) => (
                    <div key={idx} className="p-3.5 bg-slate-50 border border-slate-200/70 rounded-lg space-y-2">
                      <p className="text-xs text-slate-700 italic">"{sample.text}"</p>
                      <div className="flex items-center justify-between text-[11px]">
                        <Badge variant={sample.sentiment as any}>{sample.sentiment}</Badge>
                        <span className="text-slate-400 font-mono">Confidence: {(sample.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-xs text-slate-400">No quotes available for this topic.</p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
