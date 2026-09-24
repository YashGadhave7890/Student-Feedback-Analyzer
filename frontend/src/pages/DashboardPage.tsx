import React, { useEffect, useState } from 'react';
import { fetchDashboardSummary, fetchTopics } from '../services/api';
import type { DashboardSummaryResponse, TopicsListResponse } from '../types';
import { StatCard } from '../components/common/StatCard';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import {
  MessageSquare,
  Smile,
  Frown,
  Meh,
  Award,
  AlertTriangle,
  ArrowRight,
  Sparkles,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { Link } from 'react-router-dom';

const SENTIMENT_COLORS = {
  positive: '#10b981', // emerald-500
  neutral: '#f59e0b',  // amber-500
  negative: '#ef4444', // rose-500
};

export const DashboardPage: React.FC = () => {
  const [data, setData] = useState<DashboardSummaryResponse | null>(null);
  const [topicsData, setTopicsData] = useState<TopicsListResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchDashboardSummary(), fetchTopics()])
      .then(([dashRes, topRes]) => {
        setData(dashRes);
        setTopicsData(topRes);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || 'Failed to load dashboard data');
        setLoading(false);
      });
  }, []);

  if (loading) return <LoadingSpinner message="Loading Real-Time NLP Intelligence Summary..." />;
  if (error || !data) {
    return (
      <div className="bg-rose-50 border border-rose-200 text-rose-700 p-6 rounded-xl">
        <h3 className="font-bold">Error Loading Dashboard</h3>
        <p className="text-sm mt-1">{error || 'Unknown error occurred.'}</p>
      </div>
    );
  }

  const sentimentPieData = [
    { name: 'Positive', value: data.sentiment_distribution.positive || 0, color: SENTIMENT_COLORS.positive },
    { name: 'Neutral', value: data.sentiment_distribution.neutral || 0, color: SENTIMENT_COLORS.neutral },
    { name: 'Negative', value: data.sentiment_distribution.negative || 0, color: SENTIMENT_COLORS.negative },
  ];

  const topicBarData = Object.entries(data.topic_distribution).map(([topic, count]) => ({
    topic: topic.length > 25 ? topic.slice(0, 22) + '...' : topic,
    fullTopic: topic,
    count,
  }));

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-800">Student Feedback Intelligence Dashboard</h2>
          <p className="text-sm text-slate-500 mt-0.5">
            Real-time NLP analytics on student evaluations, sentiment classification & unsupervised LDA topic themes.
          </p>
        </div>
        <div className="flex gap-3">
          <Link
            to="/analyze"
            className="inline-flex items-center space-x-2 bg-sky-600 hover:bg-sky-700 text-white px-4 py-2 rounded-lg text-sm font-medium shadow-sm transition-colors"
          >
            <Sparkles className="w-4 h-4" />
            <span>Analyze Single Feedback</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <StatCard
          title="Total Feedback Analyzed"
          value={data.total_feedbacks}
          subtitle="Course evaluation dataset"
          icon={<MessageSquare className="w-6 h-6" />}
        />
        <StatCard
          title="Positive Sentiment"
          value={`${data.sentiment_percentages.positive || 0}%`}
          subtitle={`${data.sentiment_distribution.positive || 0} positive evaluations`}
          trend={`${data.sentiment_percentages.positive}%`}
          trendType="positive"
          icon={<Smile className="w-6 h-6 text-emerald-600" />}
        />
        <StatCard
          title="Neutral Feedback"
          value={`${data.sentiment_percentages.neutral || 0}%`}
          subtitle={`${data.sentiment_distribution.neutral || 0} balanced suggestions`}
          trend={`${data.sentiment_percentages.neutral}%`}
          trendType="neutral"
          icon={<Meh className="w-6 h-6 text-amber-600" />}
        />
        <StatCard
          title="Negative Feedback"
          value={`${data.sentiment_percentages.negative || 0}%`}
          subtitle={`${data.sentiment_distribution.negative || 0} improvement areas`}
          trend={`${data.sentiment_percentages.negative}%`}
          trendType="negative"
          icon={<Frown className="w-6 h-6 text-rose-600" />}
        />
      </div>

      {/* Actionable Insights */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="bg-emerald-50/70 border border-emerald-200/80 rounded-xl p-5 flex items-start space-x-4">
          <div className="p-2.5 bg-emerald-100 text-emerald-700 rounded-lg shrink-0">
            <Award className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-sm font-semibold text-emerald-900">Highest Rated Theme (Course Strength)</h4>
            <p className="text-base font-bold text-emerald-800 mt-0.5">{data.key_metrics.top_positive_topic}</p>
            <p className="text-xs text-emerald-700 mt-1">
              {data.key_metrics.top_positive_insight || 'Highest positive feedback theme in the evaluation dataset.'}
            </p>
          </div>
        </div>

        <div className="bg-rose-50/70 border border-rose-200/80 rounded-xl p-5 flex items-start space-x-4">
          <div className="p-2.5 bg-rose-100 text-rose-700 rounded-lg shrink-0">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-sm font-semibold text-rose-900">Primary Focus Area (Course Improvement)</h4>
            <p className="text-base font-bold text-rose-800 mt-0.5">{data.key_metrics.top_improvement_area}</p>
            <p className="text-xs text-rose-700 mt-1">
              {data.key_metrics.top_improvement_insight || 'Topic with the highest proportion of constructive suggestions and critique.'}
            </p>
          </div>
        </div>
      </div>

      {/* Charts: Sentiment Distribution & Topic Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sentiment Pie */}
        <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-bold text-slate-800">Sentiment Distribution</h3>
            <span className="text-xs font-medium text-slate-400">Classified by Multi-Class NLP</span>
          </div>
          <div className="h-64 flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={sentimentPieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={85}
                  paddingAngle={4}
                  dataKey="value"
                  label={({ name, percent }) => `${name}: ${((percent || 0) * 100).toFixed(0)}%`}
                >
                  {sentimentPieData.map((entry, index) => (
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
          <div className="flex flex-wrap justify-center gap-4 sm:gap-6 mt-2 text-xs font-medium text-slate-600">
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-full bg-emerald-500"></span>
              <span>Positive ({data.sentiment_percentages.positive}%)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-full bg-amber-500"></span>
              <span>Neutral ({data.sentiment_percentages.neutral}%)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-full bg-rose-500"></span>
              <span>Negative ({data.sentiment_percentages.negative}%)</span>
            </div>
          </div>
        </div>

        {/* Topic Breakdown Bar */}
        <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-bold text-slate-800">Discovered Topic Distribution</h3>
            <span className="text-xs font-medium text-slate-400">Unsupervised LDA ($K=5$)</span>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topicBarData} layout="vertical" margin={{ left: 10, right: 20, top: 10, bottom: 10 }}>
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="topic" type="category" width={140} tick={{ fontSize: 11 }} />
                <Tooltip
                  formatter={(value: any, _name: any, props: any) => [value, props.payload.fullTopic]}
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#f8fafc', fontSize: '12px' }}
                  itemStyle={{ color: '#f8fafc' }}
                />
                <Bar dataKey="count" fill="#0284c7" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Topic vs Sentiment Cross-Tabulation Chart */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-bold text-slate-800">Topic vs Sentiment Cross-Analysis</h3>
            <p className="text-xs text-slate-500">Distribution of Positive, Neutral, and Negative sentiments for each course theme</p>
          </div>
        </div>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.topic_sentiment_matrix} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
              <XAxis
                dataKey="topic"
                interval={0}
                angle={-18}
                textAnchor="end"
                tick={{ fontSize: 11 }}
                height={60}
              />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#f8fafc', fontSize: '12px' }}
                itemStyle={{ color: '#f8fafc' }}
              />
              <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: '10px' }} />
              <Bar dataKey="positive" name="Positive" stackId="a" fill={SENTIMENT_COLORS.positive} />
              <Bar dataKey="neutral" name="Neutral" stackId="a" fill={SENTIMENT_COLORS.neutral} />
              <Bar dataKey="negative" name="Negative" stackId="a" fill={SENTIMENT_COLORS.negative} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Top Discovered Keywords per Topic */}
      {topicsData && topicsData.topics && (
        <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-bold text-slate-800">Discovered Topic Keywords</h3>
              <p className="text-xs text-slate-500">Top high-probability vocabulary terms extracted by LDA for each theme</p>
            </div>
            <Link to="/topics" className="text-xs text-sky-600 font-semibold hover:underline">
              Explore All Topics →
            </Link>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 pt-2">
            {topicsData.topics.map((t) => (
              <div key={t.topic_id} className="p-4 bg-slate-50 border border-slate-200 rounded-lg space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-sky-700 bg-sky-100/70 px-2 py-0.5 rounded">
                    Topic #{t.topic_id + 1}
                  </span>
                  <span className="text-[11px] text-slate-500">{t.percentage}%</span>
                </div>
                <h4 className="text-xs font-bold text-slate-800 truncate">{t.label}</h4>
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {t.top_words.slice(0, 6).map((kw, idx) => (
                    <span key={idx} className="px-2 py-0.5 bg-white text-slate-600 border border-slate-200 rounded text-[11px] font-mono">
                      {kw}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
