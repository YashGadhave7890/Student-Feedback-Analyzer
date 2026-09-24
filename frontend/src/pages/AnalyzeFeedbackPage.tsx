import React, { useState } from 'react';
import { analyzeSingleFeedback } from '../services/api';
import type { SingleAnalysisResponse } from '../types';
import { Badge } from '../components/common/Badge';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { Sparkles, Tag, Layers, CheckCircle2, AlertCircle } from 'lucide-react';

const SAMPLE_INPUTS = [
  {
    label: 'Positive Sample',
    text: 'The practical lab sessions with Docker and Power BI were very helpful and the lecturer explained complex concepts clearly.',
    type: 'positive',
  },
  {
    label: 'Neutral / Balanced Sample',
    text: 'The course content was fine and logically arranged, but the pace moved a bit too fast in some lecture modules.',
    type: 'neutral',
  },
  {
    label: 'Negative / Improvement Sample',
    text: 'The instructions during the lab work were very confusing and the lecture slides were too long to revise for exams.',
    type: 'negative',
  },
];

export const AnalyzeFeedbackPage: React.FC = () => {
  const [text, setText] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<SingleAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async (inputText?: string) => {
    const textToAnalyze = inputText !== undefined ? inputText : text;
    if (!textToAnalyze.trim()) {
      setError('Please provide feedback text before analyzing.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await analyzeSingleFeedback(textToAnalyze);
      setResult(res);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectSample = (sampleText: string) => {
    setText(sampleText);
    handleAnalyze(sampleText);
  };

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-slate-800">Analyze Student Feedback</h2>
        <p className="text-sm text-slate-500 mt-1">
          Perform live NLP inference: classify sentiment, assign LDA theme, and inspect token-level attributions.
        </p>
      </div>

      {/* Input Section */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm space-y-4">
        <div>
          <label className="block text-sm font-semibold text-slate-700 mb-2">
            Enter Student Feedback Text
          </label>
          <textarea
            rows={4}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type or paste student course evaluation comment here (e.g., 'The lecturer was very engaging and the lab assignments were well structured...')"
            className="w-full p-3.5 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-sky-500 focus:border-sky-500 outline-none transition-all placeholder:text-slate-400"
          />
        </div>

        {/* Preset Sample Buttons */}
        <div>
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
            Try Sample Evaluations:
          </p>
          <div className="flex flex-wrap gap-2.5">
            {SAMPLE_INPUTS.map((sample, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handleSelectSample(sample.text)}
                className={`text-xs font-medium px-3 py-1.5 rounded-lg border transition-colors ${
                  sample.type === 'positive'
                    ? 'bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-100'
                    : sample.type === 'neutral'
                    ? 'bg-amber-50 text-amber-700 border-amber-200 hover:bg-amber-100'
                    : 'bg-rose-50 text-rose-700 border-rose-200 hover:bg-rose-100'
                }`}
              >
                {sample.label}
              </button>
            ))}
          </div>
        </div>

        <div className="pt-2 flex justify-between items-center">
          <button
            type="button"
            onClick={() => setText('')}
            className="text-xs text-slate-500 hover:text-slate-700 underline"
          >
            Clear Text
          </button>
          <button
            type="button"
            disabled={loading || !text.trim()}
            onClick={() => handleAnalyze()}
            className="inline-flex items-center space-x-2 bg-sky-600 hover:bg-sky-700 disabled:bg-slate-300 text-white px-5 py-2.5 rounded-lg text-sm font-medium shadow-sm transition-colors"
          >
            <Sparkles className="w-4 h-4" />
            <span>{loading ? 'Analyzing...' : 'Run NLP Analysis'}</span>
          </button>
        </div>

        {error && (
          <div className="p-3 bg-rose-50 border border-rose-200 text-rose-700 rounded-lg text-xs flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {loading && <LoadingSpinner message="Running TF-IDF & LDA Vectorizers..." />}

      {/* Analysis Results */}
      {result && !loading && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-bold text-slate-800 flex items-center space-x-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-600" />
              <span>Inference Results</span>
            </h3>
            <span className="text-xs text-slate-400">Processed in real-time</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Sentiment Card */}
            <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="p-2 bg-slate-100 rounded-lg text-slate-600">
                    <Tag className="w-4 h-4" />
                  </span>
                  <h4 className="text-sm font-bold text-slate-800">Predicted Sentiment</h4>
                </div>
                <Badge variant={result.sentiment as any} className="text-sm px-3 py-1 font-semibold uppercase">
                  {result.sentiment === 'positive' ? '😊 Positive' : result.sentiment === 'neutral' ? '😐 Neutral' : '😞 Negative'}
                </Badge>
              </div>

              <div>
                <div className="flex justify-between text-xs text-slate-600 mb-1">
                  <span>Confidence Score</span>
                  <span className="font-bold">{(result.sentiment_confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      result.sentiment === 'positive'
                        ? 'bg-emerald-500'
                        : result.sentiment === 'neutral'
                        ? 'bg-amber-500'
                        : 'bg-rose-500'
                    }`}
                    style={{ width: `${Math.min(100, result.sentiment_confidence * 100)}%` }}
                  ></div>
                </div>
              </div>

              <div className="pt-2 border-t border-slate-100">
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                  Class Probability Distribution:
                </p>
                <div className="space-y-2">
                  {Object.entries(result.sentiment_probabilities).map(([cls, prob]) => (
                    <div key={cls} className="flex items-center justify-between text-xs">
                      <span className="capitalize font-medium text-slate-700">{cls}</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-24 bg-slate-100 h-2 rounded-full overflow-hidden">
                          <div
                            className={`h-full ${
                              cls === 'positive'
                                ? 'bg-emerald-500'
                                : cls === 'neutral'
                                ? 'bg-amber-500'
                                : 'bg-rose-500'
                            }`}
                            style={{ width: `${prob * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-slate-500 w-10 text-right">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Topic Card */}
            <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="p-2 bg-sky-50 rounded-lg text-sky-600">
                    <Layers className="w-4 h-4" />
                  </span>
                  <h4 className="text-sm font-bold text-slate-800">Dominant Topic Theme</h4>
                </div>
                <Badge variant="info" className="text-xs font-semibold">
                  Topic #{result.dominant_topic + 1}
                </Badge>
              </div>

              <div>
                <p className="text-base font-bold text-sky-900">{result.topic_label}</p>
                <p className="text-xs text-slate-500 mt-0.5">
                  LDA Confidence: <span className="font-semibold text-slate-700">{(result.topic_confidence * 100).toFixed(1)}%</span>
                </p>
              </div>

              <div className="pt-2 border-t border-slate-100">
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                  Key Associated Words for Topic:
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {result.top_topic_keywords.map((kw, i) => (
                    <span
                      key={i}
                      className="px-2 py-0.5 bg-slate-100 text-slate-700 rounded text-xs font-mono"
                    >
                      {kw}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Token-Level Attribution / Explainability */}
          <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm space-y-3">
            <h4 className="text-sm font-bold text-slate-800">
              NLP Token Attribution & Explainability
            </h4>
            <p className="text-xs text-slate-500">
              Extracted tokens color-coded by sentiment weight calculated by the TF-IDF feature vocabulary:
            </p>

            <div className="p-4 bg-slate-50 rounded-lg border border-slate-200 flex flex-wrap gap-2 items-center">
              {result.token_attributions.length > 0 ? (
                result.token_attributions.map((token, idx) => (
                  <span
                    key={idx}
                    className={`px-2 py-1 rounded text-xs font-medium border ${
                      token.sentiment_tag === 'positive'
                        ? 'bg-emerald-100 text-emerald-800 border-emerald-300'
                        : token.sentiment_tag === 'negative'
                        ? 'bg-rose-100 text-rose-800 border-rose-300'
                        : 'bg-white text-slate-700 border-slate-200'
                    }`}
                    title={`Feature weight: ${token.weight}`}
                  >
                    {token.word}
                    <span className="text-[10px] opacity-70 ml-1">({token.weight > 0 ? `+${token.weight}` : token.weight})</span>
                  </span>
                ))
              ) : (
                <span className="text-xs text-slate-400 italic">No significant tokens detected.</span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
