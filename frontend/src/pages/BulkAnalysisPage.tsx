import React, { useState } from 'react';
import { validateDataset, analyzeBulkCsv, getExportUrl } from '../services/api';
import type { DatasetValidationResponse, BulkAnalysisResponse } from '../types';
import { Badge } from '../components/common/Badge';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import {
  UploadCloud,
  FileSpreadsheet,
  Download,
  CheckCircle2,
  AlertTriangle,
  Search,
  Filter,
} from 'lucide-react';

export const BulkAnalysisPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [validation, setValidation] = useState<DatasetValidationResponse | null>(null);
  const [selectedColumn, setSelectedColumn] = useState<string>('');
  const [validating, setValidating] = useState<boolean>(false);
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [result, setResult] = useState<BulkAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Table filter/search
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [sentimentFilter, setSentimentFilter] = useState<string>('all');

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const uploadedFile = e.target.files[0];
    setFile(uploadedFile);
    setResult(null);
    setError(null);
    setValidating(true);

    try {
      const valRes = await validateDataset(uploadedFile);
      setValidation(valRes);
      if (valRes.recommended_text_column) {
        setSelectedColumn(valRes.recommended_text_column);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'CSV validation failed');
    } finally {
      setValidating(false);
    }
  };

  const handleRunBulkAnalysis = async () => {
    if (!file) return;
    setAnalyzing(true);
    setError(null);

    try {
      const res = await analyzeBulkCsv(file, selectedColumn);
      setResult(res);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Bulk analysis failed');
    } finally {
      setAnalyzing(false);
    }
  };

  const filteredResults = result?.results.filter((item) => {
    const matchesSearch = item.text.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          item.topic_label.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesSentiment = sentimentFilter === 'all' || item.sentiment === sentimentFilter;
    return matchesSearch && matchesSentiment;
  }) || [];

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-slate-800">Bulk CSV Feedback Analysis</h2>
        <p className="text-sm text-slate-500 mt-1">
          Upload any student evaluation CSV to run batch topic modeling, sentiment classification, and download the enriched dataset.
        </p>
      </div>

      {/* Upload Box */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm space-y-5">
        <div className="border-2 border-dashed border-slate-300 rounded-xl p-8 text-center hover:border-sky-500 transition-colors">
          <UploadCloud className="w-12 h-12 text-slate-400 mx-auto mb-3" />
          <h4 className="text-sm font-semibold text-slate-700">Choose a CSV feedback file</h4>
          <p className="text-xs text-slate-500 mt-1 mb-4">Supports .csv datasets containing feedback columns</p>

          <label className="inline-flex items-center px-4 py-2 bg-sky-600 hover:bg-sky-700 text-white text-xs font-medium rounded-lg cursor-pointer transition-colors shadow-sm">
            <FileSpreadsheet className="w-4 h-4 mr-2" />
            <span>Browse CSV File</span>
            <input type="file" accept=".csv" className="hidden" onChange={handleFileChange} />
          </label>

          {file && (
            <p className="text-xs text-sky-700 font-medium mt-3">
              Selected File: <span className="font-bold">{file.name}</span> ({(file.size / 1024).toFixed(1)} KB)
            </p>
          )}
        </div>

        {validating && <LoadingSpinner message="Validating CSV schema and text columns..." />}

        {validation && validation.is_valid && (
          <div className="bg-slate-50 rounded-lg p-4 border border-slate-200 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-700 flex items-center space-x-1.5">
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                <span>Dataset Validated ({validation.total_rows} rows detected)</span>
              </span>
            </div>

            <div className="flex flex-col sm:flex-row sm:items-center gap-3 text-xs">
              <label className="font-semibold text-slate-700">Select Feedback / Text Column:</label>
              <select
                value={selectedColumn}
                onChange={(e) => {
                  setSelectedColumn(e.target.value);
                  setError(null);
                }}
                className="bg-white border border-slate-300 rounded-lg px-3 py-2 font-medium text-slate-800 outline-none focus:ring-2 focus:ring-sky-500 max-w-md"
              >
                {validation.detected_columns.map((col) => (
                  <option key={col} value={col}>
                    {col} {col === validation.recommended_text_column ? ' (★ Recommended Text Column)' : ''}
                  </option>
                ))}
              </select>
            </div>

            {selectedColumn && (
              ['timestamp', 'date', 'id', 'student_id', 'roll_number', 'roll_no', 'email', 'phone', 'consent', 'gender'].some((dis) =>
                selectedColumn.toLowerCase().includes(dis)
              ) ? (
                <div className="p-3 bg-amber-50 border border-amber-200 text-amber-800 rounded-lg text-xs flex items-center space-x-2">
                  <AlertTriangle className="w-4 h-4 shrink-0 text-amber-600" />
                  <span>Please select a feedback/text column containing natural-language responses.</span>
                </div>
              ) : null
            )}

            <div className="pt-2 flex justify-end">
              <button
                type="button"
                disabled={analyzing}
                onClick={handleRunBulkAnalysis}
                className="inline-flex items-center space-x-2 bg-sky-600 hover:bg-sky-700 text-white px-5 py-2 rounded-lg text-xs font-medium shadow-sm transition-colors disabled:bg-slate-300"
              >
                <span>{analyzing ? 'Processing NLP Pipeline...' : 'Process All Rows with NLP'}</span>
              </button>
            </div>
          </div>
        )}

        {error && (
          <div className="p-4 bg-rose-50 border border-rose-200 text-rose-700 rounded-lg text-xs flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {analyzing && <LoadingSpinner message="Running batch inference on all evaluations..." />}

      {/* Results Section */}
      {result && !analyzing && (
        <div className="space-y-6">
          {/* Summary Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-4">
              <p className="text-xs font-semibold text-emerald-700 uppercase">Positive Sentiment</p>
              <p className="text-2xl font-bold text-emerald-900 mt-1">{result.sentiment_breakdown.positive || 0}</p>
            </div>
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
              <p className="text-xs font-semibold text-amber-700 uppercase">Neutral Sentiment</p>
              <p className="text-2xl font-bold text-amber-900 mt-1">{result.sentiment_breakdown.neutral || 0}</p>
            </div>
            <div className="bg-rose-50 border border-rose-200 rounded-xl p-4">
              <p className="text-xs font-semibold text-rose-700 uppercase">Negative Sentiment</p>
              <p className="text-2xl font-bold text-rose-900 mt-1">{result.sentiment_breakdown.negative || 0}</p>
            </div>
          </div>

          {/* Table Card */}
          <div className="bg-white rounded-xl border border-slate-200/80 shadow-sm overflow-hidden space-y-4 p-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h3 className="text-base font-bold text-slate-800">Analyzed Dataset Results</h3>
                <p className="text-xs text-slate-500">Processed {result.total_processed} evaluations</p>
              </div>

              {/* Download CSV Action */}
              <a
                href={getExportUrl(result.download_token)}
                download
                className="inline-flex items-center space-x-2 bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-2 rounded-lg text-xs font-medium shadow-sm transition-colors"
              >
                <Download className="w-4 h-4" />
                <span>Download Analyzed CSV</span>
              </a>
            </div>

            {/* Filter Bar */}
            <div className="flex flex-col sm:flex-row gap-3 pt-2">
              <div className="relative flex-1">
                <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-400" />
                <input
                  type="text"
                  placeholder="Search reviews or topic names..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-9 pr-3 py-1.5 border border-slate-300 rounded-lg text-xs outline-none focus:ring-2 focus:ring-sky-500"
                />
              </div>

              <div className="flex items-center space-x-2 text-xs">
                <Filter className="w-3.5 h-3.5 text-slate-400" />
                <select
                  value={sentimentFilter}
                  onChange={(e) => setSentimentFilter(e.target.value)}
                  className="border border-slate-300 rounded-lg px-2.5 py-1.5 bg-white text-xs text-slate-700 outline-none"
                >
                  <option value="all">All Sentiments</option>
                  <option value="positive">Positive Only</option>
                  <option value="neutral">Neutral Only</option>
                  <option value="negative">Negative Only</option>
                </select>
              </div>
            </div>

            {/* Table */}
            <div className="overflow-x-auto border border-slate-200 rounded-lg">
              <table className="min-w-full divide-y divide-slate-200 text-xs text-left">
                <thead className="bg-slate-50 text-slate-600 font-semibold uppercase tracking-wider">
                  <tr>
                    <th className="px-3.5 py-2.5 w-12">#</th>
                    <th className="px-3.5 py-2.5">Feedback Text</th>
                    <th className="px-3.5 py-2.5 w-28">Sentiment</th>
                    <th className="px-3.5 py-2.5 w-48">Dominant Topic</th>
                    <th className="px-3.5 py-2.5 w-24">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 bg-white">
                  {filteredResults.length > 0 ? (
                    filteredResults.slice(0, 50).map((row) => (
                      <tr key={row.id} className="hover:bg-slate-50">
                        <td className="px-3.5 py-2.5 font-mono text-slate-400">{row.id}</td>
                        <td className="px-3.5 py-2.5 text-slate-800 font-medium max-w-md truncate" title={row.text}>
                          {row.text}
                        </td>
                        <td className="px-3.5 py-2.5">
                          <Badge variant={row.sentiment as any}>{row.sentiment}</Badge>
                        </td>
                        <td className="px-3.5 py-2.5 text-slate-700 font-medium truncate">
                          {row.topic_label}
                        </td>
                        <td className="px-3.5 py-2.5 font-mono text-slate-500">
                          {(row.sentiment_confidence * 100).toFixed(0)}%
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={5} className="px-4 py-8 text-center text-slate-400">
                        No rows matching the filter criteria.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            {filteredResults.length > 50 && (
              <p className="text-xs text-slate-400 text-right">Showing first 50 rows in preview.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
