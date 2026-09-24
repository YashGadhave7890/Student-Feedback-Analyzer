import axios from 'axios';
import type {
  DashboardSummaryResponse,
  SingleAnalysisResponse,
  BulkAnalysisResponse,
  TopicsListResponse,
  SentimentSummaryResponse,
  ModelPerformanceResponse,
  DatasetValidationResponse,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const fetchDashboardSummary = async (): Promise<DashboardSummaryResponse> => {
  const res = await api.get<DashboardSummaryResponse>('/dashboard');
  return res.data;
};

export const analyzeSingleFeedback = async (text: string): Promise<SingleAnalysisResponse> => {
  const res = await api.post<SingleAnalysisResponse>('/analyze', { text });
  return res.data;
};

export const validateDataset = async (file: File): Promise<DatasetValidationResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await api.post<DatasetValidationResponse>('/analyze/validate', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export const analyzeBulkCsv = async (file: File, textColumn?: string): Promise<BulkAnalysisResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  if (textColumn) {
    formData.append('text_column', textColumn);
  }
  const res = await api.post<BulkAnalysisResponse>('/analyze/batch', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export const getExportUrl = (downloadToken: string): string => {
  return `${API_BASE_URL}/analyze/export/${downloadToken}`;
};

export const fetchTopics = async (): Promise<TopicsListResponse> => {
  const res = await api.get<TopicsListResponse>('/topics');
  return res.data;
};

export const fetchSentimentSummary = async (): Promise<SentimentSummaryResponse> => {
  const res = await api.get<SentimentSummaryResponse>('/sentiment');
  return res.data;
};

export const fetchModelPerformance = async (): Promise<ModelPerformanceResponse> => {
  const res = await api.get<ModelPerformanceResponse>('/models');
  return res.data;
};
