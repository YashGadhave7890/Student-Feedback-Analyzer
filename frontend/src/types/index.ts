export interface TokenWeight {
  word: string;
  weight: number;
  sentiment_tag: 'positive' | 'neutral' | 'negative' | 'neutral_feature';
}

export interface SingleAnalysisResponse {
  original_text: string;
  clean_sentiment_text: string;
  clean_topic_text: string;
  sentiment: 'positive' | 'neutral' | 'negative';
  sentiment_confidence: number;
  sentiment_probabilities: Record<string, number>;
  dominant_topic: number;
  topic_label: string;
  topic_confidence: number;
  topic_probabilities: Record<string, number>;
  top_topic_keywords: string[];
  token_attributions: TokenWeight[];
}

export interface BulkItemResult {
  id: number;
  text: string;
  sentiment: 'positive' | 'neutral' | 'negative';
  sentiment_confidence: number;
  dominant_topic: number;
  topic_label: string;
  topic_confidence: number;
}

export interface BulkAnalysisResponse {
  total_processed: number;
  sentiment_breakdown: Record<string, number>;
  topic_breakdown: Record<string, number>;
  results: BulkItemResult[];
  download_token: string;
}

export interface DashboardSummaryResponse {
  total_feedbacks: number;
  sentiment_distribution: Record<string, number>;
  sentiment_percentages: Record<string, number>;
  topic_distribution: Record<string, number>;
  topic_percentages: Record<string, number>;
  topic_sentiment_matrix: Array<{
    topic: string;
    positive: number;
    neutral: number;
    negative: number;
    total: number;
  }>;
  key_metrics: {
    avg_confidence: number;
    top_positive_topic: string;
    top_improvement_area: string;
    top_positive_insight?: string;
    top_improvement_insight?: string;
    positive_rate: number;
    negative_rate: number;
  };
}

export interface TopicDetail {
  topic_id: number;
  label: string;
  total_feedbacks: number;
  percentage: number;
  top_words: string[];
  word_weights: Array<{ word: string; weight: number }>;
  sentiment_breakdown: Record<string, number>;
  representative_samples: Array<{
    text: string;
    sentiment: string;
    confidence: number;
  }>;
}

export interface TopicsListResponse {
  topics: TopicDetail[];
}

export interface SentimentSummaryResponse {
  total_analyzed: number;
  distribution: Record<string, number>;
  percentages: Record<string, number>;
  top_features: {
    positive: Array<{ term: string; coefficient: number }>;
    negative: Array<{ term: string; coefficient: number }>;
  };
}

export interface ModelBenchmarkItem {
  Accuracy: number;
  Accuracy_Std: number;
  Precision: number;
  Recall: number;
  F1_Score: number;
}

export interface ModelPerformanceResponse {
  models_benchmark: Record<string, ModelBenchmarkItem>;
  best_model: string;
  confusion_matrix: {
    labels: string[];
    matrix: number[][];
  };
  topic_metrics: {
    n_topics: number;
    perplexity: number;
    coherence_cv_score: number;
  };
  feature_importance: Record<string, Array<{ term: string; coefficient: number }>>;
}

export interface DatasetValidationResponse {
  is_valid: boolean;
  total_rows: number;
  detected_columns: string[];
  recommended_text_column?: string;
  missing_values_count: number;
  preview: Array<Record<string, any>>;
  errors: string[];
}
