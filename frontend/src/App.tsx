import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/layout/Layout';
import { DashboardPage } from './pages/DashboardPage';
import { AnalyzeFeedbackPage } from './pages/AnalyzeFeedbackPage';
import { BulkAnalysisPage } from './pages/BulkAnalysisPage';
import { TopicExplorerPage } from './pages/TopicExplorerPage';
import { SentimentExplorerPage } from './pages/SentimentExplorerPage';
import { ModelPerformancePage } from './pages/ModelPerformancePage';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<DashboardPage />} />
          <Route path="analyze" element={<AnalyzeFeedbackPage />} />
          <Route path="bulk" element={<BulkAnalysisPage />} />
          <Route path="topics" element={<TopicExplorerPage />} />
          <Route path="sentiment" element={<SentimentExplorerPage />} />
          <Route path="performance" element={<ModelPerformancePage />} />
          <Route path="*" element={<DashboardPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default App;
