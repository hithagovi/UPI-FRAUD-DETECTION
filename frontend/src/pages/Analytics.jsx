import { useState, useEffect } from 'react';
import { axiosInstance } from '@/App';
import { TrendingUp, DollarSign, BarChart3, Users } from 'lucide-react';
import { toast } from 'sonner';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const Analytics = () => {
  const [analytics, setAnalytics] = useState(null);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAnalytics();
  }, []);

  const fetchAnalytics = async () => {
    setLoading(true);
    try {
      const [analyticsRes, modelRes] = await Promise.all([
        axiosInstance.get('/analytics/metrics'),
        axiosInstance.get('/models/active').catch(() => null),
      ]);
      setAnalytics(analyticsRes.data);
      if (modelRes) {
        setModelMetrics(modelRes.data);
      }
    } catch (error) {
      toast.error('Failed to fetch analytics');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  const COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'];

  const metricCards = [
    {
      title: 'Avg Transaction Value',
      value: `$${analytics?.avg_transaction_value?.toFixed(2) || 0}`,
      icon: DollarSign,
      color: 'from-blue-500 to-blue-600',
      testId: 'avg-transaction-value'
    },
    {
      title: 'Fraud Detection Rate',
      value: `${analytics?.fraud_detection_rate || 0}%`,
      icon: TrendingUp,
      color: 'from-purple-500 to-purple-600',
      testId: 'fraud-detection-rate'
    },
    {
      title: 'Avg Fraud Amount',
      value: `$${analytics?.avg_fraud_amount?.toFixed(2) || 0}`,
      icon: BarChart3,
      color: 'from-red-500 to-red-600',
      testId: 'avg-fraud-amount'
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-slate-800">Analytics</h1>
        <p className="text-slate-600 mt-1">Detailed insights and performance metrics</p>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {metricCards.map((metric, index) => {
          const Icon = metric.icon;
          return (
            <div
              key={index}
              data-testid={metric.testId}
              className="bg-white rounded-xl p-6 shadow-lg border border-slate-200 metric-card"
            >
              <div className="flex items-center justify-between mb-4">
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${metric.color} flex items-center justify-center`}>
                  <Icon className="text-white" size={24} />
                </div>
              </div>
              <h3 className="text-sm font-medium text-slate-600 mb-1">{metric.title}</h3>
              <p className="text-3xl font-bold text-slate-800">{metric.value}</p>
            </div>
          );
        })}
      </div>

      {/* Model Performance Metrics */}
      {modelMetrics && (
        <div className="bg-white rounded-xl p-6 shadow-lg border border-slate-200">
          <h2 className="text-xl font-bold text-slate-800 mb-4">Model Performance</h2>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <p className="text-sm text-slate-600 mb-1">Accuracy</p>
              <p className="text-2xl font-bold text-blue-600" data-testid="model-accuracy">
                {(modelMetrics.metrics.accuracy * 100).toFixed(2)}%
              </p>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <p className="text-sm text-slate-600 mb-1">Precision</p>
              <p className="text-2xl font-bold text-purple-600" data-testid="model-precision">
                {(modelMetrics.metrics.precision * 100).toFixed(2)}%
              </p>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <p className="text-sm text-slate-600 mb-1">Recall</p>
              <p className="text-2xl font-bold text-green-600" data-testid="model-recall">
                {(modelMetrics.metrics.recall * 100).toFixed(2)}%
              </p>
            </div>
            <div className="text-center p-4 bg-amber-50 rounded-lg">
              <p className="text-sm text-slate-600 mb-1">F1 Score</p>
              <p className="text-2xl font-bold text-amber-600" data-testid="model-f1-score">
                {(modelMetrics.metrics.f1_score * 100).toFixed(2)}%
              </p>
            </div>
            <div className="text-center p-4 bg-pink-50 rounded-lg">
              <p className="text-sm text-slate-600 mb-1">ROC AUC</p>
              <p className="text-2xl font-bold text-pink-600" data-testid="model-roc-auc">
                {(modelMetrics.metrics.roc_auc * 100).toFixed(2)}%
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top Senders Chart */}
        <div className="bg-white rounded-xl p-6 shadow-lg border border-slate-200">
          <h2 className="text-xl font-bold text-slate-800 mb-4">Top Senders by Volume</h2>
          {analytics?.top_senders?.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={analytics.top_senders}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="sender" stroke="#64748b" fontSize={12} />
                <YAxis stroke="#64748b" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'white',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                  }}
                />
                <Bar dataKey="count" fill="#3b82f6" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-64 flex items-center justify-center text-slate-500">
              No data available
            </div>
          )}
        </div>

        {/* Amount Distribution Chart */}
        <div className="bg-white rounded-xl p-6 shadow-lg border border-slate-200">
          <h2 className="text-xl font-bold text-slate-800 mb-4">Transaction Amount Distribution</h2>
          {analytics?.amount_distribution?.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={analytics.amount_distribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="range" stroke="#64748b" fontSize={12} />
                <YAxis stroke="#64748b" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'white',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                  }}
                />
                <Bar dataKey="count" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-64 flex items-center justify-center text-slate-500">
              No data available
            </div>
          )}
        </div>
      </div>

      {/* Feature Importance */}
      {modelMetrics?.feature_importance && (
        <div className="bg-white rounded-xl p-6 shadow-lg border border-slate-200">
          <h2 className="text-xl font-bold text-slate-800 mb-4">Feature Importance (Top 10)</h2>
          <div className="space-y-3" data-testid="feature-importance-list">
            {Object.entries(modelMetrics.feature_importance)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 10)
              .map(([feature, importance], index) => (
                <div key={feature} className="flex items-center gap-4">
                  <div className="w-32 text-sm font-medium text-slate-700 truncate" title={feature}>
                    {feature}
                  </div>
                  <div className="flex-1">
                    <div className="bg-slate-200 rounded-full h-6 relative">
                      <div
                        className="bg-gradient-to-r from-blue-500 to-purple-600 h-6 rounded-full flex items-center justify-end pr-2"
                        style={{ width: `${(importance / Math.max(...Object.values(modelMetrics.feature_importance))) * 100}%` }}
                      >
                        <span className="text-xs font-medium text-white">
                          {importance.toFixed(4)}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Analytics;
