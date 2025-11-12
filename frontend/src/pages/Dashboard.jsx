import { useState, useEffect } from 'react';
import { axiosInstance } from '@/App';
import { AlertTriangle, Shield, Activity, TrendingUp, Upload, CheckCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { useDropzone } from 'react-dropzone';

const Dashboard = ({ user }) => {
  const [metrics, setMetrics] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [training, setTraining] = useState(false);
  const [uploadedDataset, setUploadedDataset] = useState(null);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const [metricsRes, alertsRes] = await Promise.all([
        axiosInstance.get('/dashboard/metrics'),
        axiosInstance.get('/alerts?limit=5'),
      ]);
      setMetrics(metricsRes.data);
      setAlerts(alertsRes.data);
    } catch (error) {
      toast.error('Failed to fetch dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const onDrop = async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('fraud_column', 'is_fraud');

    try {
      const response = await axiosInstance.post('/datasets/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setUploadedDataset(response.data);
      toast.success('Dataset uploaded successfully!');
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'], 'application/parquet': ['.parquet'] },
    maxFiles: 1,
  });

  const handleTrainModel = async () => {
    if (!uploadedDataset) return;

    setTraining(true);
    try {
      await axiosInstance.post(`/datasets/${uploadedDataset.id}/train?model_type=xgboost`);
      toast.success('Model trained successfully!');
      setUploadedDataset(null);
      fetchDashboardData();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Training failed');
    } finally {
      setTraining(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  const metricCards = [
    {
      title: 'Total Transactions',
      value: metrics?.total_transactions || 0,
      icon: Activity,
      color: 'from-blue-500 to-blue-600',
      testId: 'total-transactions-metric'
    },
    {
      title: 'Fraudulent Count',
      value: metrics?.fraudulent_count || 0,
      icon: AlertTriangle,
      color: 'from-red-500 to-red-600',
      testId: 'fraudulent-count-metric'
    },
    {
      title: 'Active Alerts',
      value: metrics?.active_alerts || 0,
      icon: Shield,
      color: 'from-amber-500 to-amber-600',
      testId: 'active-alerts-metric'
    },
    {
      title: 'Blocked Entities',
      value: metrics?.blocked_entities || 0,
      icon: Shield,
      color: 'from-purple-500 to-purple-600',
      testId: 'blocked-entities-metric'
    },
    {
      title: 'Fraud Detection Rate',
      value: `${metrics?.fraud_detection_rate || 0}%`,
      icon: TrendingUp,
      color: 'from-green-500 to-green-600',
      testId: 'fraud-detection-rate-metric'
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-800">Dashboard</h1>
          <p className="text-slate-600 mt-1">Welcome back, {user?.name}</p>
        </div>
      </div>

      {/* Upload Dataset Section */}
      <div className="bg-white rounded-xl p-6 shadow-lg border border-slate-200">
        <h2 className="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2">
          <Upload size={24} className="text-blue-600" />
          Dataset Management
        </h2>
        <div
          {...getRootProps()}
          data-testid="dataset-upload-dropzone"
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
            isDragActive
              ? 'border-blue-500 bg-blue-50'
              : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50'
          }`}
        >
          <input {...getInputProps()} />
          <Upload className="mx-auto mb-4 text-slate-400" size={48} />
          <p className="text-slate-600 font-medium">
            {uploading
              ? 'Uploading...'
              : isDragActive
              ? 'Drop the file here'
              : 'Drag & drop a CSV or Parquet file, or click to select'}
          </p>
          <p className="text-sm text-slate-500 mt-2">Supported formats: CSV, Parquet</p>
        </div>

        {uploadedDataset && (
          <div className="mt-4 p-4 bg-green-50 rounded-lg border border-green-200">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <CheckCircle className="text-green-600" size={24} />
                <div>
                  <p className="font-medium text-green-800">{uploadedDataset.filename}</p>
                  <p className="text-sm text-green-600">
                    {uploadedDataset.rows} rows, {uploadedDataset.columns.length} columns
                  </p>
                </div>
              </div>
              <Button
                data-testid="train-model-button"
                onClick={handleTrainModel}
                disabled={training}
                className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white"
              >
                {training ? 'Training...' : 'Train Model'}
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
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

      {/* Recent Alerts */}
      <div className="bg-white rounded-xl p-6 shadow-lg border border-slate-200">
        <h2 className="text-xl font-bold text-slate-800 mb-4">Recent Alerts</h2>
        {alerts.length === 0 ? (
          <div className="text-center py-8 text-slate-500">
            <Shield className="mx-auto mb-2 text-slate-300" size={48} />
            <p>No alerts yet. Upload a dataset and start detecting fraud!</p>
          </div>
        ) : (
          <div className="space-y-3" data-testid="recent-alerts-list">
            {alerts.map((alert) => (
              <div
                key={alert.id}
                data-testid={`alert-item-${alert.id}`}
                className="flex items-start gap-4 p-4 rounded-lg bg-gradient-to-r from-red-50 to-orange-50 border border-red-200"
              >
                <AlertTriangle className="text-red-600 flex-shrink-0 mt-1" size={20} />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span
                      className={`px-2 py-1 rounded-full text-xs font-medium ${
                        alert.severity === 'high'
                          ? 'bg-red-100 text-red-700'
                          : alert.severity === 'medium'
                          ? 'bg-amber-100 text-amber-700'
                          : 'bg-blue-100 text-blue-700'
                      }`}
                    >
                      {alert.severity.toUpperCase()}
                    </span>
                  </div>
                  <p className="text-slate-800 font-medium mt-1">{alert.message}</p>
                  <p className="text-sm text-slate-500 mt-1">
                    {new Date(alert.created_at).toLocaleString()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
