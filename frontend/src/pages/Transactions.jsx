import { useState, useEffect } from 'react';
import { axiosInstance } from '@/App';
import { Filter, Eye, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';

const Transactions = () => {
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [selectedTransaction, setSelectedTransaction] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);

  useEffect(() => {
    fetchTransactions();
  }, [filter]);

  const fetchTransactions = async () => {
    setLoading(true);
    try {
      const response = await axiosInstance.get(`/transactions?filter_by=${filter}&limit=100`);
      setTransactions(response.data);
    } catch (error) {
      toast.error('Failed to fetch transactions');
    } finally {
      setLoading(false);
    }
  };

  const viewDetails = (transaction) => {
    setSelectedTransaction(transaction);
    setDialogOpen(true);
  };

  const getStatusBadge = (prediction) => {
    if (prediction === 'Fraudulent') return 'status-badge-fraudulent';
    if (prediction === 'Suspicious') return 'status-badge-suspicious';
    return 'status-badge-safe';
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-800">Transactions</h1>
          <p className="text-slate-600 mt-1">Monitor and analyze all transactions</p>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="bg-white rounded-xl p-4 shadow-lg border border-slate-200">
        <div className="flex items-center gap-4 flex-wrap">
          <Filter className="text-slate-600" size={20} />
          <div className="flex gap-2">
            {['all', 'fraudulent', 'suspicious', 'safe'].map((status) => (
              <Button
                key={status}
                data-testid={`filter-${status}-button`}
                onClick={() => setFilter(status)}
                className={`capitalize ${
                  filter === status
                    ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white'
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}
              >
                {status}
              </Button>
            ))}
          </div>
        </div>
      </div>

      {/* Transactions Table */}
      <div className="bg-white rounded-xl shadow-lg border border-slate-200 overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          </div>
        ) : transactions.length === 0 ? (
          <div className="text-center py-16 px-4">
            <AlertCircle className="mx-auto mb-4 text-slate-300" size={64} />
            <p className="text-slate-600 text-lg">No transactions found</p>
            <p className="text-slate-500 text-sm mt-2">
              Upload a dataset and train a model to start detecting fraud
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full" data-testid="transactions-table">
              <thead className="bg-slate-50 border-b border-slate-200">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">
                    Transaction ID
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">
                    Fraud Probability
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">
                    Timestamp
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {transactions.map((transaction) => (
                  <tr
                    key={transaction.id}
                    data-testid={`transaction-row-${transaction.id}`}
                    className="hover:bg-slate-50 transition-colors"
                  >
                    <td className="px-6 py-4 text-sm font-mono text-slate-700">
                      {transaction.id.substring(0, 8)}...
                    </td>
                    <td className="px-6 py-4">
                      <span className={getStatusBadge(transaction.prediction)}>
                        {transaction.prediction}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-slate-200 rounded-full h-2 max-w-[100px]">
                          <div
                            className={`h-2 rounded-full ${
                              transaction.fraud_probability >= 0.7
                                ? 'bg-red-500'
                                : transaction.fraud_probability >= 0.4
                                ? 'bg-amber-500'
                                : 'bg-green-500'
                            }`}
                            style={{ width: `${transaction.fraud_probability * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm font-medium text-slate-700">
                          {(transaction.fraud_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-sm text-slate-600">
                      {new Date(transaction.timestamp).toLocaleString()}
                    </td>
                    <td className="px-6 py-4">
                      <Button
                        data-testid={`view-details-${transaction.id}`}
                        onClick={() => viewDetails(transaction)}
                        size="sm"
                        className="flex items-center gap-2 bg-blue-500 hover:bg-blue-600 text-white"
                      >
                        <Eye size={16} />
                        Details
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Transaction Details Dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="max-w-2xl" data-testid="transaction-details-dialog">
          <DialogHeader>
            <DialogTitle>Transaction Details</DialogTitle>
          </DialogHeader>
          {selectedTransaction && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-slate-600 mb-1">Transaction ID</p>
                  <p className="font-mono text-sm font-medium text-slate-800">
                    {selectedTransaction.id}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-slate-600 mb-1">Status</p>
                  <span className={getStatusBadge(selectedTransaction.prediction)}>
                    {selectedTransaction.prediction}
                  </span>
                </div>
                <div>
                  <p className="text-sm text-slate-600 mb-1">Fraud Probability</p>
                  <p className="text-lg font-bold text-slate-800">
                    {(selectedTransaction.fraud_probability * 100).toFixed(2)}%
                  </p>
                </div>
                <div>
                  <p className="text-sm text-slate-600 mb-1">Timestamp</p>
                  <p className="text-sm font-medium text-slate-800">
                    {new Date(selectedTransaction.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>

              <div>
                <p className="text-sm text-slate-600 mb-2">Transaction Data</p>
                <div className="bg-slate-50 rounded-lg p-4 max-h-96 overflow-auto">
                  <pre className="text-xs font-mono text-slate-700">
                    {JSON.stringify(selectedTransaction.transaction_data, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Transactions;
