import { useState, useEffect } from 'react';
import { axiosInstance } from '@/App';
import { Shield, Plus, Unlock, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { toast } from 'sonner';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

const BlockManagement = ({ user }) => {
  const [blocks, setBlocks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [unblockDialogOpen, setUnblockDialogOpen] = useState(false);
  const [selectedBlock, setSelectedBlock] = useState(null);
  const [blockForm, setBlockForm] = useState({
    entity_type: 'upi_id',
    entity_value: '',
    reason: '',
  });
  const [unblockReason, setUnblockReason] = useState('');

  useEffect(() => {
    fetchBlocks();
  }, []);

  const fetchBlocks = async () => {
    setLoading(true);
    try {
      const response = await axiosInstance.get('/blocks');
      setBlocks(response.data);
    } catch (error) {
      toast.error('Failed to fetch blocked entities');
    } finally {
      setLoading(false);
    }
  };

  const handleBlock = async () => {
    if (!blockForm.entity_value || !blockForm.reason) {
      toast.error('Please fill all fields');
      return;
    }

    try {
      await axiosInstance.post('/blocks', blockForm);
      toast.success('Entity blocked successfully');
      setDialogOpen(false);
      setBlockForm({ entity_type: 'upi_id', entity_value: '', reason: '' });
      fetchBlocks();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to block entity');
    }
  };

  const handleUnblock = async () => {
    if (!unblockReason) {
      toast.error('Please provide a reason for unblocking');
      return;
    }

    try {
      await axiosInstance.put(`/blocks/${selectedBlock.id}/unblock?reason=${encodeURIComponent(unblockReason)}`);
      toast.success('Entity unblocked successfully');
      setUnblockDialogOpen(false);
      setUnblockReason('');
      setSelectedBlock(null);
      fetchBlocks();
    } catch (error) {
      toast.error('Failed to unblock entity');
    }
  };

  const openUnblockDialog = (block) => {
    setSelectedBlock(block);
    setUnblockDialogOpen(true);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-800">Block Management</h1>
          <p className="text-slate-600 mt-1">Manage blocked UPI IDs and merchants</p>
        </div>
        <Button
          data-testid="add-block-button"
          onClick={() => setDialogOpen(true)}
          className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white flex items-center gap-2"
        >
          <Plus size={20} />
          Block Entity
        </Button>
      </div>

      {/* Blocked Entities List */}
      <div className="bg-white rounded-xl shadow-lg border border-slate-200 overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          </div>
        ) : blocks.length === 0 ? (
          <div className="text-center py-16 px-4">
            <Shield className="mx-auto mb-4 text-slate-300" size={64} />
            <p className="text-slate-600 text-lg">No blocked entities</p>
            <p className="text-slate-500 text-sm mt-2">Add entities to the blocklist to prevent fraudulent transactions</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full" data-testid="blocks-table">
              <thead className="bg-slate-50 border-b border-slate-200">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">
                    Entity Type
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">
                    Entity Value
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">
                    Reason
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">
                    Blocked By
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">
                    Blocked At
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {blocks.map((block) => (
                  <tr key={block.id} data-testid={`block-row-${block.id}`} className="hover:bg-slate-50 transition-colors">
                    <td className="px-6 py-4">
                      <span className="px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-700 capitalize">
                        {block.entity_type.replace('_', ' ')}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm font-medium text-slate-800">
                      {block.entity_value}
                    </td>
                    <td className="px-6 py-4 text-sm text-slate-600">
                      {block.reason}
                    </td>
                    <td className="px-6 py-4 text-sm text-slate-600">
                      {block.blocked_by}
                    </td>
                    <td className="px-6 py-4 text-sm text-slate-600">
                      {new Date(block.blocked_at).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4">
                      <Button
                        data-testid={`unblock-button-${block.id}`}
                        onClick={() => openUnblockDialog(block)}
                        size="sm"
                        className="flex items-center gap-2 bg-green-500 hover:bg-green-600 text-white"
                      >
                        <Unlock size={16} />
                        Unblock
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Block Entity Dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent data-testid="block-entity-dialog">
          <DialogHeader>
            <DialogTitle>Block Entity</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Entity Type
              </label>
              <Select
                value={blockForm.entity_type}
                onValueChange={(value) => setBlockForm({ ...blockForm, entity_type: value })}
              >
                <SelectTrigger data-testid="entity-type-select">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="upi_id">UPI ID</SelectItem>
                  <SelectItem value="merchant">Merchant</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Entity Value
              </label>
              <Input
                data-testid="entity-value-input"
                type="text"
                value={blockForm.entity_value}
                onChange={(e) => setBlockForm({ ...blockForm, entity_value: e.target.value })}
                placeholder="e.g., user@upi or merchant_id"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Reason
              </label>
              <Input
                data-testid="block-reason-input"
                type="text"
                value={blockForm.reason}
                onChange={(e) => setBlockForm({ ...blockForm, reason: e.target.value })}
                placeholder="Reason for blocking"
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              data-testid="cancel-block-button"
              onClick={() => setDialogOpen(false)}
              className="bg-slate-200 text-slate-700 hover:bg-slate-300"
            >
              Cancel
            </Button>
            <Button
              data-testid="confirm-block-button"
              onClick={handleBlock}
              className="bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white"
            >
              Block Entity
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Unblock Entity Dialog */}
      <Dialog open={unblockDialogOpen} onOpenChange={setUnblockDialogOpen}>
        <DialogContent data-testid="unblock-entity-dialog">
          <DialogHeader>
            <DialogTitle>Unblock Entity</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div className="p-4 bg-amber-50 rounded-lg border border-amber-200 flex items-start gap-3">
              <AlertCircle className="text-amber-600 flex-shrink-0 mt-0.5" size={20} />
              <div>
                <p className="text-sm font-medium text-amber-800">Confirm Unblock</p>
                <p className="text-sm text-amber-700 mt-1">
                  You are about to unblock: <span className="font-semibold">{selectedBlock?.entity_value}</span>
                </p>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Reason for Unblocking
              </label>
              <Input
                data-testid="unblock-reason-input"
                type="text"
                value={unblockReason}
                onChange={(e) => setUnblockReason(e.target.value)}
                placeholder="Provide reason for unblocking"
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              data-testid="cancel-unblock-button"
              onClick={() => {
                setUnblockDialogOpen(false);
                setUnblockReason('');
                setSelectedBlock(null);
              }}
              className="bg-slate-200 text-slate-700 hover:bg-slate-300"
            >
              Cancel
            </Button>
            <Button
              data-testid="confirm-unblock-button"
              onClick={handleUnblock}
              className="bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white"
            >
              Unblock Entity
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default BlockManagement;
