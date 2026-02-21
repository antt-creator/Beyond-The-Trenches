import { useState, useEffect } from 'react';
import { format } from 'date-fns';
import { CheckCircle, Clock, ExternalLink } from 'lucide-react';
import { supabase } from '../lib/supabase';

interface Order {
  id: string;
  date: string;
  name: string;
  country: string;
  phone: string;
  address: string;
  qty: number;
  paymentType: string;
  status: string;
  receiptUrl: string | null;
  notes: string | null;
}

export default function AdminDashboard() {
  const [orders, setOrders] = useState<Order[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchOrders = async () => {
    try {
      const { data, error } = await supabase
        .from('orders')
        .select('*')
        .order('date', { ascending: false });
      
      if (error) throw error;
      
      if (Array.isArray(data)) {
        setOrders(data);
      } else {
        setOrders([]);
      }
    } catch (error) {
      console.error('Failed to fetch orders:', error);
      setOrders([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOrders();
  }, []);

  const updateStatus = async (id: string, currentStatus: string) => {
    const newStatus = currentStatus === 'Pending' ? 'Confirmed' : 'Pending';
    try {
      const { error } = await supabase
        .from('orders')
        .update({ status: newStatus })
        .eq('id', id);
      
      if (error) throw error;
      fetchOrders();
    } catch (error) {
      console.error('Failed to update status:', error);
    }
  };

  if (loading) return <div className="p-8 text-center">Loading orders...</div>;

  return (
    <div className="p-4 md:p-8 max-w-7xl mx-auto">
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-3xl font-serif font-bold">Order Management</h2>
        <button 
          onClick={fetchOrders}
          className="text-sm text-olive hover:underline"
        >
          Refresh Data
        </button>
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-black/5 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-warm-bg/50 border-bottom border-black/5">
                <th className="p-4 text-xs font-semibold uppercase tracking-wider text-black/40">Order ID</th>
                <th className="p-4 text-xs font-semibold uppercase tracking-wider text-black/40">Date</th>
                <th className="p-4 text-xs font-semibold uppercase tracking-wider text-black/40">Customer</th>
                <th className="p-4 text-xs font-semibold uppercase tracking-wider text-black/40">Country</th>
                <th className="p-4 text-xs font-semibold uppercase tracking-wider text-black/40">Address</th>
                <th className="p-4 text-xs font-semibold uppercase tracking-wider text-black/40">Qty</th>
                <th className="p-4 text-xs font-semibold uppercase tracking-wider text-black/40">Payment</th>
                <th className="p-4 text-xs font-semibold uppercase tracking-wider text-black/40">Status</th>
                <th className="p-4 text-xs font-semibold uppercase tracking-wider text-black/40">Receipt</th>
                <th className="p-4 text-xs font-semibold uppercase tracking-wider text-black/40">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-black/5">
              {orders.map((order) => (
                <tr key={order.id} className="hover:bg-warm-bg/20 transition-colors">
                  <td className="p-4 font-mono text-xs">{order.id}</td>
                  <td className="p-4 text-sm">
                    {order.date ? format(new Date(order.date), 'MMM d, HH:mm') : 'N/A'}
                  </td>
                  <td className="p-4">
                    <div className="text-sm font-medium">{order.name}</div>
                    <div className="text-xs text-black/50">{order.phone}</div>
                  </td>
                  <td className="p-4 text-sm">{order.country}</td>
                  <td className="p-4 text-sm max-w-[200px] truncate" title={order.address}>{order.address}</td>
                  <td className="p-4 text-sm">{order.qty}</td>
                  <td className="p-4 text-sm">{order.paymentType}</td>
                  <td className="p-4">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      order.status === 'Confirmed' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                    }`}>
                      {order.status === 'Confirmed' ? <CheckCircle className="w-3 h-3 mr-1" /> : <Clock className="w-3 h-3 mr-1" />}
                      {order.status}
                    </span>
                  </td>
                  <td className="p-4">
                    {order.receiptUrl ? (
                      <a 
                        href={order.receiptUrl} 
                        target="_blank" 
                        rel="noreferrer"
                        className="text-olive hover:text-olive/70 inline-flex items-center text-xs"
                      >
                        View <ExternalLink className="w-3 h-3 ml-1" />
                      </a>
                    ) : (
                      <span className="text-xs text-black/30">N/A</span>
                    )}
                  </td>
                  <td className="p-4">
                    <button 
                      onClick={() => updateStatus(order.id, order.status)}
                      className="text-xs font-medium text-olive hover:underline"
                    >
                      Toggle Status
                    </button>
                  </td>
                </tr>
              ))}
              {orders.length === 0 && (
                <tr>
                  <td colSpan={9} className="p-8 text-center text-black/40">No orders found.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
