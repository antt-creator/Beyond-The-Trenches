import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { COUNTRIES, BOOK_INFO, CountryCode } from './constants';
import OrderForm from './components/OrderForm';
import AdminDashboard from './components/AdminDashboard';
import { ShoppingCart, MapPin, Users, CheckCircle, ArrowRight, BookOpen } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState<'shop' | 'admin'>('shop');
  const [selectedCountry, setSelectedCountry] = useState<CountryCode>('TH');
  const [showOrderForm, setShowOrderForm] = useState(false);
  const [orderSuccess, setOrderSuccess] = useState<string | null>(null);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted) return null;

  const country = COUNTRIES[selectedCountry];

  const handleOrderSuccess = (orderId: string) => {
    setOrderSuccess(orderId);
    setShowOrderForm(false);
  };

  return (
    <div className="min-h-screen flex flex-col bg-warm-bg">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 bg-warm-bg/80 backdrop-blur-md border-b border-black/5">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BookOpen className="w-6 h-6 text-olive" />
            <span className="font-serif font-bold text-xl tracking-tight">{BOOK_INFO.publisher.toUpperCase()}</span>
          </div>
          <div className="flex gap-6">
            <button 
              onClick={() => setActiveTab('shop')}
              className={`text-sm font-medium transition-colors ${activeTab === 'shop' ? 'text-olive' : 'text-black/40 hover:text-black'}`}
            >
              The Book
            </button>
            <button 
              onClick={() => setActiveTab('admin')}
              className={`text-sm font-medium transition-colors ${activeTab === 'admin' ? 'text-olive' : 'text-black/40 hover:text-black'}`}
            >
              Admin
            </button>
          </div>
        </div>
      </nav>

      <main className="flex-grow">
        <AnimatePresence mode="wait">
          {activeTab === 'shop' ? (
            <motion.div
              key="shop"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="max-w-7xl mx-auto px-4 py-12 md:py-20"
            >
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
                {/* Book Cover */}
                <div className="relative group">
                  <div className="absolute -inset-4 bg-olive/5 rounded-[40px] blur-2xl group-hover:bg-olive/10 transition-all duration-500" />
                  <motion.img
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.2 }}
                    src={BOOK_INFO.coverImage}
                    alt={BOOK_INFO.title}
                    className="relative w-full max-w-md mx-auto rounded-2xl shadow-2xl transform group-hover:rotate-1 transition-transform duration-500"
                  />
                  <div className="absolute top-4 right-4 md:top-8 md:right-8 bg-white/90 backdrop-blur px-4 py-2 rounded-full shadow-lg border border-black/5">
                    <span className="text-xs font-bold text-olive uppercase tracking-widest flex items-center gap-2">
                      <CheckCircle className="w-3 h-3" />
                      {BOOK_INFO.stockStatus}
                    </span>
                  </div>
                </div>

                {/* Content */}
                <div className="space-y-8">
                  <div className="space-y-4">
                    <h1 className="text-5xl md:text-7xl font-serif font-bold leading-tight">
                      {BOOK_INFO.title}
                    </h1>
                    <p className="text-lg text-black/60 italic font-serif">by {BOOK_INFO.author}</p>
                  </div>

                  <p className="text-lg leading-relaxed text-black/70 max-w-xl">
                    {BOOK_INFO.synopsis}
                  </p>

                    <div className="space-y-6 pt-4">
                      <div className="space-y-4">
                        <label className="text-[10px] font-bold uppercase tracking-widest text-black/40">Select Your Country</label>
                        <div className="flex flex-wrap gap-2">
                          {(Object.keys(COUNTRIES) as CountryCode[]).map((code) => (
                            <button
                              key={code}
                              onClick={() => setSelectedCountry(code)}
                              className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                                selectedCountry === code 
                                ? 'bg-olive text-white border-olive shadow-md' 
                                : 'bg-white text-black/60 border-black/10 hover:border-olive/30'
                              }`}
                            >
                              {COUNTRIES[code].name}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="flex flex-wrap gap-4 items-center">
                        <div className="space-y-1">
                          <label className="text-[10px] font-bold uppercase tracking-widest text-black/40">Price</label>
                          <p className="text-3xl font-serif font-bold text-olive">
                            {country.currency} {country.price.toLocaleString()}
                          </p>
                          {country.shippingFee && (
                            <p className="text-[10px] text-black/40">+ {country.currency} {country.shippingFee} Shipping</p>
                          )}
                        </div>
                      </div>

                      {country.isDirectOrder ? (
                        <button 
                          onClick={() => setShowOrderForm(true)}
                          className="btn-primary flex items-center gap-2 group"
                        >
                          <ShoppingCart className="w-5 h-5" />
                          Buy Now
                          <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                        </button>
                      ) : (
                        <div className="p-4 bg-olive/5 rounded-2xl border border-olive/10">
                          <p className="text-sm text-olive font-medium">
                            ဤနိုင်ငံအတွက် အောက်ပါကိုယ်စားလှယ်များထံ Messenger မှတစ်ဆင့် တိုက်ရိုက်ဆက်သွယ်ဝယ်ယူနိုင်ပါသည်။
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Agents */}
                    {!country.isDirectOrder && country.agents.length > 0 && (
                      <div className="pt-8 border-t border-black/5">
                        <h3 className="text-sm font-bold uppercase tracking-widest text-black/40 mb-6 flex items-center gap-2">
                          <Users className="w-4 h-4" />
                          Authorized Agents in {country.name}
                        </h3>
                        <div className="flex flex-col gap-3">
                          {country.agents.map((agent, idx) => (
                            <a 
                              key={idx} 
                              href={agent.link}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="bg-white p-5 rounded-2xl border border-black/5 hover:border-olive/40 hover:bg-olive/[0.02] hover:shadow-lg transition-all group flex items-center justify-between"
                            >
                              <div className="flex items-center gap-4">
                                <div className="w-12 h-12 rounded-full bg-olive/10 flex items-center justify-center text-olive group-hover:bg-olive group-hover:text-white transition-colors">
                                  <Users className="w-6 h-6" />
                                </div>
                                <div>
                                  <p className="font-bold text-base group-hover:text-olive transition-colors">{agent.name}</p>
                                  {agent.location && (
                                    <div className="flex items-center gap-1 text-xs text-black/50 mt-1">
                                      <MapPin className="w-3 h-3" />
                                      {agent.location}
                                    </div>
                                  )}
                                  <p className="text-[10px] font-bold uppercase text-olive mt-1 tracking-widest">Contact via Messenger</p>
                                </div>
                              </div>
                              <div className="w-10 h-10 rounded-full bg-black/5 flex items-center justify-center group-hover:bg-olive/10 transition-colors">
                                <ArrowRight className="w-5 h-5 text-black/20 group-hover:text-olive transition-colors" />
                              </div>
                            </a>
                          ))}
                        </div>
                      </div>
                    )}
                </div>
              </div>

              {/* Telegram Contact Section - Only for Thailand */}
              {selectedCountry === 'TH' && (
                <section className="mt-20 pt-12 border-t border-black/5">
                  <div className="bg-olive/5 rounded-[32px] p-8 md:p-12 flex flex-col md:flex-row items-center justify-between gap-8">
                    <div className="text-center md:text-left">
                      <h3 className="text-2xl md:text-3xl font-serif font-bold mb-2">စာအုပ်မှာယူရာတွင် အခက်အခဲတစုံတရာရှိပါက</h3>
                    </div>
                    <a 
                      href="https://t.me/antt21" 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="flex items-center gap-3 bg-[#229ED9] text-white px-8 py-4 rounded-2xl font-bold hover:shadow-xl hover:scale-105 transition-all"
                    >
                      <svg className="w-6 h-6 fill-current" viewBox="0 0 24 24">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm4.64 6.8c-.15 1.58-.8 5.42-1.13 7.19-.14.75-.42 1-.68 1.03-.58.05-1.02-.38-1.58-.75-.88-.58-1.38-.94-2.23-1.5-.99-.65-.35-1.01.22-1.59.15-.15 2.71-2.48 2.76-2.69.01-.03.01-.14-.07-.2-.08-.06-.19-.04-.27-.02-.11.02-1.93 1.23-5.46 3.62-.51.35-.98.52-1.4.51-.46-.01-1.35-.26-2.01-.48-.81-.27-1.45-.42-1.39-.89.03-.24.37-.49 1.02-.73 4-1.74 6.67-2.88 8-3.43 3.81-1.56 4.6-1.83 5.12-1.84.11 0 .37.03.54.17.14.12.18.28.2.45.02.07.02.15.01.23z"/>
                      </svg>
                      Telegram သို့ ဆက်သွယ်ရန်
                    </a>
                  </div>
                </section>
              )}
            </motion.div>
          ) : (
            <motion.div
              key="admin"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <AdminDashboard />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Order Modal */}
      <AnimatePresence>
        {showOrderForm && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowOrderForm(false)}
              className="absolute inset-0 bg-black/40 backdrop-blur-sm"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="relative w-full max-w-2xl bg-warm-bg rounded-3xl shadow-2xl overflow-hidden"
            >
              <div className="p-6 md:p-8 overflow-y-auto max-h-[90vh]">
                <div className="flex justify-between items-center mb-8">
                  <div>
                    <h2 className="text-3xl font-serif font-bold">Secure Order</h2>
                    <p className="text-sm text-black/50">Shipping to {country.name}</p>
                  </div>
                  <button 
                    onClick={() => setShowOrderForm(false)}
                    className="w-10 h-10 rounded-full bg-black/5 flex items-center justify-center hover:bg-black/10 transition-colors"
                  >
                    ✕
                  </button>
                </div>
                <OrderForm 
                  countryCode={selectedCountry} 
                  onSuccess={handleOrderSuccess} 
                />
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Success Modal */}
      <AnimatePresence>
        {orderSuccess && (
          <div className="fixed inset-0 z-[110] flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 bg-black/60 backdrop-blur-md"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="relative bg-white rounded-[40px] p-12 text-center max-w-md w-full shadow-2xl"
            >
              <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <CheckCircle className="w-10 h-10 text-green-600" />
              </div>
              <h2 className="text-3xl font-serif font-bold mb-2">Order Confirmed!</h2>
              <p className="text-black/60 mb-8">
                Thank you for your purchase. Your order has been received and is being processed.
              </p>
              <div className="bg-warm-bg p-4 rounded-2xl mb-8">
                <p className="text-[10px] font-bold uppercase tracking-widest text-black/40 mb-1">Order ID</p>
                <p className="text-2xl font-mono font-bold text-olive">{orderSuccess}</p>
              </div>
              <button 
                onClick={() => setOrderSuccess(null)}
                className="btn-primary w-full"
              >
                Back to Home
              </button>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      <footer className="py-12 border-t border-black/5 text-center">
        <p className="text-sm text-black/40">© 2026 {BOOK_INFO.publisher}. All rights reserved.</p>
      </footer>
    </div>
  );
}
