import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { motion, AnimatePresence } from 'motion/react';
import { COUNTRIES, CountryCode } from '../constants';
import { Upload, CheckCircle2, Loader2 } from 'lucide-react';
import { supabase } from '../lib/supabase';

const orderSchema = z.object({
  name: z.string().min(2, 'Name is too short'),
  phone: z.string().min(8, 'Invalid phone number'),
  address: z.string().min(5, 'Please provide a more detailed address'),
  qty: z.number().min(1, 'Minimum 1 copy'),
  paymentType: z.enum(['Prepaid', 'COD'], {
    message: 'ကျေးဇူးပြု၍ ငွေပေးချေမှုစနစ်ကို ရွေးချယ်ပေးပါ။',
  }),
  notes: z.string().optional(),
  receipt: z.any().optional(),
});

type OrderFormData = z.infer<typeof orderSchema>;

interface OrderFormProps {
  countryCode: CountryCode;
  onSuccess: (orderId: string) => void;
}

export default function OrderForm({ countryCode, onSuccess }: OrderFormProps) {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [receiptPreview, setReceiptPreview] = useState<string | null>(null);
  const country = COUNTRIES[countryCode];

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    formState: { errors, isDirty, isValid },
  } = useForm<OrderFormData>({
    resolver: zodResolver(orderSchema),
    mode: 'onBlur',
    defaultValues: {
      qty: 1,
      phone: country.phonePrefix || '',
    },
  });

  const paymentType = watch('paymentType');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setReceiptPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const onSubmit = async (data: OrderFormData) => {
    setIsSubmitting(true);
    setSubmitError(null);
    
    try {
      // We omit 'id' and 'date' to let Supabase use its default values (gen_random_uuid() and now())
      const { data: insertedData, error } = await supabase
        .from('orders')
        .insert([
          {
            name: data.name,
            phone: data.phone,
            address: data.address,
            qty: data.qty,
            paymentType: data.paymentType,
            receiptUrl: receiptPreview || null,
            notes: data.notes || null
          }
        ])
        .select();

      if (error) throw error;
      
      const generatedId = insertedData && insertedData[0] ? insertedData[0].id : "Success";
      onSuccess(generatedId);
    } catch (error: any) {
      console.error('Order submission failed:', error);
      setSubmitError(`Order တင်ခြင်း မအောင်မြင်ပါ။ ကျေးဇူးပြု၍ ပြန်လည်ကြိုးစားပေးပါ။ (${error.message || error})`);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
      {Object.keys(errors).length > 0 && (
        <div className="p-3 bg-red-50 border border-red-100 rounded-xl">
          <p className="text-xs text-red-600 font-medium">ကျေးဇူးပြု၍ လိုအပ်သော အချက်အလက်များကို မှန်ကန်စွာ ဖြည့်စွက်ပေးပါရန်။</p>
        </div>
      )}

      {submitError && (
        <div className="p-3 bg-red-50 border border-red-100 rounded-xl">
          <p className="text-xs text-red-600 font-medium">{submitError}</p>
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-black/50 mb-2">Full Name</label>
          <input {...register('name')} className="input-field" placeholder="John Doe" />
          {errors.name && <p className="text-red-500 text-xs mt-1">{errors.name.message}</p>}
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-black/50 mb-2">Phone Number</label>
          <input {...register('phone')} className="input-field" placeholder={country.phonePrefix || 'Phone number'} />
          {errors.phone && <p className="text-red-500 text-xs mt-1">{errors.phone.message}</p>}
        </div>
      </div>

      <div>
        <label className="block text-xs font-semibold uppercase tracking-wider text-black/50 mb-2">Full Delivery Address</label>
        <textarea {...register('address')} rows={3} className="input-field resize-none" placeholder="Eg. 24/8, Jelly Road, Mae Sot, Tak 63110" />
        {errors.address && <p className="text-red-500 text-xs mt-1">{errors.address.message}</p>}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-black/50 mb-2">Quantity</label>
          <input 
            type="number" 
            {...register('qty', { valueAsNumber: true })} 
            className="input-field" 
          />
          {errors.qty && <p className="text-red-500 text-xs mt-1">{errors.qty.message}</p>}
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-black/50 mb-2">Payment Method</label>
          <div className="flex flex-col gap-2 mt-2">
            <label className="flex items-center gap-2 cursor-pointer p-3 rounded-xl border border-black/5 hover:bg-black/5 transition-colors">
              <input type="radio" value="COD" {...register('paymentType')} className="accent-olive" />
              <span className="text-sm">အိမ်ရောက်ငွေချေစနစ်ဖြင့် မှာယူမည်</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer p-3 rounded-xl border border-black/5 hover:bg-black/5 transition-colors">
              <input type="radio" value="Prepaid" {...register('paymentType')} className="accent-olive" />
              <span className="text-sm">ကြိုတင်ငွေချေမည်</span>
            </label>
          </div>
          {errors.paymentType && <p className="text-red-500 text-xs mt-1">{errors.paymentType.message}</p>}
        </div>
      </div>


      <AnimatePresence>
        {paymentType === 'Prepaid' && country.bankDetails && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="p-6 bg-olive/5 rounded-2xl border border-olive/10 space-y-4">
              <h4 className="font-serif font-bold text-lg">Payment Details</h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {country.bankDetails.map((bank, idx) => (
                  <div key={idx} className="bg-white p-3 rounded-xl border border-black/5">
                    <p className="text-[10px] font-bold uppercase text-olive">{bank.provider}</p>
                    <p className="text-sm font-medium">{bank.accountName}</p>
                    <p className="text-sm font-mono">{bank.accountNumber}</p>
                  </div>
                ))}
              </div>

              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-black/50 mb-2">ငွေလွှဲပြေစာ upload ရန်</label>
                <div className="relative group">
                  <input 
                    type="file" 
                    accept="image/*" 
                    onChange={handleFileChange}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                  />
                  <div className="border-2 border-dashed border-olive/20 rounded-xl p-8 text-center group-hover:border-olive/40 transition-colors">
                    {receiptPreview ? (
                      <div className="flex items-center justify-center gap-2 text-olive">
                        <CheckCircle2 className="w-5 h-5" />
                        <span className="text-sm font-medium">Screenshot Uploaded</span>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <Upload className="w-8 h-8 mx-auto text-olive/40" />
                        <p className="text-xs text-black/40">Click or drag to upload payment proof</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div>
        <label className="block text-xs font-semibold uppercase tracking-wider text-black/50 mb-2">Other Notes (Optional)</label>
        <input {...register('notes')} className="input-field" placeholder="Special instructions..." />
      </div>

      <button 
        type="submit" 
        disabled={isSubmitting}
        className="btn-primary w-full flex items-center justify-center gap-2"
      >
        {isSubmitting ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Processing Order...
          </>
        ) : (
          `Confirm Order - ${country.currency} ${((country.price * watch('qty')) + (country.shippingFee || 0)).toLocaleString()}`
        )}
      </button>
    </form>
  );
}
