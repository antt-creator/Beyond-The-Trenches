import express from "express";
import { createServer as createViteServer } from "vite";
import { createClient } from "@supabase/supabase-js";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Lazy Supabase initialization
let supabaseClient: any = null;
function getSupabase() {
  if (!supabaseClient) {
    const url = process.env.SUPABASE_URL;
    const key = process.env.SUPABASE_ANON_KEY;
    if (!url || !key) {
      console.warn("Missing SUPABASE_URL or SUPABASE_ANON_KEY. Admin features will not work.");
      return null;
    }
    supabaseClient = createClient(url, key);
  }
  return supabaseClient;
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '10mb' }));

  // API Routes
  app.get("/api/orders", async (req, res) => {
    const supabase = getSupabase();
    if (!supabase) return res.status(500).json({ error: "Supabase not configured" });

    try {
      const { data, error } = await supabase
        .from("orders")
        .select("*")
        .order("date", { ascending: false });
      
      if (error) throw error;
      res.json(data || []);
    } catch (error) {
      console.error(error);
      res.status(500).json({ error: "Failed to fetch orders", details: error });
    }
  });

  app.post("/api/orders", async (req, res) => {
    const supabase = getSupabase();
    if (!supabase) return res.status(500).json({ error: "Supabase not configured" });

    try {
      const order = req.body;
      const { data, error } = await supabase
        .from("orders")
        .insert([
          {
            name: order.name,
            phone: order.phone,
            address: order.address,
            qty: order.qty,
            paymentType: order.paymentType,
            receiptUrl: order.receiptUrl || null,
            notes: order.notes || null
          }
        ])
        .select();
      
      if (error) throw error;
      const generatedId = data && data[0] ? data[0].id : "Success";
      res.status(201).json({ success: true, orderId: generatedId });
    } catch (error) {
      console.error(error);
      res.status(500).json({ error: "Failed to create order", details: error });
    }
  });

  const vite = await createViteServer({
    server: { middlewareMode: true },
    appType: "spa",
  });

  app.use(vite.middlewares);

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
