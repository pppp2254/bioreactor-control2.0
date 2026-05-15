import { Elysia } from "elysia"
import { cors } from "@elysiajs/cors"
import { staticPlugin } from "@elysiajs/static"

const PYTHON_API = process.env.PYTHON_API || "http://localhost:8000"

const app = new Elysia()
  .use(cors())
  .use(staticPlugin({ assets: "public", prefix: "/" }))
  .get("/", () => Bun.file("public/index.html"))

  .get("/api/available", async () =>
    fetch(`${PYTHON_API}/available`).then(r => r.json()))

  .get("/api/batches", async () =>
    fetch(`${PYTHON_API}/batches`).then(r => r.json()))

  .get("/api/signals/:batch", async ({ params, query }) => {
    const cols = (query as any).cols || "all"
    return fetch(`${PYTHON_API}/signals/${params.batch}?cols=${cols}`)
      .then(r => r.json())
  })

  .get("/api/sparse/:batch", async ({ params }) =>
    fetch(`${PYTHON_API}/sparse/${params.batch}`).then(r => r.json()))

  .get("/api/simulate/:batch", async ({ params }) =>
    fetch(`${PYTHON_API}/simulate/${params.batch}`).then(r => r.json()))

  .get("/api/predict/:batch", async ({ params }) =>
    fetch(`${PYTHON_API}/predict/${params.batch}`).then(r => r.json()))

  .get("/api/transforms/:batch", async ({ params }) =>
    fetch(`${PYTHON_API}/transforms/${params.batch}`).then(r => r.json()))

  .post("/api/upload", async ({ request }) => {
    const formData = await request.formData()
    const r = await fetch(`${PYTHON_API}/upload`, {
      method: "POST",
      body: formData,
    })
    return r.json()
  })

  .post("/api/step", async ({ body }) =>
    fetch(`${PYTHON_API}/step`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(r => r.json()))

  .post("/api/reset/:batch", async ({ params }) =>
    fetch(`${PYTHON_API}/reset/${params.batch}`, { method: "POST" })
      .then(r => r.json()))

  .ws("/ws/simulate", {
    open(ws) { console.log("Client connected") },

    async message(ws, msg: any) {
      const { batch, speed = 1 } = msg

      await fetch(`${PYTHON_API}/reset/${batch}`, { method: "POST" })

      const [continuous, sparse] = await Promise.all([
        fetch(`${PYTHON_API}/signals/${batch}`).then(r => r.json()),
        fetch(`${PYTHON_API}/sparse/${batch}`).then(r => r.json()),
      ]) as [any[], any[]]

      if (!sparse || sparse.length === 0) {
        ws.send(JSON.stringify({ error: "no sparse data" }))
        return
      }

      ws.send(JSON.stringify({
        type:             "meta",
        total_sparse:     sparse.length,
        has_continuous:   continuous?.length > 0,
      }))

      let lastControl: any = {
        phase: "growth", layer_used: "rules", action: "none",
        methanol_feed_pct: 0, stirrer_setpoint: 300,
        ml_confidence: null, L1_predicted: null, alerts: []
      }

      for (const row of sparse) {
        // Find nearest continuous reading within 1h
        let contReading: any = {}
        if (continuous?.length > 0) {
          const nearest = continuous.reduce((a: any, b: any) =>
            Math.abs(b.time_h - row.time_h) < Math.abs(a.time_h - row.time_h)
              ? b : a
          )
          if (Math.abs(nearest.time_h - row.time_h) < 1.0) {
            contReading = nearest
          }
        }

        const merged = { ...contReading, ...row }

        try {
          lastControl = await fetch(`${PYTHON_API}/step`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ batch, ...merged }),
          }).then(r => r.json())
        } catch(e) {
          console.error("Step error:", e)
        }

        ws.send(JSON.stringify({
          type:    "step",
          time_h:  row.time_h,
          reading: merged,
          control: lastControl,
        }))

        await new Promise(r => setTimeout(r, Math.max(50, 500 / speed)))
      }

      ws.send(JSON.stringify({ type: "done" }))
    },

    close(ws) { console.log("Client disconnected") }
  })

  .listen(3000)

console.log("Elysia running at http://localhost:3000")
