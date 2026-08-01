# Deployment

Frontend on **Vercel**. Backend on either **Render** (`deploy/render/`) or
**Hugging Face Spaces** (`deploy/hf-space/`) — both are prepared and verified.

Both need **your** accounts. Everything that can be prepared in advance already
is; what remains is authentication and a push.

## Which backend host

Measured, not assumed. Peak RSS with all models loaded and a 107k-token input
compressed is **408 MB**; the lite build (no torch) is ~150 MB.

| Host / plan | vCPU | RAM | Build | Fact survival | Cost |
|---|---|---|---|---|---|
| HF Spaces (free) | 2 | 16 GB | full | **80.8%** | free |
| Render Starter | 0.5 | 512 MB | full (fits at 408 MB) | **80.8%** | $7/mo |
| Render Standard | 1 | 2 GB | full | **80.8%** | $25/mo |
| Render Free | 0.1 | 512 MB | **lite only** | 69.2% | free |

Render's free tier is the one to think twice about: 512 MB is survivable, but
**0.1 vCPU is not** — MiniLM encoding on a tenth of a core makes every request
feel broken. If free is a hard requirement, use the lite build, which does no
neural encoding at all. It costs 11.6 points of fact survival (80.8% → 69.2%);
compression ratio is unchanged at ~72% because the token budget forces that
either way.

---

## Backend option A → Render

```bash
# Blueprint: Render dashboard > New > Blueprint > point at this repo.
# It reads deploy/render/render.yaml.
```

Or configure a Web Service manually:

| Setting | Value |
|---|---|
| Runtime | Docker |
| Dockerfile path | `./deploy/render/Dockerfile` |
| Docker context | `.` (repo root) |
| Health check path | `/health` |
| Env var | `CCE_FORCE_FAST_MODE=1` |

For the **lite** build add build arg `BUILD=lite` (Render: *Environment* →
*Docker Build Arguments*). Default is `full`.

Render injects `$PORT`; the Dockerfile binds to it. Binding to a fixed port is
the usual cause of a service that builds fine and then never passes its health
check.

Verify:

```bash
curl https://<service>.onrender.com/health | jq '{warm, force_fast_mode}'
curl https://<service>.onrender.com/health | jq '.embeddings.available'
# false on a lite build - that is expected, and the UI reports it
```

**Free-tier services sleep after 15 minutes** and take ~30-60 s to wake with
models loading. Hit `/health` a few minutes before demoing.

---

## Backend option B → Hugging Face Space

```bash
./deploy/build-space.sh          # assembles deploy/.space-build (~700 KB)
```

Then create the Space at <https://huggingface.co/new-space> with **SDK: Docker**
and **Hardware: CPU basic (free)**, and push:

```bash
cd deploy/.space-build
git init && git add -A && git commit -m "Context compression engine API"
git remote add origin https://huggingface.co/spaces/<user>/<space>
git push -u origin main
```

If the push asks for credentials, run `huggingface-cli login` first (a token
from <https://huggingface.co/settings/tokens> with *write* scope).

**First build takes ~10 minutes** — it bakes MiniLM, the spaCy pipeline and the
tiktoken BPE table into the image so nothing is downloaded at request time.
Watch the *Build logs* tab.

Verify when it goes live:

```bash
curl https://<user>-<space>.hf.space/health
```

`warm: true` and `force_fast_mode: true` is the expected state.

### What the hosted build does differently

| | Local | Hosted Space |
|---|---|---|
| Stage 6 (abstractive) | available via Ollama | **skipped** (`CCE_FORCE_FAST_MODE=1`) |
| Embeddings | Apple Metal (MPS) | CPU |
| Answering model | llama3.2:3b via Ollama | not present — `/evaluate` serves the measured report |

This is an intentional fallback, not a degraded build. Stage 6 saved **0 tokens**
on this corpus while costing 3–6 s per chunk, so removing it costs nothing
measurable. Compression ratio and cost reduction transfer unchanged; **latency
figures do not** — the 4.05× benchmark was measured on an M2.

`/evaluate` serves the committed `reports/latest.json`. That file is
force-included in `.gitignore` precisely so the deployed dashboard has real
numbers; without it the hero cards would be empty.

---

## Frontend → Vercel

```bash
cd frontend
echo "VITE_API_URL=https://<user>-<space>.hf.space" > .env.production
npm run build          # dist/ is ~100 KB gzipped
npx vercel --prod      # or drag dist/ onto https://app.netlify.com/drop
```

The backend already allows `*.vercel.app`, `*.netlify.app` and `*.onrender.com`
origins via CORS regex, so no backend change is needed. For any other domain,
add it to `allow_origins` in `backend/main.py`.

---

## Verify the deployed instance end to end

Do this before considering deployment done — a URL that loads but cannot
compress is worse than no URL.

```bash
API=https://<user>-<space>.hf.space

curl -s $API/health | jq '{warm, force_fast_mode}'
curl -s -X POST $API/evaluate -H 'Content-Type: application/json' \
     -d '{"run":false}' | jq '.aggregate'
curl -s -X POST $API/compress -H 'Content-Type: application/json' \
     -d '{"text":"def a(x):\n    return x+1\n\ndef b(y):\n    return y+1\n","name":"t.py","budget_ratio":0.5}' \
     | jq '.summary.compression_pct'
```

Then open the Vercel URL and run one compression through the UI. Confirm the
hero cards populate, the diff renders, and stage 6 shows its skip note.

**Cold starts:** a free Space sleeps after inactivity and takes ~30 s to wake.
Hit `/health` a few minutes before demoing so the first judge click is warm.

---

## Local one-command run

```bash
./run.sh            # API on :8000, dashboard on :5173
./run.sh --check    # verify the environment without starting anything
./run.sh --backend  # API only
```
