# Deployment Guide — Vast.ai

## What You Need to Upload (not in git)

These files are too large for GitHub and must be uploaded manually:

| File | Size | Path on server |
|------|------|----------------|
| Dataset CSV | 617MB | `data/synthetic_corrosion_dataset.csv` |
| Model checkpoint | 4.3MB | `outputs/exp3_bilstm_optA/models/best_model.pt` |
| Feature scaler | 1.4KB | `outputs/scalers/feature_scaler.joblib` |
| Scaled columns | 365B | `outputs/scalers/scaled_columns.joblib` |

---

## Fresh Instance Setup (from scratch)

### 1. Rent an instance

- Go to [vast.ai](https://vast.ai)
- Template: **PyTorch (Vast)**
- Any cheap GPU/CPU works (the dashboard runs on CPU only)
- Minimum 4GB RAM, 2GB disk free

### 2. SSH into the instance

```bash
ssh -p <PORT> root@<IP>
```

### 3. Clone and build

```bash
apt update && apt install -y git
git clone https://github.com/Abdulrazak2004/GP-M078.git GP
cd GP && bash dashboard/deploy.sh
```

The deploy script will fail at startup because the data files aren't in git. That's expected — continue to step 4.

### 4. Create directories on the server

```bash
mkdir -p /workspace/GP/GP/data
mkdir -p /workspace/GP/GP/outputs/exp3_bilstm_optA/models
mkdir -p /workspace/GP/GP/outputs/scalers
```

### 5. Upload data files (from your Mac)

Open a **new terminal** on your Mac (not the SSH session):

```bash
cd ~/GP

scp -P <PORT> data/synthetic_corrosion_dataset.csv root@<IP>:/workspace/GP/GP/data/

scp -P <PORT> outputs/exp3_bilstm_optA/models/best_model.pt root@<IP>:/workspace/GP/GP/outputs/exp3_bilstm_optA/models/

scp -P <PORT> outputs/scalers/feature_scaler.joblib root@<IP>:/workspace/GP/GP/outputs/scalers/

scp -P <PORT> outputs/scalers/scaled_columns.joblib root@<IP>:/workspace/GP/GP/outputs/scalers/
```

Replace `<PORT>` and `<IP>` with your Vast.ai instance values.

### 6. Start the server (on the Vast.ai SSH)

```bash
cd /workspace/GP/GP/dashboard/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Wait for: `Ready. Serving at http://localhost:8000`

### 7. Get your public URL

Go to your Vast.ai instance page → **Tunnels** tab → find the tunnel for `http://localhost:8000` → copy the `trycloudflare.com` URL.

If no tunnel exists, click **Create New Tunnel** → enter `http://localhost:8000`.

Share the tunnel URL with professors.

---

## Restart (instance already set up, just stopped/restarted)

### 1. SSH in

```bash
ssh -p <PORT> root@<IP>
```

### 2. Start the server

```bash
cd /workspace/GP/GP/dashboard/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Get the new tunnel URL

Tunnel URLs change on restart. Go to Vast.ai → Tunnels → copy the new URL for port 8000.

---

## Pull Latest Code Changes

If you pushed updates from your Mac:

```bash
cd /workspace/GP/GP && git pull
cd dashboard/backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Rebuild Frontend (if you changed frontend code)

```bash
cd /workspace/GP/GP/dashboard/frontend
npm run build
cd ../backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: .csv` | Upload the dataset (step 5) |
| `FileNotFoundError: best_model.pt` | Upload model + scalers (step 5) |
| `cuDNN not compatible` | Already fixed — inference runs on CPU |
| `TypeError: only 0-dimensional arrays` | Already fixed — `git pull` to get latest |
| Server slow on first well load | Normal — 15 MC Dropout passes on CPU takes ~30s |
| Tunnel URL doesn't work | Check Vast.ai Tunnels tab for the current URL |
| `npm: command not found` | Run: `curl -fsSL https://deb.nodesource.com/setup_20.x \| bash - && apt install -y nodejs` |

---

## Cost

- $0.057/hr for the cheapest instance
- ~$1.37/day if left running 24/7
- **Remember to stop the instance when not in use**
