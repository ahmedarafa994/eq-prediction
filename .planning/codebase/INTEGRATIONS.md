# External Integrations

**Analysis Date:** 2026-04-05

## APIs & External Services

**Audio Processing:**
- No external audio processing APIs used
  - Custom differentiable STFT implementation (dsp_frontend.py)
  - Biquad filter coefficients computed using Robert Bristow-Johnson formulas (differentiable_eq.py)

**Data Storage:**
- Local filesystem only
  - Precomputed dataset caching: data/dataset_musdb_200k.pt
  - Checkpoint storage: insight/checkpoints/
  - Training logs: insight/*.log files (train_200k.log, train_200k_opt.log)

**ML Experiment Tracking:**
- No experiment tracking integration detected
  - wandb (Weights & Biases) - Listed in requirements as optional but not used

## Data Storage

**Databases:**
- No database used
  - Training data generated on-the-fly or precomputed to disk (torch tensor cache)

**File Storage:**
- Local filesystem only
  - Checkpoints: `insight/checkpoints/`
  - Dataset cache: `insight/data/dataset_musdb_200k.pt`
  - Training logs: `insight/*.log`
  - MUSDB18 dataset: `insight/data/musdb18/` (optional)

**Caching:**
- Memory caching (precompute mode)
  - SyntheticEQDataset.precompute() caches mel-spectrograms in memory
- No external caching service

## Authentication & Identity

**Auth Provider:**
- No authentication system used
  - All training and evaluation scripts run locally
  - No API keys or authentication required

## Monitoring & Observability

**Error Tracking:**
- No error tracking integration detected
  - No Sentry, Datadog, or similar services

**Logs:**
- Console logging only
  - Training progress logged via print statements
  - Log files in insight/*.log format

## CI/CD & Deployment

**Hosting:**
- No cloud hosting detected
  - Scripts are designed for local execution

**CI Pipeline:**
- No CI/CD configuration detected
  - No GitHub Actions, GitLab CI, or similar configs

## Environment Configuration

**Required env vars:**
- None detected in codebase

**Secrets location:**
- No secrets required for local development
- No API keys or credentials in repository
- MUSDB18 path is configured in YAML config files

## Webhooks & Callbacks

**Incoming:**
- No webhook endpoints configured
- No API callbacks

**Outgoing:**
- No external API calls or webhook notifications
- Export script generates ONNX files for local use only

---

*Integration audit: 2026-04-05*
