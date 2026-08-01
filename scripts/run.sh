#!/usr/bin/env bash
#
# run.sh — one-shot test harness for the knowbase system.
#
# Drop your two keys into .env (see .env.example), then:
#
#   ./scripts/run.sh all      # setup DB + deps, full ingest, eval matrix
#
# Or run stages individually:
#
#   ./scripts/run.sh setup    # start pgvector, wait for healthy, uv sync
#   ./scripts/run.sh ingest   # full re-ingest (clones corpus, ~3k LLM calls, 1-3h)
#   ./scripts/run.sh eval     # run vector/fts/hybrid/hybrid+rerank, capture numbers
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RAW_OUT="docs/blog/p3-eval-raw.md"

# --- helpers ----------------------------------------------------------------

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found on PATH. $2"; }

compose() {
  # Prefer the v2 plugin ("docker compose"); fall back to legacy "docker-compose".
  if docker compose version >/dev/null 2>&1; then docker compose "$@";
  else docker-compose "$@"; fi
}

load_env() {
  [ -f .env ] || die "No .env file. Copy .env.example to .env and fill in your keys."
  set -a; # shellcheck disable=SC1091
  . ./.env; set +a
}

require_key() {
  # require_key VAR placeholder-value
  local name="$1" placeholder="$2" val="${!1:-}"
  [ -n "$val" ]              || die "$name is empty in .env"
  [ "$val" != "$placeholder" ] || die "$name is still the placeholder value; put your real key in .env"
}

require_llm_key() {
  # Validate an LLM key for whatever provider config.yaml selects, reusing the
  # app's own resolver (LLM_API_KEY / OPENAI_API_KEY / CEREBRAS_API_KEY, or
  # ANTHROPIC_API_KEY). Env from .env is already exported by load_env.
  if uv run python -c "import sys; from knowbase.config import load_config; from knowbase.llm import resolve_api_key; sys.exit(0 if resolve_api_key(load_config().llm_provider) else 1)" 2>/dev/null; then
    return 0
  fi
  local prov; prov=$(uv run python -c "from knowbase.config import load_config; print(load_config().llm_provider)" 2>/dev/null || echo '?')
  die "No LLM API key set for provider '$prov'. Add the matching key to .env (see .env.example: LLM_API_KEY / OPENAI_API_KEY / CEREBRAS_API_KEY / ANTHROPIC_API_KEY)."
}

# --- stages -----------------------------------------------------------------

cmd_setup() {
  need docker "Install Docker Desktop and make sure the daemon is running."
  need uv     "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
  docker info >/dev/null 2>&1 || die "Docker daemon is not running. Start Docker Desktop first."

  log "Starting pgvector (docker compose) and waiting for it to be healthy..."
  compose up -d --wait db

  log "Installing Python deps (uv sync)..."
  uv sync

  log "Setup complete. DB is up on postgresql://knowbase:knowbase@localhost:5433/knowbase"
}

cmd_ingest() {
  load_env
  require_key GITHUB_TOKEN ghp_yourtokenhere
  require_llm_key
  compose up -d --wait db

  warn "Full ingest clones the corpus and makes ~3k LLM calls (distillation + bursting)."
  warn "Expect roughly 1-3 hours. Leave it running."
  log  "Starting full re-ingest..."
  local t0; t0=$(date +%s)
  uv run kb ingest --full
  log "Ingest finished in $(( ($(date +%s) - t0) / 60 )) min."
}

run_eval() {
  # run_eval <label> <extra kb eval args...>
  local label="$1"; shift
  {
    echo
    echo "## $label"
    echo '```'
  } >> "$RAW_OUT"
  uv run kb eval "$@" | tee -a "$RAW_OUT"
  echo '```' >> "$RAW_OUT"
}

cmd_eval() {
  load_env
  require_llm_key   # needed for the --rerank run
  compose up -d --wait db

  log "Running eval matrix over the 32-question set; capturing to $RAW_OUT"
  {
    echo "# P3 eval — raw capture"
    echo
    echo "_Generated $(date -u '+%Y-%m-%d %H:%M UTC') by scripts/run.sh eval, over docs/blog corpus._"
  } > "$RAW_OUT"

  run_eval "vector (baseline)"      --mode vector
  run_eval "fts"                    --mode fts
  run_eval "hybrid"                 --mode hybrid
  run_eval "hybrid + rerank (P4)"   --mode hybrid --rerank

  log "Done. Numbers captured in $RAW_OUT"
  log "Next: turn these into docs/blog/p3-results.md (vector vs hybrid vs fts on the distilled/burst corpus)"
  log "      and update docs/blog/p4-results.md (hybrid vs hybrid+rerank)."
}

cmd_all() { cmd_setup; cmd_ingest; cmd_eval; }

# --- dispatch ---------------------------------------------------------------

case "${1:-}" in
  setup)  cmd_setup ;;
  ingest) cmd_ingest ;;
  eval)   cmd_eval ;;
  all)    cmd_all ;;
  *) cat >&2 <<EOF
usage: ./scripts/run.sh <stage>

  setup    start pgvector (waits for healthy) + uv sync
  ingest   full re-ingest: clone corpus + distill + burst (~3k LLM calls, 1-3h)
  eval     run vector/fts/hybrid/hybrid+rerank and capture numbers to $RAW_OUT
  all      setup -> ingest -> eval

Before running ingest/eval: copy .env.example to .env and add your GITHUB_TOKEN and an LLM API key (matching config.yaml llm.provider).
EOF
    exit 2 ;;
esac
