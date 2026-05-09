#!/usr/bin/env bash
set -euo pipefail

# Manage the OpenAI-compatible Irodori TTS launchd service.

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

LABEL="com.ramo.irodori-tts-openai-api"
SRC_PLIST="$PROJECT_ROOT/launchd/$LABEL.plist"
DST_PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"
LOG_DIR="$PROJECT_ROOT/logs"
STDOUT_LOG="$LOG_DIR/openai-api.out.log"
STDERR_LOG="$LOG_DIR/openai-api.err.log"

DEFAULT_PORT="19841"
PORT="$DEFAULT_PORT"
HEALTH_URL=""
HEALTH_TIMEOUT="420"
DO_SYNC="1"
DO_WAIT="1"

# ---------- helpers ----------

log()  { printf '[irodori-svc] %s\n' "$*"; }
warn() { printf '[irodori-svc] WARN: %s\n' "$*" >&2; }
err()  { printf '[irodori-svc] ERROR: %s\n' "$*" >&2; }

run() {
  printf '[irodori-svc] $ %s\n' "$*"
  "$@"
}

run_ok() {
  # Run a command, ignore non-zero exit but print it.
  printf '[irodori-svc] $ %s\n' "$*"
  if ! "$@"; then
    local rc=$?
    warn "command exited with status $rc (ignored)"
    return 0
  fi
}

resolve_health_url() {
  if [[ -z "$HEALTH_URL" ]]; then
    HEALTH_URL="http://127.0.0.1:${PORT}/v1/health"
  fi
}

is_loaded() {
  launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1
}

wait_unloaded() {
  local timeout="${1:-30}"
  local elapsed=0
  while is_loaded; do
    if (( elapsed >= timeout )); then
      warn "service still loaded after ${timeout}s"
      return 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  return 0
}

wait_healthy() {
  resolve_health_url
  local timeout="$HEALTH_TIMEOUT"
  local elapsed=0
  log "waiting for health: $HEALTH_URL (timeout=${timeout}s)"
  while (( elapsed < timeout )); do
    if curl -fsS --max-time 3 "$HEALTH_URL" >/dev/null 2>&1; then
      log "health OK after ${elapsed}s"
      curl -fsS --max-time 3 "$HEALTH_URL" || true
      printf '\n'
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  err "health check timed out after ${timeout}s: $HEALTH_URL"
  return 1
}

require_src_plist() {
  if [[ ! -f "$SRC_PLIST" ]]; then
    err "source plist not found: $SRC_PLIST"
    return 1
  fi
}

require_dst_plist() {
  if [[ ! -f "$DST_PLIST" ]]; then
    err "destination plist not found: $DST_PLIST (run: $0 install)"
    return 1
  fi
}

# ---------- actions ----------

action_install() {
  require_src_plist
  log "installing plist for $LABEL"
  run mkdir -p "$LOG_DIR"
  run mkdir -p "$(dirname "$DST_PLIST")"
  run cp "$SRC_PLIST" "$DST_PLIST"
  run chmod 644 "$DST_PLIST"
  run plutil -lint "$DST_PLIST"
}

action_sync() {
  log "running uv sync in $PROJECT_ROOT"
  ( cd "$PROJECT_ROOT" && run uv sync )
}

action_stop() {
  log "stopping $LABEL"
  if is_loaded; then
    run_ok launchctl bootout "$DOMAIN/$LABEL"
    if is_loaded && [[ -f "$DST_PLIST" ]]; then
      run_ok launchctl bootout "$DOMAIN" "$DST_PLIST"
    fi
    if ! wait_unloaded 30; then
      warn "service did not fully unload; continuing"
    else
      log "service unloaded"
    fi
  else
    log "service not loaded; nothing to stop"
  fi
}

action_start() {
  require_dst_plist
  log "starting $LABEL"
  if is_loaded; then
    run launchctl kickstart -k "$DOMAIN/$LABEL"
  else
    run launchctl bootstrap "$DOMAIN" "$DST_PLIST"
    run_ok launchctl kickstart -k "$DOMAIN/$LABEL"
  fi
  if [[ "$DO_WAIT" == "1" ]]; then
    wait_healthy
  else
    log "skipping health wait (--no-wait)"
  fi
}

action_restart() {
  action_stop
  action_start
}

action_deploy() {
  action_stop
  if [[ "$DO_SYNC" == "1" ]]; then
    action_sync
  else
    log "skipping uv sync (--no-sync)"
  fi
  action_install
  action_start
}

action_reload() {
  action_stop
  action_install
  action_start
}

action_status() {
  resolve_health_url
  if is_loaded; then
    log "launchd: LOADED ($DOMAIN/$LABEL)"
    run_ok launchctl print "$DOMAIN/$LABEL"
  else
    log "launchd: NOT loaded"
  fi
  log "health URL: $HEALTH_URL"
  if curl -fsS --max-time 3 "$HEALTH_URL" 2>/dev/null; then
    printf '\n'
  else
    warn "health endpoint not reachable"
  fi
}

action_logs() {
  log "tailing $STDOUT_LOG and $STDERR_LOG (Ctrl-C to stop)"
  exec tail -F "$STDOUT_LOG" "$STDERR_LOG"
}

action_test() {
  resolve_health_url
  local base_url
  base_url="${HEALTH_URL%/v1/health}"
  local speech_url="${base_url}/v1/audio/speech"
  local text="Hello world from irodori test."
  local i
  for i in 1 2; do
    log "test request $i -> $speech_url"
    run_ok curl -sS -o /dev/null \
      -D - \
      -w 'time_total=%{time_total}s http_code=%{http_code}\n' \
      -H 'Content-Type: application/json' \
      --max-time 60 \
      -X POST "$speech_url" \
      --data "$(printf '{"model":"irodori","voice":"default","input":"%s","response_format":"wav"}' "$text")" \
      | grep -Ei '^(HTTP/|x-irodori|time_total|http_code)' || true
  done
}

action_uninstall() {
  action_stop
  if [[ -f "$DST_PLIST" ]]; then
    run rm -f "$DST_PLIST"
    log "removed $DST_PLIST"
  else
    log "no destination plist to remove: $DST_PLIST"
  fi
}

# ---------- usage ----------

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] <action>

Actions:
  install      Copy repo plist to ~/Library/LaunchAgents, lint, set perms.
  sync         Run 'uv sync' in the project root.
  stop         bootout the service if loaded; tolerate already-stopped.
  start        bootstrap (or kickstart) and wait for /v1/health.
  restart      stop, then start using already-installed plist.
  deploy       stop, sync, install, start (recommended after code/config update).
  reload       stop, install, start (no uv sync).
  status       Print whether launchd service is loaded and curl /v1/health.
  logs         tail -F stdout and stderr logs.
  test         POST two /v1/audio/speech requests; print time_total + X-Irodori headers.
  uninstall    stop then remove destination plist.
  help         Print this help.

Options (place before action):
  --no-sync           Skip 'uv sync' during deploy.
  --no-wait           Skip waiting for /v1/health after start/restart/deploy/reload.
  --timeout SECONDS   Health-wait timeout. Default: ${HEALTH_TIMEOUT}.
  --health-url URL    Override full health URL.
  --port PORT         Override default port (only affects default health URL).

Defaults:
  LABEL        $LABEL
  SRC_PLIST    $SRC_PLIST
  DST_PLIST    $DST_PLIST
  DOMAIN       $DOMAIN
  HEALTH_URL   http://127.0.0.1:${DEFAULT_PORT}/v1/health

Notes:
  - This script never kills processes by name. Stray processes started
    outside launchd are NOT terminated by 'stop'. Clean those up manually
    if needed (e.g., 'pkill -f openai_api_server.py' at your own risk).
  - 'stop' uses 'launchctl bootout' and never reboots the launchd domain.
EOF
}

# ---------- arg parsing ----------

ACTION=""
while (( $# > 0 )); do
  case "$1" in
    --no-sync)      DO_SYNC="0"; shift ;;
    --no-wait)      DO_WAIT="0"; shift ;;
    --timeout)
      [[ $# -ge 2 ]] || { err "--timeout requires a value"; exit 2; }
      HEALTH_TIMEOUT="$2"; shift 2 ;;
    --health-url)
      [[ $# -ge 2 ]] || { err "--health-url requires a value"; exit 2; }
      HEALTH_URL="$2"; shift 2 ;;
    --port)
      [[ $# -ge 2 ]] || { err "--port requires a value"; exit 2; }
      PORT="$2"; shift 2 ;;
    -h|--help|help)
      usage; exit 0 ;;
    install|sync|stop|start|restart|deploy|reload|status|logs|test|uninstall)
      ACTION="$1"; shift; break ;;
    *)
      err "unknown argument: $1"
      usage >&2
      exit 2 ;;
  esac
done

if [[ -z "$ACTION" ]]; then
  err "no action given"
  usage >&2
  exit 2
fi

if (( $# > 0 )); then
  err "unexpected extra arguments after action: $*"
  exit 2
fi

case "$ACTION" in
  install)   action_install ;;
  sync)      action_sync ;;
  stop)      action_stop ;;
  start)     action_start ;;
  restart)   action_restart ;;
  deploy)    action_deploy ;;
  reload)    action_reload ;;
  status)    action_status ;;
  logs)      action_logs ;;
  test)      action_test ;;
  uninstall) action_uninstall ;;
  *)         usage; exit 2 ;;
esac
