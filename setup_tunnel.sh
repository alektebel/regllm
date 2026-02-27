#!/bin/bash
# One-time Cloudflare Named Tunnel setup for regllm.xyz
# Run this once on any machine you want to expose.
#
# Usage: ./setup_tunnel.sh

set -e
cd "$(dirname "$0")"

TUNNEL_NAME="regllm"
DOMAIN="regllm.xyz"
PORT=7860

# ─── Check cloudflared ────────────────────────────────────────────────────────
if ! command -v cloudflared &>/dev/null && [ ! -x "$HOME/bin/cloudflared" ]; then
  echo "cloudflared not found. Installing..."
  ARCH=$(uname -m)
  if [ "$ARCH" = "x86_64" ]; then
    URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
  elif [ "$ARCH" = "aarch64" ]; then
    URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
  else
    echo "ERROR: Unsupported architecture: $ARCH"
    exit 1
  fi
  mkdir -p "$HOME/bin"
  curl -L "$URL" -o "$HOME/bin/cloudflared"
  chmod +x "$HOME/bin/cloudflared"
  export PATH="$HOME/bin:$PATH"
  echo "cloudflared installed at $HOME/bin/cloudflared"
fi

CLOUDFLARED=$(command -v cloudflared 2>/dev/null || echo "$HOME/bin/cloudflared")

# ─── Authenticate ─────────────────────────────────────────────────────────────
echo ""
echo "Step 1/3 — Authenticating with Cloudflare"
echo "  A browser window will open. Log in and select the zone for $DOMAIN"
echo ""
"$CLOUDFLARED" login

# ─── Create tunnel ────────────────────────────────────────────────────────────
echo ""
echo "Step 2/3 — Creating tunnel '$TUNNEL_NAME'"
TUNNEL_OUTPUT=$("$CLOUDFLARED" tunnel create "$TUNNEL_NAME" 2>&1)
echo "$TUNNEL_OUTPUT"

TUNNEL_ID=$(echo "$TUNNEL_OUTPUT" | grep -oP '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | head -1)
if [ -z "$TUNNEL_ID" ]; then
  # Tunnel may already exist — fetch the ID
  TUNNEL_ID=$("$CLOUDFLARED" tunnel list | grep "$TUNNEL_NAME" | awk '{print $1}')
fi

if [ -z "$TUNNEL_ID" ]; then
  echo "ERROR: Could not determine tunnel ID. Check output above."
  exit 1
fi

echo "  Tunnel ID: $TUNNEL_ID"

# ─── Route DNS ────────────────────────────────────────────────────────────────
echo ""
echo "Step 3/3 — Routing $DOMAIN → tunnel '$TUNNEL_NAME'"
"$CLOUDFLARED" tunnel route dns "$TUNNEL_NAME" "$DOMAIN"

# ─── Write config ─────────────────────────────────────────────────────────────
CONFIG_FILE="$HOME/.cloudflared/config.yml"
mkdir -p "$HOME/.cloudflared"

cat > "$CONFIG_FILE" << EOF
tunnel: $TUNNEL_NAME
credentials-file: $HOME/.cloudflared/$TUNNEL_ID.json

ingress:
  - hostname: $DOMAIN
    service: http://localhost:$PORT
  - service: http_status:404
EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete"
echo "  Tunnel  : $TUNNEL_NAME ($TUNNEL_ID)"
echo "  Domain  : https://$DOMAIN"
echo "  Config  : $CONFIG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Run the app:  ./launch_ui.sh"
echo ""
