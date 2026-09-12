#!/usr/bin/env bash
# Deploy viz/dist to a fresh Ubuntu VM (Oracle Always Free works) running Caddy.
#   ./deploy/deploy.sh ubuntu@<vm-public-ip> [domain]
set -euo pipefail

HOST="${1:?usage: deploy.sh <ssh-host> [domain]}"
DOMAIN="${2:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"

rsync -az --delete "$HERE/../viz/dist/" "$HOST:grpo-dist/"
scp -q "$HERE/Caddyfile" "$HOST:grpo-Caddyfile"

ssh "$HOST" "DOMAIN='$DOMAIN' bash -s" <<'EOF'
set -euo pipefail
command -v caddy >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq caddy; }

# Oracle Ubuntu images carry a netfilter REJECT rule that drops 80/443 even
# after the VCN security list allows them.
sudo iptables -C INPUT -p tcp -m multiport --dports 80,443 -j ACCEPT 2>/dev/null \
  || sudo iptables -I INPUT 5 -p tcp -m multiport --dports 80,443 -j ACCEPT

sudo mkdir -p /var/www/grpo
sudo cp -r ~/grpo-dist/. /var/www/grpo/
if [ -n "$DOMAIN" ]; then
  sed "s/^:80/$DOMAIN/" ~/grpo-Caddyfile | sudo tee /etc/caddy/Caddyfile >/dev/null
else
  sudo cp ~/grpo-Caddyfile /etc/caddy/Caddyfile
fi
sudo systemctl reload caddy 2>/dev/null || sudo systemctl restart caddy
echo "deployed: http://$(curl -s ifconfig.me || echo '<vm-ip>')${DOMAIN:+  (https after DNS propagates: https://$DOMAIN)}"
EOF
