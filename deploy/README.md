# Hosting the dashboard on an Oracle Always Free VM

The dashboard (`viz/dist/`) is a static site — no backend, no GPU. An Oracle
Cloud **Always Free** instance is more than enough.

## 1. Create the VM

- Shape: `VM.Standard.A1.Flex` (Ampere ARM, up to 4 OCPU / 24 GB free) if your
  region has capacity, otherwise `VM.Standard.E2.1.Micro` (AMD, always
  available, 1 GB RAM — plenty for static files).
- Image: Ubuntu 22.04 or 24.04.
- Networking: assign a public IPv4. Download/attach your SSH key.

## 2. Open the ports — two places, both required

Oracle drops traffic twice if you only fix one:

1. **VCN security list** (Console → Networking → your VCN → Security Lists):
   ingress rules for `0.0.0.0/0` TCP ports **80** and **443** (port 22 is
   already there).
2. **Instance firewall**: Oracle's Ubuntu images ship a netfilter REJECT
   rule that ignores the security list for non-SSH ports. `deploy.sh`
   inserts the ACCEPT rule for you; if you configure things by hand:
   `sudo iptables -I INPUT 5 -p tcp -m multiport --dports 80,443 -j ACCEPT`
   (persist with `iptables-persistent` if you want it to survive reboots).

## 3. Deploy

From the repo root on your machine:

```bash
./deploy/deploy.sh ubuntu@<vm-public-ip>
# → http://<vm-public-ip>
```

With a domain (once you register one — A record pointing at the VM):

```bash
./deploy/deploy.sh ubuntu@<vm-public-ip> your.domain
# Caddy provisions a cert automatically → https://your.domain
```

## 4. Update after a new training run

```bash
uv run python scripts/plot_curves.py --runs runs/tier_a --copy-run-json viz/dist/run.json
./deploy/deploy.sh ubuntu@<vm-public-ip>
```

## Cost note

Everything above (the VM, the public IP, Caddy) is inside the Always Free
allowance. The only thing that could ever cost money is a reserved public IP
if you delete and recreate the VM — ephemeral IPs on a running instance are
free.

## Even simpler alternative

If you don't want a VM at all: `viz/dist/` works on GitHub Pages, Cloudflare
Pages, or Netlify as-is — push `viz/dist` to a `gh-pages` branch or drag the
folder onto Netlify. The Oracle route is here because a real box you can SSH
into is more instructive (and it can host the blog later on the same Caddy).
