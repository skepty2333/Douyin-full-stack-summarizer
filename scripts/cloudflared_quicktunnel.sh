#!/usr/bin/env bash
# 启动 Cloudflare 快速隧道（指向本地 MCP 服务），并将随机网址写入 URL_FILE。
# 注意：快速隧道每次启动都会生成新的随机网址，Qoder 的 MCP 配置需随之更新。
set -u
URL_FILE=/root/douyin-bot/cloudflare-tunnel-url.txt

/usr/local/bin/cloudflared tunnel --url http://127.0.0.1:8090 --no-autoupdate 2>&1 \
  | while IFS= read -r line; do
      echo "$line"
      url=$(printf '%s\n' "$line" | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | head -n 1)
      if [ -n "$url" ]; then
        echo "$url" > "$URL_FILE"
      fi
    done
