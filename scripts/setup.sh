#!/usr/bin/env bash
# 抖音视频总结 Bot - Alibaba Cloud Linux 3 部署脚本
set -Eeuo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

BOT_DIR="/root/douyin-bot"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${BOT_DIR}/venv"

if [[ ${EUID} -ne 0 ]]; then
    echo -e "${RED}请使用 root 运行：sudo bash scripts/setup.sh${NC}" >&2
    exit 1
fi

if [[ "$(readlink -f -- "${SOURCE_DIR}")" != "${BOT_DIR}" ]]; then
    echo -e "${RED}项目必须位于 ${BOT_DIR}，当前目录为 ${SOURCE_DIR}${NC}" >&2
    exit 1
fi

echo "======================================"
echo "  抖音视频总结 Bot 部署脚本"
echo "======================================"

echo -e "\n${GREEN}[1/6] 安装系统依赖...${NC}"
yum install -y \
    python3.11 python3.11-pip python3.11-devel gcc openssl-devel \
    pango-devel libffi-devel cairo cairo-devel glib2-devel \
    shared-mime-info fontconfig gdk-pixbuf2 curl xz

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo -e "${YELLOW}安装 ffmpeg...${NC}"
    yum install -y epel-release 2>/dev/null || true
    if ! yum install -y ffmpeg 2>/dev/null; then
        echo "软件源中未找到 ffmpeg，改用静态版本..."
        FFMPEG_TMP_DIR="$(mktemp -d /tmp/douyin-ffmpeg.XXXXXX)"
        trap 'rm -rf -- "${FFMPEG_TMP_DIR}"' EXIT
        curl --fail --location --retry 3 \
            --output "${FFMPEG_TMP_DIR}/ffmpeg.tar.xz" \
            https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
        tar -xf "${FFMPEG_TMP_DIR}/ffmpeg.tar.xz" -C "${FFMPEG_TMP_DIR}"
        install -m 0755 "${FFMPEG_TMP_DIR}"/ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ffmpeg
        install -m 0755 "${FFMPEG_TMP_DIR}"/ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ffprobe
    fi
fi
echo "ffmpeg: $(ffmpeg -version 2>&1 | head -n 1)"

echo -e "\n${GREEN}[2/6] 安装中文字体...${NC}"
yum install -y google-noto-sans-cjk-ttc-fonts
fc-cache -f

echo -e "\n${GREEN}[3/6] 创建 Python 3.11 虚拟环境...${NC}"
python3.11 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -r "${BOT_DIR}/requirements.txt"
"${VENV_DIR}/bin/python" -m pip check
"${VENV_DIR}/bin/python" -m compileall -q \
    "${BOT_DIR}/app" "${BOT_DIR}/main.py" "${BOT_DIR}/mcp_server.py"
"${VENV_DIR}/bin/python" -c \
    'import fastapi, httpx, uvicorn, weasyprint, mcp, pydantic'
echo "Python: $("${VENV_DIR}/bin/python" --version 2>&1)"

echo -e "\n${GREEN}[4/6] 配置环境变量...${NC}"
if [[ ! -f "${BOT_DIR}/.env" ]]; then
    install -m 0600 /dev/null "${BOT_DIR}/.env"
    cat > "${BOT_DIR}/.env" <<'ENVEOF'
# 企业微信
CORP_ID=your_corp_id
AGENT_ID=1000002
CORP_SECRET=your_corp_secret
CALLBACK_TOKEN=your_callback_token
CALLBACK_AES_KEY=your_encoding_aes_key

# 阿里云百炼（华北 2 / 北京）
# API Key 与接入地址必须属于同一地域、同一 Workspace。
DASHSCOPE_API_KEY=
DASHSCOPE_BASE_URL=https://YOUR_WORKSPACE_ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
DASHSCOPE_NATIVE_BASE_URL=https://YOUR_WORKSPACE_ID.cn-beijing.maas.aliyuncs.com/api/v1

# 全阿里模型管线
ALIYUN_ASR_MODEL=qwen-audio-3.0-asr-flash-filetrans
ALIYUN_ASR_FALLBACK_MODEL=qwen3-asr-flash
ALIYUN_VISUAL_MODEL=qwen3.8-max
ALIYUN_RESEARCH_MODEL=qwen3.7-plus
ALIYUN_FINAL_MODEL=qwen3.8-max
ALIYUN_TAG_MODEL=qwen3.7-flash

# AI 稳定性与容量
AI_REQUEST_TIMEOUT_SECONDS=240
AI_MAX_RETRIES=3
AI_MAX_CONCURRENCY=3
ASR_SEGMENT_SECONDS=240
ASR_MAX_FILE_MB=7
ASR_FILE_POLL_INTERVAL_SECONDS=2
ASR_FILE_TIMEOUT_SECONDS=1800
MAX_CONCURRENT_JOBS=2
JOB_TIMEOUT_SECONDS=3600
DOWNLOAD_TIMEOUT_SECONDS=600

# Bot 服务（仅供本机 Nginx 反向代理）
SERVER_HOST=127.0.0.1
SERVER_PORT=8080
TEMP_DIR=/tmp/douyin-bot
TEMP_FILE_TTL_HOURS=24
LOG_LEVEL=INFO
KNOWLEDGE_DB_PATH=/root/douyin-bot/knowledge.db
KNOWLEDGE_ASSETS_DIR=/root/douyin-bot/knowledge_assets

# MCP 默认仅监听本机；经带认证的反向代理或受控隧道访问。
MCP_HOST=127.0.0.1
MCP_PORT=8090
ENVEOF
    echo -e "${RED}请先配置 ${BOT_DIR}/.env，再启动服务。${NC}"
else
    echo "保留现有 ${BOT_DIR}/.env。"
fi
chmod 0600 "${BOT_DIR}/.env"
install -d -m 0700 /tmp/douyin-bot
install -d -m 0700 "${BOT_DIR}/knowledge_assets"

echo -e "\n${GREEN}[5/6] 安装 Bot 与 MCP systemd 服务...${NC}"
install -m 0644 "${BOT_DIR}/deployment/douyin-bot.service" /etc/systemd/system/douyin-bot.service
install -m 0644 "${BOT_DIR}/deployment/douyin-mcp.service" /etc/systemd/system/douyin-mcp.service
systemctl daemon-reload
systemd-analyze verify \
    /etc/systemd/system/douyin-bot.service \
    /etc/systemd/system/douyin-mcp.service
systemctl enable douyin-bot.service douyin-mcp.service

echo -e "\n${GREEN}[6/6] 完成部署...${NC}"
echo "======================================"
echo -e "${GREEN}部署完成。${NC}"
echo "======================================"
echo "1. 配置文件：vi ${BOT_DIR}/.env"
echo "2. 启动服务：systemctl start douyin-bot douyin-mcp"
echo "3. 查看日志：journalctl -u douyin-bot -u douyin-mcp -f"
echo "4. 先配置并验证 Nginx/HTTPS，再使用：https://你的域名/callback"
echo "5. 仅向公网开放反向代理的 80/443 端口，不要开放 8080/8090。"
