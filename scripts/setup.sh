#!/bin/bash
# ========================================
# 抖音视频总结Bot - 一键部署脚本
# Alibaba Cloud Linux 3
# ========================================
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

BOT_DIR="/root/douyin-bot"

echo "======================================"
echo "  抖音视频总结 Bot 部署脚本"
echo "======================================"

# 1. 系统依赖
echo -e "\n${GREEN}[1/7] 安装系统依赖...${NC}"
sudo yum install -y python3 python3-pip python3-devel gcc openssl-devel \
    pango-devel libffi-devel cairo cairo-devel glib2-devel \
    shared-mime-info fontconfig gdk-pixbuf2

# ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${YELLOW}安装 ffmpeg...${NC}"
    sudo yum install -y epel-release 2>/dev/null || true
    sudo yum install -y ffmpeg 2>/dev/null || {
        echo "从静态编译版本安装 ffmpeg..."
        cd /tmp
        curl -L -o ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
        tar xf ffmpeg.tar.xz
        sudo cp ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/
        sudo cp ffmpeg-*-amd64-static/ffprobe /usr/local/bin/
        sudo chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe
        cd -
    }
fi
echo "ffmpeg: $(ffmpeg -version 2>&1 | head -1)"

# 2. 中文字体
echo -e "\n${GREEN}[2/7] 安装中文字体...${NC}"
sudo yum install -y google-noto-sans-cjk-ttc-fonts
# 刷新字体缓存
sudo fc-cache -fv
echo "已安装的中文字体:"
fc-list :lang=zh --format='  %{family}\n' | sort -u | head -5



# 3. 项目目录
echo -e "\n${GREEN}[3/7] 设置项目目录...${NC}"
mkdir -p "$BOT_DIR"
if [ -f "main.py" ]; then
    cp -r *.py requirements.txt "$BOT_DIR/" 2>/dev/null || true
fi
cd "$BOT_DIR"

# 4. 虚拟环境
echo -e "\n${GREEN}[4/7] 创建 Python 虚拟环境...${NC}"
python3.8 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt



# 6. 环境变量
echo -e "\n${GREEN}[6/7] 配置环境变量...${NC}"
if [ ! -f ".env" ]; then
    cat > .env << 'ENVEOF'
# ========== 企业微信 ==========
CORP_ID=your_corp_id
AGENT_ID=1000002
CORP_SECRET=your_corp_secret
CALLBACK_TOKEN=your_callback_token
CALLBACK_AES_KEY=your_encoding_aes_key

# ========== Gemini API (uiuiapi) ==========
GEMINI_API_KEY=your_api_key
GEMINI_BASE_URL=https://api.uiuiapi.com/v1
GEMINI_MODEL=gemini-3-pro-preview-thinking-512

# ========== 服务 ==========
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
TEMP_DIR=/tmp/douyin-bot
LOG_LEVEL=INFO
ENVEOF
    echo -e "${RED}⚠️  请编辑 ${BOT_DIR}/.env 填入实际配置!${NC}"
fi

# 7. systemd 服务
echo -e "\n${GREEN}[7/7] 配置系统服务...${NC}"
sudo tee /etc/systemd/system/douyin-bot.service > /dev/null << SVCEOF
[Unit]
Description=Douyin Video Summarizer Bot
After=network.target

[Service]
Type=simple
User=admin
Group=admin
WorkingDirectory=${BOT_DIR}
EnvironmentFile=${BOT_DIR}/.env
ExecStart=/usr/bin/python3.8 -m uvicorn main:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SVCEOF
sudo systemctl daemon-reload

echo ""
echo "======================================"
echo -e "${GREEN}  ✅ 部署完成!${NC}"
echo "======================================"
echo ""
echo -e "${YELLOW}后续步骤:${NC}"
echo ""
echo "1. 编辑配置:   nano ${BOT_DIR}/.env"
echo ""
echo "2. 启动服务:   sudo systemctl start douyin-bot"
echo "               sudo systemctl enable douyin-bot"
echo ""
echo "3. 查看日志:   journalctl -u douyin-bot -f"
echo ""
echo "4. 企业微信回调URL:  http://你的IP:8080/callback"
echo ""
echo "5. 开放端口:"
echo "   sudo firewall-cmd --add-port=8080/tcp --permanent"
echo "   sudo firewall-cmd --reload"
echo "   + 阿里云安全组开放 8080/TCP"
echo ""
