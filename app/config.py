"""
配置文件 - 部署时需要修改这些值
"""
import os

# ========================
# 企业微信配置
# ========================
CORP_ID = os.getenv("CORP_ID", "your_corp_id")
AGENT_ID = int(os.getenv("AGENT_ID", "1000002"))
CORP_SECRET = os.getenv("CORP_SECRET", "your_corp_secret")
CALLBACK_TOKEN = os.getenv("CALLBACK_TOKEN", "your_callback_token")
CALLBACK_AES_KEY = os.getenv("CALLBACK_AES_KEY", "your_encoding_aes_key")

# ========================
# AI API 配置 (uiuiapi 聚合平台)
# ========================
# 通用 API 端点
API_BASE_URL = os.getenv("API_BASE_URL", "https://sg.uiuiapi.com/v1")

# 阶段1: Gemini - 初始转写+总结
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview-thinking-512")

# 阶段2: DeepSeek Reasoner - 深度审视与提问
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your_deepseek_api_key")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v3.2-thinking")

# 阶段3: Sonnet 4.5 - 联网搜索 + 最终交付
SONNET_API_KEY = os.getenv("SONNET_API_KEY", "your_sonnet_api_key")
SONNET_MODEL = os.getenv("SONNET_MODEL", "claude-sonnet-4-5-20250929-thinking")

# ========================
# 服务配置
# ========================
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8080"))
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/douyin-bot")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
KNOWLEDGE_DB_PATH = os.getenv("KNOWLEDGE_DB_PATH", "/home/admin/douyin-bot/knowledge.db")
