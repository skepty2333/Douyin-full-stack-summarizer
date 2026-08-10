"""Application configuration loaded from environment variables."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数，当前值为 {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} 必须大于等于 {minimum}")
    return value


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字，当前值为 {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} 必须大于等于 {minimum}")
    return value


# 企业微信
CORP_ID = os.getenv("CORP_ID", "your_corp_id")
AGENT_ID = _env_int("AGENT_ID", 1000002)
CORP_SECRET = os.getenv("CORP_SECRET", "your_corp_secret")
CALLBACK_TOKEN = os.getenv("CALLBACK_TOKEN", "your_callback_token")
CALLBACK_AES_KEY = os.getenv("CALLBACK_AES_KEY", "your_encoding_aes_key")

# 阿里云百炼（华北 2 / 北京）
# 生产环境建议设置为 Workspace 专用 OpenAI 兼容地址。
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
).rstrip("/")
DASHSCOPE_NATIVE_BASE_URL = os.getenv(
    "DASHSCOPE_NATIVE_BASE_URL",
    "https://dashscope.aliyuncs.com/api/v1",
).rstrip("/")

# 稳定别名通常有更高限流；严格回归场景可通过环境变量换成日期快照。
ALIYUN_ASR_MODEL = os.getenv(
    "ALIYUN_ASR_MODEL", "qwen-audio-3.0-asr-flash-filetrans"
)
ALIYUN_ASR_FALLBACK_MODEL = os.getenv(
    "ALIYUN_ASR_FALLBACK_MODEL", "qwen3-asr-flash"
)
ALIYUN_VISUAL_MODEL = os.getenv(
    "ALIYUN_VISUAL_MODEL",
    # Backward-compatible migration path for existing deployments.
    os.getenv("ALIYUN_DRAFT_MODEL", "qwen3.8-max"),
)
# Deprecated code-level alias; new deployments should use ALIYUN_VISUAL_MODEL.
ALIYUN_DRAFT_MODEL = ALIYUN_VISUAL_MODEL
ALIYUN_RESEARCH_MODEL = os.getenv("ALIYUN_RESEARCH_MODEL", "qwen3.7-plus")
ALIYUN_FINAL_MODEL = os.getenv("ALIYUN_FINAL_MODEL", "qwen3.8-max")
ALIYUN_TAG_MODEL = os.getenv("ALIYUN_TAG_MODEL", "qwen3.7-flash")

AI_REQUEST_TIMEOUT_SECONDS = _env_float("AI_REQUEST_TIMEOUT_SECONDS", 240.0, 1.0)
AI_MAX_RETRIES = _env_int("AI_MAX_RETRIES", 3, 0)
AI_MAX_CONCURRENCY = _env_int("AI_MAX_CONCURRENCY", 3)
ASR_SEGMENT_SECONDS = _env_int("ASR_SEGMENT_SECONDS", 240, 30)
ASR_MAX_FILE_MB = _env_int("ASR_MAX_FILE_MB", 7)
ASR_FILE_POLL_INTERVAL_SECONDS = _env_float(
    "ASR_FILE_POLL_INTERVAL_SECONDS", 2.0, 0.1
)
ASR_FILE_TIMEOUT_SECONDS = _env_int("ASR_FILE_TIMEOUT_SECONDS", 1800, 10)

# 服务与运行时容量
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = _env_int("SERVER_PORT", 8080)
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = _env_int("MCP_PORT", 8090)
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/douyin-bot")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
KNOWLEDGE_DB_PATH = os.getenv("KNOWLEDGE_DB_PATH", str(PROJECT_ROOT / "knowledge.db"))
KNOWLEDGE_ASSETS_DIR = os.getenv(
    "KNOWLEDGE_ASSETS_DIR",
    str(Path(KNOWLEDGE_DB_PATH).expanduser().parent / "knowledge_assets"),
)
MAX_CONCURRENT_JOBS = _env_int("MAX_CONCURRENT_JOBS", 2)
JOB_TIMEOUT_SECONDS = _env_int("JOB_TIMEOUT_SECONDS", 3600, 60)
DOWNLOAD_TIMEOUT_SECONDS = _env_int("DOWNLOAD_TIMEOUT_SECONDS", 600, 30)
TEMP_FILE_TTL_HOURS = _env_int("TEMP_FILE_TTL_HOURS", 24)


def validate_ai_config() -> list[str]:
    """Return local AI configuration problems without making a paid API call."""
    problems: list[str] = []
    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY.startswith("your_"):
        problems.append("DASHSCOPE_API_KEY 未配置")
    if not DASHSCOPE_BASE_URL.startswith("https://"):
        problems.append("DASHSCOPE_BASE_URL 必须使用 https://")
    if not all(
        (
            ALIYUN_ASR_MODEL,
            ALIYUN_ASR_FALLBACK_MODEL,
            ALIYUN_VISUAL_MODEL,
            ALIYUN_RESEARCH_MODEL,
            ALIYUN_FINAL_MODEL,
            ALIYUN_TAG_MODEL,
        )
    ):
        problems.append("一个或多个阿里云模型名为空")
    if not DASHSCOPE_NATIVE_BASE_URL.startswith("https://"):
        problems.append("DASHSCOPE_NATIVE_BASE_URL 必须使用 https://")
    return problems
