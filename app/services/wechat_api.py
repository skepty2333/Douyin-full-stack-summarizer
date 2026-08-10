"""Reliable asynchronous Enterprise WeChat application messaging client."""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from app.config import AGENT_ID, CORP_ID, CORP_SECRET


logger = logging.getLogger(__name__)
_TOKEN_ERROR_CODES = {40001, 40014, 42001}
_RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}
_PARTIAL_DELIVERY_FIELDS = (
    "invaliduser",
    "invalidparty",
    "invalidtag",
    "unlicenseduser",
)

_access_token = ""
_token_expires_at = 0.0
_token_lock = asyncio.Lock()
_http_client: Optional[httpx.AsyncClient] = None


class WeChatAPIError(RuntimeError):
    pass


def _client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
    return _http_client


async def close_wechat_client() -> None:
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


async def _request_json(
    method: str,
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    json: Optional[dict[str, Any]] = None,
    max_retries: int = 2,
) -> dict[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            response = await _client().request(method, url, params=params, json=json)
            if response.status_code in _RETRYABLE_HTTP_CODES and attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise WeChatAPIError("企业微信返回格式异常")
            return data
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            if attempt >= max_retries:
                raise WeChatAPIError(f"企业微信网络请求失败: {type(exc).__name__}") from exc
            await asyncio.sleep(2 ** attempt)
        except ValueError as exc:
            raise WeChatAPIError("企业微信返回了无效 JSON") from exc
    raise WeChatAPIError("企业微信请求异常退出")


async def get_access_token(force_refresh: bool = False) -> str:
    """Get and cache an access token, preventing concurrent refresh storms."""
    global _access_token, _token_expires_at
    if not force_refresh and _access_token and time.time() < _token_expires_at - 60:
        return _access_token

    async with _token_lock:
        if not force_refresh and _access_token and time.time() < _token_expires_at - 60:
            return _access_token
        data = await _request_json(
            "GET",
            "https://qyapi.weixin.qq.com/cgi-bin/gettoken",
            params={"corpid": CORP_ID, "corpsecret": CORP_SECRET},
        )
        if data.get("errcode") != 0:
            raise WeChatAPIError(f"Token 获取失败: {data.get('errmsg', 'unknown error')}")
        token = data.get("access_token")
        if not isinstance(token, str) or not token:
            raise WeChatAPIError("Token 响应缺少 access_token")
        _access_token = token
        _token_expires_at = time.time() + int(data.get("expires_in", 7200))
        return token


async def _send_payload(payload: dict[str, Any]) -> None:
    global _access_token, _token_expires_at
    for token_attempt in range(2):
        token = await get_access_token(force_refresh=token_attempt > 0)
        data = await _request_json(
            "POST",
            "https://qyapi.weixin.qq.com/cgi-bin/message/send",
            params={"access_token": token},
            json=payload,
        )
        errcode = int(data.get("errcode", -1))
        if errcode == 0:
            rejected = {
                field: data[field]
                for field in _PARTIAL_DELIVERY_FIELDS
                if data.get(field)
            }
            if rejected:
                raise WeChatAPIError(
                    "企业微信返回成功但存在未投递目标: "
                    + ", ".join(sorted(rejected))
                )
            logger.info(
                "企业微信消息发送成功: user=%s type=%s",
                payload.get("touser", ""),
                payload.get("msgtype", ""),
            )
            return
        if errcode in _TOKEN_ERROR_CODES and token_attempt == 0:
            _access_token = ""
            _token_expires_at = 0
            continue
        raise WeChatAPIError(
            f"企业微信发送失败: errcode={errcode}, errmsg={data.get('errmsg', 'unknown error')}"
        )
    raise WeChatAPIError("企业微信发送失败: Token 刷新后仍不可用")


async def send_text_message(user_id: str, content: str) -> None:
    """Send text, splitting by UTF-8 byte length."""
    max_bytes = 1900
    parts: list[str] = []
    remaining = content
    while remaining:
        if len(remaining.encode("utf-8")) <= max_bytes:
            parts.append(remaining)
            break
        cut = min(len(remaining), max_bytes)
        while cut > 1 and len(remaining[:cut].encode("utf-8")) > max_bytes:
            cut -= 1
        newline = remaining[:cut].rfind("\n")
        if newline > cut // 2:
            cut = newline + 1
        parts.append(remaining[:cut])
        remaining = remaining[cut:]

    for index, part in enumerate(parts):
        if len(parts) > 1:
            part = f"[{index + 1}/{len(parts)}]\n{part}"
        await _send_payload(
            {
                "touser": user_id,
                "msgtype": "text",
                "agentid": AGENT_ID,
                "text": {"content": part},
            }
        )


async def send_markdown_message(user_id: str, content: str) -> None:
    """Send Markdown in byte-safe chunks."""
    max_bytes = 1800
    parts: list[str] = []
    current = ""
    for raw_line in content.splitlines(keepends=True):
        line = raw_line or "\n"
        if len((current + line).encode("utf-8")) > max_bytes and current:
            parts.append(current)
            current = ""
        if len(line.encode("utf-8")) > max_bytes:
            if current:
                parts.append(current)
                current = ""
            encoded_part = ""
            for char in line:
                if len((encoded_part + char).encode("utf-8")) > max_bytes:
                    parts.append(encoded_part)
                    encoded_part = char
                else:
                    encoded_part += char
            current = encoded_part
        else:
            current += line
    if current:
        parts.append(current)

    for part in parts:
        await _send_payload(
            {
                "touser": user_id,
                "msgtype": "markdown",
                "agentid": AGENT_ID,
                "markdown": {"content": part},
            }
        )
        if len(parts) > 1:
            await asyncio.sleep(0.2)


async def upload_temp_media(file_path: str, media_type: str = "file") -> str:
    """Upload a temporary media file and validate the returned media ID."""
    path = Path(file_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise WeChatAPIError("待上传文件不存在或为空")

    global _access_token, _token_expires_at
    for token_attempt in range(2):
        token = await get_access_token(force_refresh=token_attempt > 0)
        with path.open("rb") as media:
            try:
                response = await _client().post(
                    "https://qyapi.weixin.qq.com/cgi-bin/media/upload",
                    params={"access_token": token, "type": media_type},
                    files={"media": (path.name, media)},
                )
                response.raise_for_status()
                data = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise WeChatAPIError(f"企业微信文件上传请求失败: {type(exc).__name__}") from exc
        errcode = int(data.get("errcode", 0))
        if errcode in _TOKEN_ERROR_CODES and token_attempt == 0:
            _access_token = ""
            _token_expires_at = 0
            continue
        if errcode != 0:
            raise WeChatAPIError(
                f"文件上传失败: errcode={errcode}, errmsg={data.get('errmsg', 'unknown error')}"
            )
        media_id = data.get("media_id")
        if not isinstance(media_id, str) or not media_id:
            raise WeChatAPIError("文件上传响应缺少 media_id")
        return media_id
    raise WeChatAPIError("文件上传失败: Token 刷新后仍不可用")


async def send_file_message(user_id: str, media_id: str) -> None:
    if not media_id:
        raise WeChatAPIError("media_id 为空")
    await _send_payload(
        {
            "touser": user_id,
            "msgtype": "file",
            "agentid": AGENT_ID,
            "file": {"media_id": media_id},
        }
    )
