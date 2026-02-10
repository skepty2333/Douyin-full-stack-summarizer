"""企业微信消息发送 API"""
import time
import logging
import httpx
from app.config import CORP_ID, CORP_SECRET, AGENT_ID

logger = logging.getLogger(__name__)

_access_token = ""
_token_expires_at = 0


async def get_access_token() -> str:
    """获取 Access Token (带缓存)"""
    global _access_token, _token_expires_at

    if _access_token and time.time() < _token_expires_at - 60:
        return _access_token

    url = "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
    params = {"corpid": CORP_ID, "corpsecret": CORP_SECRET}

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        data = resp.json()

    if data.get("errcode") != 0:
        logger.error(f"Token获取失败: {data}")
        raise Exception(f"Token获取失败: {data.get('errmsg')}")

    _access_token = data["access_token"]
    _token_expires_at = time.time() + data.get("expires_in", 7200)
    return _access_token


async def send_text_message(user_id: str, content: str):
    """发送文本消息 (自动分段)"""
    token = await get_access_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"

    # 分制 2000 字符
    max_len = 2000
    parts = []
    while content:
        if len(content.encode('utf-8')) <= max_len:
            parts.append(content)
            break
        cut = max_len
        while len(content[:cut].encode('utf-8')) > max_len:
            cut -= 1
        last_newline = content[:cut].rfind('\n')
        if last_newline > cut // 2:
            cut = last_newline + 1
        parts.append(content[:cut])
        content = content[cut:]

    async with httpx.AsyncClient() as client:
        for i, part in enumerate(parts):
            if len(parts) > 1:
                part = f"[{i+1}/{len(parts)}]\n{part}" if i > 0 else part

            payload = {
                "touser": user_id,
                "msgtype": "text",
                "agentid": AGENT_ID,
                "text": {"content": part},
            }
            await client.post(url, json=payload)


async def send_markdown_message(user_id: str, content: str):
    """发送 Markdown 消息 (自动分段)"""
    token = await get_access_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"

    MAX_BYTES = 1800
    parts = []
    current_part = ""
    
    for p in content.split('\n'):
        line = p + '\n'
        if len((current_part + line).encode('utf-8')) > MAX_BYTES:
            if current_part:
                parts.append(current_part)
            current_part = line
        else:
            current_part += line
            
    if current_part:
        parts.append(current_part)

    async with httpx.AsyncClient() as client:
        for i, part in enumerate(parts):
            payload = {
                "touser": user_id,
                "msgtype": "markdown",
                "agentid": AGENT_ID,
                "markdown": {"content": part},
            }
            try:
                await client.post(url, json=payload)
            except Exception as e:
                logger.error(f"发送异常: {e}") 
            
            if len(parts) > 1:
                await asyncio.sleep(0.2)


async def upload_temp_media(file_path: str, media_type: str = "file") -> str:
    """上传临时素材"""
    token = await get_access_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/media/upload?access_token={token}&type={media_type}"

    async with httpx.AsyncClient() as client:
        with open(file_path, "rb") as f:
            resp = await client.post(url, files={"media": f})
            data = resp.json()

    if data.get("errcode") and data["errcode"] != 0:
        raise Exception(f"上传失败: {data.get('errmsg')}")

    return data.get("media_id", "")
