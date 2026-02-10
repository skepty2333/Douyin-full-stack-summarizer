"""
抖音视频知识总结 Bot - 主服务

交互流程:
1. 用户发送链接 -> Bot 回复收到
2. 用户补充要求 -> Bot 更新任务
3. 超时或显式开始 -> 执行解析下载总结
4. 完成 -> 发送文本/PDF/文件
"""
import asyncio
import logging
import os
import time
import random
import string
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional, Dict

from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse

from app.config import (
    CORP_ID, CALLBACK_TOKEN, CALLBACK_AES_KEY,
    TEMP_DIR, LOG_LEVEL, AGENT_ID, SERVER_HOST, SERVER_PORT
)
from app.utils.wechat_crypto import WXBizMsgCrypt
from app.services.wechat_api import send_text_message, send_markdown_message, upload_temp_media
from app.services.douyin_parser import (
    extract_url_from_text, extract_user_requirement,
    resolve_and_download, extract_audio, cleanup_files,
)
from app.services.ai_summarizer import summarize_with_audio
from app.services.pdf_generator import generate_pdf
from app.database.knowledge_store import KnowledgeStore, KnowledgeEntry, extract_tags_from_markdown

# 初始化
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("douyin-bot")

crypto = WXBizMsgCrypt(CALLBACK_TOKEN, CALLBACK_AES_KEY, CORP_ID)
knowledge_db = KnowledgeStore()

# 消息去重
_processed_msgs: Dict[str, float] = {}
MSG_DEDUP_TTL = 300


def generate_video_code() -> str:
    """生成5位随机视频码"""
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choices(chars, k=5))


# 会话管理
WAIT_SECONDS = 120  # 等待用户输入要求的时间


@dataclass
class PendingTask:
    """待处理任务"""
    user_id: str
    share_url: str
    share_text: str
    extra_requirement: str = ""
    created_at: float = field(default_factory=time.time)
    timer_task: Optional[asyncio.Task] = None
    processing: bool = False
    
    # Duplicate Check State
    waiting_for_dup_confirm: bool = False
    dup_video_code: str = ""
    dup_timestamp: str = ""
    parsed_title: str = ""
    parsed_author: str = ""
    parsed_video_id: str = ""
    parsed_video_path: str = ""


_pending: Dict[str, PendingTask] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(TEMP_DIR, exist_ok=True)
    # 初始化/重置数据库
    # knowledge_db._init_db() # 已经在 __init__ 中调用，但为了强制重置 schema，这里显式调用一次?
    # 不，_init_db 在实例化时调用。如果 knowledge_store.py 修改了 _init_db 逻辑，重启服务时会自动执行。
    logger.info("🚀 Bot 启动")
    yield
    logger.info("Bot 关闭")


app = FastAPI(title="抖音视频总结Bot", lifespan=lifespan)


@app.get("/callback")
async def verify_callback(
    msg_signature: str = Query(...),
    timestamp: str = Query(...),
    nonce: str = Query(...),
    echostr: str = Query(...),
):
    """GET - 验证URL有效性"""
    try:
        echo = crypto.verify_url(msg_signature, timestamp, nonce, echostr)
        logger.info("URL验证成功")
        return PlainTextResponse(content=echo)
    except Exception as e:
        logger.error(f"URL验证失败: {e}")
        return PlainTextResponse(content="error", status_code=403)


@app.post("/callback")
async def receive_message(
    request: Request,
    msg_signature: str = Query(...),
    timestamp: str = Query(...),
    nonce: str = Query(...),
):
    """POST - 接收消息"""
    body = (await request.body()).decode("utf-8")

    try:
        xml_text = crypto.decrypt_msg(body, msg_signature, timestamp, nonce)
        xml_root = ET.fromstring(xml_text)
        msg_type = xml_root.find("MsgType").text
        from_user = xml_root.find("FromUserName").text

        # 简单的去重
        msg_id = (xml_root.find("MsgId").text or "") if xml_root.find("MsgId") is not None else ""
        create_time = (xml_root.find("CreateTime").text or "") if xml_root.find("CreateTime") is not None else ""
        dedup_key = f"{msg_id}_{create_time}"
        now = time.time()
        if dedup_key in _processed_msgs and now - _processed_msgs[dedup_key] < MSG_DEDUP_TTL:
            return PlainTextResponse(content="success")
        _processed_msgs[dedup_key] = now
        
        # 清理过期
        for k in [k for k, v in _processed_msgs.items() if now - v > MSG_DEDUP_TTL]:
            del _processed_msgs[k]

        if msg_type == "text":
            content = xml_root.find("Content").text or ""
            logger.info(f"📩 {from_user}: {content[:50]}")
            asyncio.create_task(handle_message(from_user, content))
        else:
            logger.info(f"忽略消息类型: {msg_type}")

    except Exception as e:
        logger.error(f"处理回调异常: {e}", exc_info=True)

    return PlainTextResponse(content="success")


async def handle_message(user_id: str, content: str):
    """消息路由"""
    try:
        content_stripped = content.strip()
        
        # 情况1: 用户在等待列表中
        if user_id in _pending:
            pending = _pending[user_id]
            
            # A. 正在等待重复确认 (waiting_for_dup_confirm)
            if pending.waiting_for_dup_confirm:
                if content_stripped in ("覆盖", "Overwrite"):
                    if pending.timer_task and not pending.timer_task.done():
                        pending.timer_task.cancel()
                    
                    # 覆盖：删除旧记录，生成新记录 (新视频码)
                    try:
                        knowledge_db.delete_by_video_code(pending.dup_video_code)
                        logger.info(f"覆盖操作: 已删除旧记录 {pending.dup_video_code}")
                    except Exception as e:
                        logger.error(f"覆盖删除失败: {e}")

                    new_code = generate_video_code()
                    await send_text_message(user_id, f"确认覆盖，视频码：{new_code}，开始处理...")
                    await _execute_summary_task(user_id, pending, reuse_video_code=new_code)
                    
                elif content_stripped in ("新增", "New"):
                    if pending.timer_task and not pending.timer_task.done():
                        pending.timer_task.cancel()
                    new_code = generate_video_code()
                    await send_text_message(user_id, f"确认新增，视频码：{new_code}，开始处理...")
                    # 新增：使用新的 video_code
                    await _execute_summary_task(user_id, pending, reuse_video_code=new_code)
                    
                elif content_stripped in ("取消", "Cancel"):
                    if pending.timer_task and not pending.timer_task.done():
                        pending.timer_task.cancel()
                    await send_text_message(user_id, "收到，取消处理。")
                    _cleanup_pending_files(pending)
                    del _pending[user_id]
                    
                else:
                    await send_text_message(user_id, "输入“覆盖”、“新增”或“取消”。")
                return

            # B. 正常等待 (尚未开始处理)
            if not pending.processing:
                # 取消指令
                if content_stripped in ("取消", "Cancel"):
                    if pending.timer_task and not pending.timer_task.done():
                        pending.timer_task.cancel()
                    await send_text_message(user_id, "收到，取消处理。")
                    _cleanup_pending_files(pending)
                    del _pending[user_id]
                    return

                # 检查是否新链接 (打断当前，开始新的)
                new_url = extract_url_from_text(content)
                if new_url:
                    if pending.timer_task and not pending.timer_task.done():
                        pending.timer_task.cancel()
                    # 清理旧文件
                    _cleanup_pending_files(pending)
                    del _pending[user_id]
                    await _start_new_task(user_id, content, new_url)
                    return

                # 立即开始
                if content_stripped.lower() in ("开始", "start", "ok", "好"):
                    if pending.timer_task and not pending.timer_task.done():
                        pending.timer_task.cancel()
                    await send_text_message(user_id, "正在开始处理...")  # Added immediate feedback
                    await _process_task_init(user_id) # 立即触发处理流程 (含查重)
                    return

                # 补充要求
                pending.extra_requirement = content_stripped
                logger.info(f"📝 {user_id} 补充: {content[:30]}")
                # 重新计时? 用户说 "等待两分钟"，通常是指从第一条消息开始。
                # 但如果在最后一秒补充了要求，是否应该延时？
                # "2分钟内可补充..."，所以这里保持原定时器，不重置，除非为了更好体验。
                # 简单起见，不重置定时器，只更新要求。
                # 但如果用户希望确认收到，可以回个简单的确认？
                # 用户没要求回确认，只说 "回复前两个要求..." 是指最终回复。
                return

        # 情况2: 新链接
        url = extract_url_from_text(content)
        if url:
            # 如果之前有任务正在处理中 (processing=True)，是否允许插入？
            # "视频正在处理中，请稍候..."
            # 但如果 user_id 不在 _pending (说明处理完了)，则直接开始。
            if user_id in _pending and _pending[user_id].processing:
                await send_text_message(user_id, "视频正在处理中，请稍候...")
                return

            await _start_new_task(user_id, content, url)
            return

        # 情况3: 帮助信息
        await send_text_message(
            user_id,
            "收到，发送“开始”立即处理，“取消”以取消操作，或输入具体要求。2分钟后默认处理。"
        )

    except Exception as e:
        logger.error(f"handle_message异常: {e}", exc_info=True)
        try:
            await send_text_message(user_id, "❌ 系统繁忙")
        except: pass


async def _start_new_task(user_id: str, content: str, url: str):
    """创建新任务"""
    inline_req = extract_user_requirement(content, url)
    task = PendingTask(user_id=user_id, share_url=url, share_text=inline_req)
    _pending[user_id] = task

    await send_text_message(user_id, "收到，发送“开始”立即处理，“取消”以取消操作，或输入具体要求。2分钟后默认处理。")
    task.timer_task = asyncio.create_task(_wait_then_process(user_id))


async def _wait_then_process(user_id: str):
    """超时自动处理"""
    try:
        await asyncio.sleep(WAIT_SECONDS)
        if user_id in _pending:
            task = _pending[user_id]
            
            # 如果是在等待重复确认状态超时
            if task.waiting_for_dup_confirm:
                logger.info(f"⏰ {user_id} 重复确认超时，默认取消")
                await send_text_message(user_id, "两分钟超时，默认取消处理。")
                _cleanup_pending_files(task)
                del _pending[user_id]
                return

            # 正常超时，开始处理
            if not task.processing:
                logger.info(f"⏰ {user_id} 超时，开始处理")
                await _process_task_init(user_id)
                
    except asyncio.CancelledError:
        pass


async def _process_task_init(user_id: str):
    """任务处理入口: 解析 -> 查重 -> (执行 或 等待确认)"""
    if user_id not in _pending: return
    task = _pending[user_id]
    task.processing = True 

    try:
        # 1. 解析下载 (获取 Title, Author, ID)
        # 注意: 这里会下载视频，稍微有点耗时，但必须下载解析才能知道 Title/Author。
        # 如果是大V账号，可能解析耗时较长。
        video_info = await resolve_and_download(task.share_url)
        
        task.parsed_video_id = video_info["video_id"]
        task.parsed_title = video_info["title"] or "未知标题"
        task.parsed_author = video_info["author"] or "未知作者"
        task.parsed_video_path = video_info["video_path"]

        # 2. 查重 (Title + Author)
        # 获取最新的那一条
        duplicates = knowledge_db.get_by_title_and_author(task.parsed_title, task.parsed_author)
        
        if duplicates:
            latest = duplicates[0] # 按时间倒序，取第一个
            
            # 进入确认模式
            task.waiting_for_dup_confirm = True
            task.processing = False # 暂停 processing 状态，允许响应消息
            task.dup_video_code = latest.get("video_code", "N/A")
            task.dup_timestamp = latest.get("timestamp", "未知时间")
            
            msg = (
                f"查询到重复视频\n"
                f"视频码：{task.dup_video_code}\n"
                f"时间戳：{task.dup_timestamp}\n\n"
                f"输入“覆盖”以覆盖旧视频，“新增”以直接添加新条目，“取消”以取消处理。\n"
                f"两分钟后默认取消。"
            )
            await send_text_message(user_id, msg)
            
            # 重设超时计时器 (2分钟)
            task.timer_task = asyncio.create_task(_wait_then_process(user_id))
            return 
        
        # 无重复，直接执行
        await _execute_summary_task(user_id, task, reuse_video_code=None)

    except Exception as e:
        logger.error(f"任务初始化失败: {e}", exc_info=True)
        await send_text_message(user_id, f"处理失败: {str(e)[:100]}")
        _cleanup_pending_files(task)
        _pending.pop(user_id, None)


async def _execute_summary_task(user_id: str, task: PendingTask, reuse_video_code: Optional[str] = None):
    """执行 AI 总结和后续流程"""
    task.processing = True
    video_id = task.parsed_video_id
    
    try:
        # 合并要求
        req = task.share_text
        if task.extra_requirement:
            if task.extra_requirement.strip().lower() not in ("开始", "start", "ok", "好"):
                req = task.extra_requirement

        # 提取音频
        audio_path = extract_audio(task.parsed_video_path)
        
        video_code = reuse_video_code if reuse_video_code else generate_video_code()
        
        await send_text_message(user_id, f"视频: {task.parsed_title}\n作者: {task.parsed_author}\n视频码: {video_code}\n\n处理中...")

        # 3. AI 总结
        async def progress(msg): pass
        summary = await summarize_with_audio(audio_path, task.parsed_title, task.parsed_author, req, progress_callback=progress)

        # 存入知识库
        try:
            tags = extract_tags_from_markdown(summary)
            entry = KnowledgeEntry(
                video_id=video_id, title=task.parsed_title, author=task.parsed_author, source_url=task.share_url,
                summary_markdown=summary, tags=tags, user_requirement=req, video_code=video_code,
            )
            knowledge_db.save(entry)
        except Exception as e:
            logger.error(f"知识库保存失败: {e}")

        # 4. 生成 PDF
        pdf_path = os.path.join(TEMP_DIR, f"{video_id}_summary.pdf")
        pdf_success = False
        try:
            if generate_pdf(summary, pdf_path):
                media_id = await upload_temp_media(pdf_path, "file")
                await _send_file_message(user_id, media_id)
                pdf_success = True
            else:
                logger.warning("PDF 生成失败")
        except Exception as e:
            logger.error(f"PDF 流程异常: {e}")

        if not pdf_success:
            await send_text_message(user_id, "PDF失败，发送文本:")
            await send_markdown_message(user_id, summary)

        logger.info(f"完成: {task.parsed_title}")

    except Exception as e:
        logger.error(f"任务执行失败: {e}", exc_info=True)
        await send_text_message(user_id, f"处理失败: {str(e)[:100]}")

    finally:
        _cleanup_pending_files(task)
        _pending.pop(user_id, None)


def _cleanup_pending_files(task: PendingTask):
    """清理临时文件"""
    if task.parsed_video_id:
        cleanup_files(task.parsed_video_id)
    if task.parsed_video_path and os.path.exists(task.parsed_video_path):
        try: os.remove(task.parsed_video_path)
        except: pass


async def _send_file_message(user_id: str, media_id: str):
    """发送文件消息 (辅助)"""
    # 简单调用 wechat_api 即可，这里为了独立性还是保留了部分逻辑？
    # 其实可以直接调用 wechat_api 里的通用接口，如果不想暴露太多细节。
    # 这里直接复用 wechat_api 里的 httpx 调用逻辑，或者直接构造请求。
    # 为了简化，直接手动构造，因为 wechat_api 只有 upload 和 send_text/markdown。
    # 建议在 wechat_api 添加 send_file_message。
    # 但为了不修改 wechat_api 太多，这里局部实现。
    from app.services.wechat_api import get_access_token
    import httpx
    
    token = await get_access_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"
    payload = {
        "touser": user_id, "msgtype": "file", "agentid": AGENT_ID,
        "file": {"media_id": media_id},
    }
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload)


@app.get("/health")
async def health_check():
    return {"status": "ok", "pending": len(_pending)}


@app.get("/")
async def root():
    return {"message": "Douyin Bot Running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
