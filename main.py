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
import hashlib
import shutil
import uuid
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from app.config import (
    CORP_ID, CALLBACK_TOKEN, CALLBACK_AES_KEY,
    TEMP_DIR, LOG_LEVEL, SERVER_HOST, SERVER_PORT,
    KNOWLEDGE_ASSETS_DIR, MAX_CONCURRENT_JOBS, JOB_TIMEOUT_SECONDS, DOWNLOAD_TIMEOUT_SECONDS,
    MODEL_USAGE_LOG_ENABLED,
    validate_ai_config,
)
from app.utils.wechat_crypto import WXBizMsgCrypt
from app.services.wechat_api import (
    close_wechat_client,
    send_file_message,
    send_markdown_message,
    send_text_message,
    upload_temp_media,
)
from app.services.douyin_parser import (
    extract_url_from_text, extract_user_requirement,
    resolve_and_download, cleanup_files, cleanup_stale_job_dirs,
)
from app.services.ai_summarizer import summarize_with_artifacts, generate_tags_with_ai
from app.services.aliyun_client import close_aliyun_client
from app.services.pdf_generator import generate_pdf
from app.services.video_frames import (
    persist_video_frame_markers_for_storage,
    render_video_frame_markers_for_pdf,
    strip_video_frame_markers_for_storage,
)
from app.database.knowledge_store import KnowledgeAsset, KnowledgeStore, KnowledgeEntry
from app.database.model_usage_store import (
    bind_model_usage_context,
    get_model_usage_store,
)

# 初始化
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# httpx logs complete URLs at INFO level.  Some WeCom endpoints carry their
# short-lived access token in the query string, so keep transport logs out of
# production journals while retaining application-level failures.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("douyin-bot")

crypto = WXBizMsgCrypt(CALLBACK_TOKEN, CALLBACK_AES_KEY, CORP_ID)
knowledge_db = KnowledgeStore()

# 消息去重
_processed_msgs: Dict[str, float] = {}
MSG_DEDUP_TTL = 300
MAX_CALLBACK_BODY_BYTES = 1024 * 1024


def generate_video_code() -> str:
    """Generate an unused five-character public code."""
    chars = string.ascii_lowercase + string.digits
    for _ in range(20):
        code = ''.join(random.choices(chars, k=5))
        if not knowledge_db.video_code_exists(code):
            return code
    raise RuntimeError("无法生成唯一视频码，请重试")


# 会话管理
WAIT_SECONDS = 120  # 等待用户输入要求的时间
MAX_QUEUE_SIZE = 3  # 每用户最大排队数


@dataclass
class PendingTask:
    """待处理任务"""
    user_id: str
    share_url: str
    share_text: str
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
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
    parsed_media_url: str = ""


@dataclass
class UserTaskQueue:
    """每用户任务队列"""
    active: Optional[PendingTask] = None       # 当前活跃任务 (等待要求/处理中)
    queue: List[PendingTask] = field(default_factory=list)  # 排队中的任务

    @property
    def total_count(self) -> int:
        return (1 if self.active else 0) + len(self.queue)

    @property
    def is_processing(self) -> bool:
        return self.active is not None and self.active.processing


_pending: Dict[str, UserTaskQueue] = {}
_background_tasks: set[asyncio.Task] = set()
_job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
_download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
_accepting_messages = True


def _spawn_background(coro) -> asyncio.Task:
    """Track background tasks and log otherwise-lost exceptions."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)

    def done(completed: asyncio.Task) -> None:
        _background_tasks.discard(completed)
        if completed.cancelled():
            return
        error = completed.exception()
        if error:
            logger.error("后台任务异常", exc_info=(type(error), error, error.__traceback__))

    task.add_done_callback(done)
    return task


def _active_task(user_id: str, job_id: Optional[str] = None) -> Optional[PendingTask]:
    queue = _pending.get(user_id)
    task = queue.active if queue else None
    if task is None or (job_id is not None and task.job_id != job_id):
        return None
    return task


def _claim_for_processing(user_id: str, job_id: str) -> Optional[PendingTask]:
    """Atomically claim the current job before the next await point."""
    task = _active_task(user_id, job_id)
    if task is None or task.processing or task.waiting_for_dup_confirm:
        return None
    task.processing = True
    return task


async def _safe_send_text(user_id: str, content: str) -> bool:
    """Best-effort notification that must not block queue cleanup or advancement."""
    try:
        await send_text_message(user_id, content)
        return True
    except Exception:
        logger.exception("企业微信通知失败: user=%s", user_id)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _accepting_messages
    os.makedirs(TEMP_DIR, exist_ok=True)
    cleanup_stale_job_dirs()
    try:
        await asyncio.to_thread(knowledge_db.prune_orphan_assets)
    except Exception:
        logger.exception("清理过期孤儿知识图片失败，继续启动")
    if MODEL_USAGE_LOG_ENABLED:
        try:
            await asyncio.to_thread(get_model_usage_store)
        except Exception:
            logger.exception("模型调用日志数据库初始化失败")
    _accepting_messages = True
    logger.info("Bot 启动")
    yield
    _accepting_messages = False
    timers = [
        queue.active.timer_task
        for queue in _pending.values()
        if queue.active and queue.active.timer_task and not queue.active.timer_task.done()
    ]
    for timer in timers:
        timer.cancel()
    shutdown_deadline = asyncio.get_running_loop().time() + 20
    while _background_tasks:
        remaining = shutdown_deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        await asyncio.wait(tuple(_background_tasks), timeout=remaining)
    pending_tasks = tuple(_background_tasks)
    for task in pending_tasks:
        task.cancel()
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)
    for queue in tuple(_pending.values()):
        if queue.active:
            _cleanup_pending_files(queue.active)
        for queued_task in queue.queue:
            _cleanup_pending_files(queued_task)
    _pending.clear()
    await close_aliyun_client()
    await close_wechat_client()
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
    if not _accepting_messages:
        return PlainTextResponse(content="service unavailable", status_code=503)
    content_length = request.headers.get("content-length", "").strip()
    if content_length.isdigit() and int(content_length) > MAX_CALLBACK_BODY_BYTES:
        return PlainTextResponse(content="payload too large", status_code=413)
    raw_body = await request.body()
    if len(raw_body) > MAX_CALLBACK_BODY_BYTES:
        return PlainTextResponse(content="payload too large", status_code=413)
    body = raw_body.decode("utf-8")

    try:
        xml_text = crypto.decrypt_msg(body, msg_signature, timestamp, nonce)
        xml_root = ET.fromstring(xml_text)
        msg_type = xml_root.find("MsgType").text
        from_user = xml_root.find("FromUserName").text

        # 简单的去重
        msg_id = (xml_root.find("MsgId").text or "") if xml_root.find("MsgId") is not None else ""
        create_time = (xml_root.find("CreateTime").text or "") if xml_root.find("CreateTime") is not None else ""
        fallback_id = hashlib.sha256(xml_text.encode("utf-8")).hexdigest()[:20]
        dedup_key = f"{from_user}_{msg_id or fallback_id}_{create_time}"
        now = time.time()
        if dedup_key in _processed_msgs and now - _processed_msgs[dedup_key] < MSG_DEDUP_TTL:
            return PlainTextResponse(content="success")
        _processed_msgs[dedup_key] = now
        
        # 清理过期
        for k in [k for k, v in _processed_msgs.items() if now - v > MSG_DEDUP_TTL]:
            del _processed_msgs[k]

        if msg_type == "text":
            content = xml_root.find("Content").text or ""
            logger.info(f"收到消息 {from_user}: {content[:50]}")
            _spawn_background(handle_message(from_user, content))
        else:
            logger.info(f"忽略消息类型: {msg_type}")

    except Exception as e:
        logger.error(f"处理回调异常: {e}", exc_info=True)

    return PlainTextResponse(content="success")


async def handle_message(user_id: str, content: str):
    """消息路由"""
    try:
        content_stripped = content.strip()
        
        # 队列状态查询
        if content_stripped.lower() in ("队列", "queue", "状态"):
            if user_id in _pending:
                uq = _pending[user_id]
                active_info = "正在处理1个" if uq.is_processing else "等待开始1个"
                queue_info = f"排队等待{len(uq.queue)}个"
                await send_text_message(user_id, f"当前队列状态: {active_info}, {queue_info}。")
            else:
                await send_text_message(user_id, "当前无任务。")
            return

        # 情况1: 用户有活跃任务
        if user_id in _pending:
            uq = _pending[user_id]
            active = uq.active
            
            if active:
                # A. 正在等待重复确认
                if active.waiting_for_dup_confirm:
                    if content_stripped in ("覆盖", "Overwrite"):
                        if active.timer_task and not active.timer_task.done():
                            active.timer_task.cancel()
                        active.waiting_for_dup_confirm = False
                        active.processing = True
                        old_code = active.dup_video_code
                        await _safe_send_text(user_id, f"确认覆盖, 沿用视频码: {old_code}, 开始处理...")
                        await _execute_summary_task(
                            user_id,
                            active,
                            video_code_override=old_code,
                            allow_overwrite=True,
                        )
                        
                    elif content_stripped in ("新增", "New"):
                        if active.timer_task and not active.timer_task.done():
                            active.timer_task.cancel()
                        active.waiting_for_dup_confirm = False
                        active.processing = True
                        await _safe_send_text(user_id, "确认新增，开始处理...")
                        await _execute_summary_task(
                            user_id,
                            active,
                            allow_overwrite=False,
                        )
                        
                    elif content_stripped in ("取消", "Cancel"):
                        if active.timer_task and not active.timer_task.done():
                            active.timer_task.cancel()
                        active.waiting_for_dup_confirm = False
                        _cleanup_pending_files(active)
                        _advance_queue(user_id, active.job_id)
                        await _safe_send_text(user_id, "收到, 取消处理。")
                        
                    else:
                        await send_text_message(user_id, '输入"覆盖"、"新增"或"取消"。')
                    return

                # B. 正在处理中 -> 新链接入队
                if active.processing:
                    new_url = extract_url_from_text(content)
                    if new_url:
                        await _enqueue_task(user_id, content, new_url)
                    else:
                        await send_text_message(user_id, "当前有视频正在处理, 可发送新链接加入队列。")
                    return

                # C. 活跃任务尚未开始处理
                if content_stripped in ("取消", "Cancel"):
                    if active.timer_task and not active.timer_task.done():
                        active.timer_task.cancel()
                    _cleanup_pending_files(active)
                    _advance_queue(user_id, active.job_id)
                    await _safe_send_text(user_id, "收到, 取消处理。")
                    return

                # 检查是否新链接 (替换当前等待中的任务)
                new_url = extract_url_from_text(content)
                if new_url:
                    if active.timer_task and not active.timer_task.done():
                        active.timer_task.cancel()
                    _cleanup_pending_files(active)
                    inline_req = extract_user_requirement(content, new_url)
                    new_task = PendingTask(user_id=user_id, share_url=new_url, share_text=inline_req)
                    uq.active = new_task
                    new_task.timer_task = _spawn_background(_wait_then_process(user_id, new_task.job_id))
                    await _safe_send_text(user_id, '收到, 发送"开始"立即处理, "取消"以取消操作, 或输入具体要求。2分钟后默认处理。')
                    return

                # 立即开始
                if content_stripped.lower() in ("开始", "start", "ok", "好"):
                    if active.timer_task and not active.timer_task.done():
                        active.timer_task.cancel()
                    if not _claim_for_processing(user_id, active.job_id):
                        return
                    await _safe_send_text(user_id, "正在开始处理...")
                    await _process_task_init(user_id, active.job_id, already_claimed=True)
                    return

                # 补充要求 -> 自动开始
                if active.timer_task and not active.timer_task.done():
                    active.timer_task.cancel()
                active.extra_requirement = content_stripped
                if not _claim_for_processing(user_id, active.job_id):
                    return
                logger.info(f"补充要求 {user_id}: {content[:30]}")
                await _safe_send_text(user_id, "已收到补充要求, 正在开始处理...")
                await _process_task_init(user_id, active.job_id, already_claimed=True)
                return

        # 情况2: 新链接
        url = extract_url_from_text(content)
        if url:
            await _start_new_task(user_id, content, url)
            return

        # 情况3: 帮助信息
        await send_text_message(
            user_id,
            '收到, 发送"开始"立即处理, "取消"以取消操作, 或输入具体要求。2分钟后默认处理。'
        )

    except Exception as e:
        logger.error(f"handle_message异常: {e}", exc_info=True)
        try:
            await send_text_message(user_id, "系统繁忙, 请稍后重试。")
        except: pass


async def _enqueue_task(user_id: str, content: str, url: str):
    """将新任务加入队列 (直接排队, 无需等待用户输入要求)"""
    uq = _pending[user_id]
    if len(uq.queue) >= MAX_QUEUE_SIZE:
        await send_text_message(user_id, f"队列已满 ({MAX_QUEUE_SIZE}/{MAX_QUEUE_SIZE}), 请等待当前任务完成。")
        return
    
    inline_req = extract_user_requirement(content, url)
    task = PendingTask(user_id=user_id, share_url=url, share_text=inline_req)
    uq.queue.append(task)
    pos = len(uq.queue)
    await send_text_message(user_id, f"已加入队列, 当前位置: 第{pos + 1}个, 前方还有{pos}个任务。")


async def _start_new_task(user_id: str, content: str, url: str):
    """创建新任务 (首个任务, 无队列)"""
    inline_req = extract_user_requirement(content, url)
    task = PendingTask(user_id=user_id, share_url=url, share_text=inline_req)
    
    if user_id not in _pending:
        _pending[user_id] = UserTaskQueue()
    _pending[user_id].active = task

    task.timer_task = _spawn_background(_wait_then_process(user_id, task.job_id))
    await _safe_send_text(user_id, '收到, 发送"开始"立即处理, "取消"以取消操作, 或输入具体要求。2分钟后默认处理。')


async def _wait_then_process(user_id: str, job_id: str):
    """超时自动处理"""
    try:
        await asyncio.sleep(WAIT_SECONDS)
        if not _accepting_messages:
            return
        task = _active_task(user_id, job_id)
        if not task:
            return
            
        # 如果是在等待重复确认状态超时
        if task.waiting_for_dup_confirm:
            logger.info(f"{user_id} 重复确认超时, 默认取消")
            await _safe_send_text(user_id, "两分钟超时, 默认取消处理。")
            _cleanup_pending_files(task)
            _advance_queue(user_id, task.job_id)
            return

        # 正常超时，开始处理
        if _claim_for_processing(user_id, job_id):
            logger.info(f"{user_id} 超时, 开始处理")
            await _process_task_init(user_id, job_id, already_claimed=True)
                
    except asyncio.CancelledError:
        pass


async def _process_task_init(
    user_id: str,
    job_id: str,
    already_claimed: bool = False,
):
    """任务处理入口: 解析 -> 查重 -> (执行 或 等待确认)"""
    task = _active_task(user_id, job_id)
    if not task:
        return
    if not already_claimed and not _claim_for_processing(user_id, job_id):
        return

    try:
        async with _download_semaphore:
            async with asyncio.timeout(DOWNLOAD_TIMEOUT_SECONDS):
                video_info = await resolve_and_download(task.share_url, task.job_id)
        
        task.parsed_video_id = video_info["video_id"]
        task.parsed_title = video_info["title"] or "未知标题"
        task.parsed_author = video_info["author"] or "未知作者"
        task.parsed_video_path = video_info["video_path"]
        task.parsed_media_url = video_info.get("video_url", "")

        # 查重 (Title + Author)
        duplicates = await asyncio.to_thread(
            knowledge_db.get_by_title_and_author,
            task.parsed_title,
            task.parsed_author,
        )
        
        if duplicates:
            latest = duplicates[0]
            task.waiting_for_dup_confirm = True
            task.processing = False
            task.dup_video_code = latest.get("video_code", "N/A")
            task.dup_timestamp = latest.get("timestamp", "未知时间")
            
            msg = (
                f"查询到重复视频\n"
                f"视频码: {task.dup_video_code}\n"
                f"时间戳: {task.dup_timestamp}\n\n"
                f'输入"覆盖"以覆盖旧视频, "新增"以直接添加新条目, "取消"以取消处理。\n'
                f"两分钟后默认取消。"
            )
            await send_text_message(user_id, msg)
            task.timer_task = _spawn_background(_wait_then_process(user_id, task.job_id))
            return 
        
        await _execute_summary_task(user_id, task)

    except asyncio.CancelledError:
        _cleanup_pending_files(task)
        raise
    except asyncio.TimeoutError:
        logger.error("视频解析或下载超时: job=%s", task.job_id)
        try:
            await _safe_send_text(user_id, "视频解析或下载超时，请稍后重试。")
        finally:
            _cleanup_pending_files(task)
            _advance_queue(user_id, task.job_id)
    except Exception as e:
        logger.error(f"任务初始化失败: {e}", exc_info=True)
        try:
            await send_text_message(user_id, f"处理失败: {str(e)[:100]}")
        finally:
            _cleanup_pending_files(task)
            _advance_queue(user_id, task.job_id)


async def _execute_summary_task(
    user_id: str,
    task: PendingTask,
    video_code_override: Optional[str] = None,
    allow_overwrite: bool = False,
):
    """执行 AI 总结和后续流程"""
    task.processing = True
    video_id = task.parsed_video_id

    try:
        async with _job_semaphore:
            async with asyncio.timeout(JOB_TIMEOUT_SECONDS):
                req = task.share_text
                if task.extra_requirement:
                    if task.extra_requirement.strip().lower() not in ("开始", "start", "ok", "好"):
                        req = task.extra_requirement

                video_code = video_code_override or await asyncio.to_thread(generate_video_code)
                await _safe_send_text(
                    user_id,
                    f"视频: {task.parsed_title}\n作者: {task.parsed_author}\n"
                    f"视频码: {video_code}\n\n处理中...",
                )

                async def progress(msg):
                    await _safe_send_text(user_id, msg)

                with bind_model_usage_context(
                    job_id=task.job_id,
                    video_code=video_code,
                ):
                    summary_result = await summarize_with_artifacts(
                        task.parsed_video_path,
                        task.parsed_title,
                        task.parsed_author,
                        req,
                        progress_callback=progress,
                        media_url=task.parsed_media_url,
                    )
                    summary = strip_video_frame_markers_for_storage(
                        summary_result.markdown
                    )
                    tags = await generate_tags_with_ai(
                        summary,
                        task.parsed_title,
                        task.parsed_author,
                    )
                storage_summary, persisted_frames = await asyncio.to_thread(
                    persist_video_frame_markers_for_storage,
                    summary_result.markdown,
                    summary_result.video_frames,
                    video_code=video_code,
                    asset_root=KNOWLEDGE_ASSETS_DIR,
                )
                knowledge_assets = [
                    KnowledgeAsset(
                        asset_key=frame.annotation_id,
                        relative_path=frame.relative_path,
                        mime_type=frame.mime_type,
                        timestamp_ms=frame.timestamp_ms,
                        caption=frame.caption,
                        kind=frame.kind,
                        confidence=frame.confidence,
                        width=frame.width,
                        height=frame.height,
                        byte_size=frame.byte_size,
                        sha256=frame.sha256,
                        display_order=frame.display_order,
                        quality_score=frame.quality_score,
                    )
                    for frame in persisted_frames
                ]
                entry = KnowledgeEntry(
                    video_id=video_id,
                    title=task.parsed_title,
                    author=task.parsed_author,
                    source_url=task.share_url,
                    summary_markdown=storage_summary,
                    tags=tags,
                    user_requirement=req,
                    duration_seconds=summary_result.diagnostics.duration_seconds or 0.0,
                    video_code=video_code,
                )
                await asyncio.to_thread(
                    knowledge_db.save,
                    entry,
                    allow_overwrite,
                    knowledge_assets,
                )

                pdf_path = os.path.join(os.path.dirname(task.parsed_video_path), "summary.pdf")
                pdf_delivered = False
                try:
                    pdf_markdown = render_video_frame_markers_for_pdf(
                        summary_result.markdown,
                        summary_result.video_frames,
                    )
                    if await asyncio.to_thread(generate_pdf, pdf_markdown, pdf_path):
                        media_id = await upload_temp_media(pdf_path, "file")
                        await send_file_message(user_id, media_id)
                        pdf_delivered = True
                except Exception:
                    logger.exception("PDF 生成或发送失败，准备发送 Markdown")

                if not pdf_delivered:
                    await _safe_send_text(user_id, "PDF 发送失败，改为发送文本。")
                    await send_markdown_message(user_id, summary)

                logger.info("完成: job=%s title=%s", task.job_id, task.parsed_title)

    except asyncio.TimeoutError:
        logger.error("任务超时: job=%s", task.job_id)
        await _safe_send_text(user_id, "处理超时，任务已终止，请稍后重试。")
    except Exception as e:
        logger.error(f"任务执行失败: {e}", exc_info=True)
        await _safe_send_text(user_id, f"处理失败: {str(e)[:100]}")

    finally:
        _cleanup_pending_files(task)
        _advance_queue(user_id, task.job_id)


def _cleanup_pending_files(task: PendingTask):
    """清理临时文件"""
    cleanup_files(task.job_id)


def _advance_queue(user_id: str, completed_job_id: str):
    """推进队列：激活下一个排队任务，或清理空队列"""
    if user_id not in _pending:
        return
    uq = _pending[user_id]
    if not uq.active or uq.active.job_id != completed_job_id:
        logger.warning(
            "忽略过期队列推进: user=%s completed=%s active=%s",
            user_id,
            completed_job_id,
            uq.active.job_id if uq.active else None,
        )
        return
    if not _accepting_messages:
        _cleanup_pending_files(uq.active)
        for queued_task in uq.queue:
            _cleanup_pending_files(queued_task)
        del _pending[user_id]
        return
    if uq.queue:
        next_task = uq.queue.pop(0)
        uq.active = next_task
        remaining = len(uq.queue)
        msg = f"开始处理队列中的下一个视频。剩余排队: {remaining}个。"
        _spawn_background(_advance_and_notify(user_id, next_task.job_id, msg))
    else:
        uq.active = None
        del _pending[user_id]


async def _advance_and_notify(user_id: str, job_id: str, msg: str):
    """通知用户并启动下一个任务"""
    await asyncio.sleep(2)  # 等待 PDF 文件消息送达
    if not _accepting_messages:
        return
    await _safe_send_text(user_id, msg)
    if _claim_for_processing(user_id, job_id):
        await _process_task_init(user_id, job_id, already_claimed=True)


@app.get("/health")
async def health_check():
    return await readiness_check()


@app.get("/live")
async def liveness_check():
    return {"status": "ok"}


@app.get("/ready")
async def readiness_check():
    problems = validate_ai_config()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        problems.append("ffmpeg/ffprobe 不可用")
    try:
        if not await asyncio.to_thread(knowledge_db.health_check):
            problems.append("知识库健康检查失败")
    except Exception as exc:
        problems.append(f"知识库不可用: {type(exc).__name__}")
    if MODEL_USAGE_LOG_ENABLED:
        try:
            usage_store = await asyncio.to_thread(get_model_usage_store)
            if not await asyncio.to_thread(usage_store.health_check):
                problems.append("模型调用日志库健康检查失败")
        except Exception as exc:
            problems.append(f"模型调用日志库不可用: {type(exc).__name__}")
    try:
        os.makedirs(TEMP_DIR, exist_ok=True)
        if not os.access(TEMP_DIR, os.W_OK):
            problems.append("临时目录不可写")
        elif shutil.disk_usage(TEMP_DIR).free < 512 * 1024 * 1024:
            problems.append("临时目录可用空间不足 512MB")
    except OSError as exc:
        problems.append(f"临时目录不可用: {type(exc).__name__}")

    payload = {
        "status": "ready" if not problems and _accepting_messages else "not_ready",
        "pending_users": len(_pending),
        "active_background_tasks": len(_background_tasks),
        "problems": problems,
    }
    return JSONResponse(payload, status_code=200 if payload["status"] == "ready" else 503)


@app.get("/")
async def root():
    return {"message": "Douyin Bot Running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
