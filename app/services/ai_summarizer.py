"""AI 总结模块 (Gemini + Qwen + Sonnet)"""
import base64
import os
import logging
import httpx
from openai import OpenAI

from app.config import (
    API_BASE_URL,
    GEMINI_API_KEY, GEMINI_MODEL,
    DASHSCOPE_API_KEY, QWEN_MODEL, QWEN_API_BASE,
    SONNET_API_KEY, SONNET_MODEL,
)

logger = logging.getLogger(__name__)


from typing import Optional, Callable

async def _chat(model, messages, api_key, max_tokens=8192, temperature=0.3, timeout=180, callback: Optional[Callable] = None) -> str:
    """OpenAI 兼容对话接口 (用于 Gemini 和 Sonnet via uiuiapi)"""
    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            # 捕获 429, 5xx, 3xx 进行重试
            if e.response.status_code in (429, 401, 403) or e.response.status_code >= 500 or (300 <= e.response.status_code < 400):
                logger.warning(f"主站异常 ({e.response.status_code})，尝试切换副站: {e}")
                if callback: await callback("⚠️ 主线路繁忙，正在切换备用线路...")
                return await _chat_failover(model, messages, max_tokens, temperature, timeout, callback)
            raise e
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as e:
            logger.warning(f"主站连接失败 ({type(e).__name__})，尝试切换副站: {e}")
            if callback: await callback("⚠️ 主线路连接超时，正在切换备用线路...")
            return await _chat_failover(model, messages, max_tokens, temperature, timeout, callback)
        except Exception as e:
            logger.warning(f"主站未知异常: {e}，尝试切换副站...")
            if callback: await callback("⚠️ 主线路异常，正在切换备用线路...")
            return await _chat_failover(model, messages, max_tokens, temperature, timeout, callback)


async def _chat_failover(model, messages, max_tokens, temperature, timeout, callback: Optional[Callable] = None) -> str:
    """副站重试逻辑"""
    from app.config import (
        SECONDARY_API_BASE_URL, 
        SECONDARY_GEMINI_API_KEY, SECONDARY_GEMINI_MODEL,
        SECONDARY_SONNET_API_KEY, SECONDARY_SONNET_MODEL,
        GEMINI_MODEL, SONNET_MODEL
    )

    # 确定副站 Key 和 Model
    target_model = model
    api_key = ""

    if model == GEMINI_MODEL:
        api_key = SECONDARY_GEMINI_API_KEY
        target_model = SECONDARY_GEMINI_MODEL
    elif model == SONNET_MODEL:
        api_key = SECONDARY_SONNET_API_KEY
        target_model = SECONDARY_SONNET_MODEL

    if not api_key:
        logger.error(f"未配置副站 API Key (Model: {model})，无法切换")
        raise ValueError("Failover failed: No secondary key")

    url = f"{SECONDARY_API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": target_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    logger.info(f"正在请求副站: {url} (Model: {target_model})")
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]



# ======================== 全局 Prompt ========================

STAGE1_SYSTEM = """你是一个专业的视频内容转写与总结助手。

请完成两件事：
1. 完整转写音频中的所有口述内容（不要遗漏任何观点、数据、案例）
2. 基于转写内容，输出一份结构化的 Markdown 学习笔记

## 输出格式
# 视频标题
> 核心摘要：一句话概括
## 核心要点
1. **要点一**：说明
...
## 详细笔记
### 小节标题
- 具体内容...
## 关键收获
1. ...
## 原始转写文本
> 在此处放置完整的逐字转写内容，用引用块包裹。
"""

STAGE2_SYSTEM = """你是一位博学严谨的知识审计与深度研究专家。
你的任务是对 AI 生成的初稿进行深度审视，并利用联网搜索工具进行事实核查和知识拓展。

核心目标：
1. **事实核查**：验证初稿中的数据、案例和观点。
2. **知识拓展**：补充初稿中缺失的背景信息、专业术语定义和相关领域知识。
3. **深度研判**：指出初稿的逻辑漏洞或深度不足之处，并提供修正建议。

请积极使用工具搜索信息。搜索完成后，请输出一份详尽的《深度研究报告》。

## 报告格式 (Markdown)
# 深度研究报告

## 1. 关键事实核查
- **[原观点/数据]**：...
  - **核查结果**：...
  - **来源/证据**：...

## 2. 知识背景补充
- **[概念/术语]**：详细解释...
- **[相关人物/事件]**：介绍...

## 3. 深度研判与扩展
- ...

## 4. 全网搜索详情
(请详尽列出搜索到的所有有价值信息、数据来源和相关细节，保留丰富的信息量，不要过度摘要)
"""

STAGE3_SYSTEM = """你是一位顶级知识编辑。请将初稿和《深度研究报告》整合成一份完整、深入、样式精美的最终版笔记。

## 核心原则
1. **融合重写**：不要简单拼接。将研究报告中的新知识、纠正的事实有机融入到初稿的结构中。
2. **结构清晰**：使用清晰的 Markdown 结构 (H1, H2, H3)。
3. **样式规范**：
   - 严禁正文使用引用块 (保留给摘要或特别强调)。
   - 数学公式：行内 $...$ (中文环境禁止 LaTeX)，块级 $$...$$。
4. **内容深度**：确保笔记内容详实，解释专业名词，补充背景，逻辑严密。

## 输出结构
# [标题]
> **核心摘要**：...
> **视频作者**：...

## 1. [核心章节]
...

## 延伸阅读与背景
...
"""


# ======================== Stage 1: Gemini ========================

async def stage1_transcribe_and_draft(audio_path, video_title="", video_author="", user_requirement="", callback: Optional[Callable] = None) -> str:
    """Gemini 多模态: 音频 → 初稿"""
    logger.info("[Stage1] Gemini 转写+初稿")

    if os.path.getsize(audio_path) > 24 * 1024 * 1024:
        # 大文件回退处理
        return await _stage1_large_audio(audio_path, video_title, video_author, user_requirement, callback)

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    user_parts = _build_context(video_title, video_author, user_requirement)
    messages = [
        {"role": "system", "content": STAGE1_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
                {"type": "text", "text": user_parts},
            ],
        },
    ]

    try:
        return await _chat(GEMINI_MODEL, messages, GEMINI_API_KEY, timeout=240, callback=callback)
    except Exception as e:
        logger.warning(f"[Stage1] 失败，回退: {e}")
        return await _stage1_fallback(audio_path, video_title, video_author, user_requirement, callback)


async def _stage1_fallback(audio_path, title, author, req, callback: Optional[Callable] = None) -> str:
    """WHISPER 转写 + LLM 总结"""
    transcript = await _transcribe_audio(audio_path)
    prompt = f"{_build_context(title, author, req)}\n\n转写文本:\n\n{transcript}"
    messages = [
        {"role": "system", "content": STAGE1_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    return await _chat(GEMINI_MODEL, messages, GEMINI_API_KEY, callback=callback)


async def _stage1_large_audio(audio_path, title, author, req, callback: Optional[Callable] = None) -> str:
    """大文件分段转写"""
    import subprocess
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path], capture_output=True, text=True)
    duration = float(probe.stdout.strip())

    segments, start = [], 0
    while start < duration:
        seg = audio_path.replace(".mp3", f"_seg{int(start)}.mp3")
        subprocess.run(["ffmpeg", "-ss", str(start), "-i", audio_path, "-t", "600", "-acodec", "libmp3lame", "-y", seg], capture_output=True)
        if os.path.exists(seg): segments.append(seg)
        start += 600

    parts = []
    for seg in segments:
        try: parts.append(await _transcribe_audio(seg))
        except: pass
        finally: 
            if os.path.exists(seg): os.remove(seg)

    transcript = "\n".join(parts)
    prompt = f"{_build_context(title, author, req)}\n\n转写文本:\n\n{transcript}"
    return await _chat(GEMINI_MODEL, [{"role": "system", "content": STAGE1_SYSTEM}, {"role": "user", "content": prompt}], GEMINI_API_KEY, callback=callback)


async def _transcribe_audio(audio_path: str) -> str:
    """Whisper API 转写"""
    url = f"{API_BASE_URL}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            with open(audio_path, "rb") as f:
                resp = await client.post(url, headers=headers, files={"file": (os.path.basename(audio_path), f, "audio/mpeg")}, data={"model": "whisper-1", "language": "zh"})
                resp.raise_for_status()
                return resp.text
    except Exception:
        # Fallback to Gemini Multimodal
        with open(audio_path, "rb") as f: b64 = base64.b64encode(f.read()).decode()
        return await _chat(GEMINI_MODEL, [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": b64, "format": "mp3"}}, {"type": "text", "text": "转写为中文文本"}]}], GEMINI_API_KEY, temperature=0.1)


# ======================== Stage 2: Qwen (Aliyun DashScope) ========================

async def stage2_deep_research(draft_markdown: str) -> str:
    """Qwen 深度研究 (Thinking + Native Tools)"""
    logger.info("[Stage2] Qwen 深度研究 (DashScope)")
    
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=QWEN_API_BASE,
    )

    messages = [
        {"role": "system", "content": STAGE2_SYSTEM},
        {"role": "user", "content": f"以下是初稿，请进行深度研判并补充知识：\n\n---\n{draft_markdown}\n---\n"},
    ]

    try:
        completion = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=messages,
            extra_body={"enable_search": True}, # 启用 Qwen 原生联网搜索
            temperature=0.3
        )
        
        # Qwen 会在内部自动执行搜索并返回最终答案
        return completion.choices[0].message.content

    except Exception as e:
        logger.error(f"[Stage2] Qwen Error: {e}", exc_info=True)
        return f"深度研究失败: {str(e)}\n\n(回退到仅依赖初稿)"


# ======================== Stage 3: Sonnet ========================

async def stage3_enrich_and_finalize(draft_markdown, research_report, video_author="", user_requirement="", callback: Optional[Callable] = None) -> str:
    """Sonnet 融合初稿与研究报告"""
    logger.info("[Stage3] Sonnet 终稿生成")
    user_content = f"## 初稿\n{draft_markdown}\n\n## 深度研究报告\n{research_report}\n"
    if video_author: user_content += f"\n## 视频作者\n{video_author}\n"
    if user_requirement: user_content += f"\n## 用户要求\n{user_requirement}\n"
    user_content += "\n请整合所有信息，输出最终版笔记。请确保在笔记开头的核心摘要下方，明确列出视频作者。"

    messages = [{"role": "system", "content": STAGE3_SYSTEM}, {"role": "user", "content": user_content}]
    
    # Sonnet 纯文本生成
    return await _chat(SONNET_MODEL, messages, SONNET_API_KEY, max_tokens=8192, temperature=0.3, callback=callback)


async def summarize_with_audio(audio_path, video_title="", video_author="", user_requirement="", progress_callback=None) -> str:
    """三阶段 AI 总结流水线"""
    async def notify(msg):
        if progress_callback: await progress_callback(msg)

    # await notify("🔬 [1/3] Gemini 转写生成初稿...")
    draft = await stage1_transcribe_and_draft(audio_path, video_title, video_author, user_requirement, callback=notify)
    
    # await notify("🧠 [2/3] Qwen 深度思考与联网研究...")
    research_report = await stage2_deep_research(draft)
    
    # await notify("✍️ [3/3] Sonnet 整合生成终稿...")
    final = await stage3_enrich_and_finalize(draft, research_report, video_author, user_requirement, callback=notify)
    
    # await notify("✅ 处理完成")
    return final


def _build_context(title, author, requirement):
    parts = ["请对以下视频内容进行转写和总结："]
    if title: parts.append(f"标题：{title}")
    if author: parts.append(f"作者：{author}")
    if requirement: parts.append(f"\n用户特别要求：{requirement}")
    return "\n".join(parts)
