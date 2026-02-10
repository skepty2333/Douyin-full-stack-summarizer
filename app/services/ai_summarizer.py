"""AI 总结模块 (Gemini + DeepSeek + Sonnet)"""
import base64
import os
import logging
import httpx
from app.config import (
    API_BASE_URL,
    GEMINI_API_KEY, GEMINI_MODEL,
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL,
    SONNET_API_KEY, SONNET_MODEL,
)

logger = logging.getLogger(__name__)


async def _chat(model, messages, api_key, max_tokens=8192, temperature=0.3, timeout=180) -> str:
    """OpenAI 兼容对话接口"""
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

STAGE2_SYSTEM = """你是一位博学严谨的知识审计专家。请对 AI 生成的学习笔记初稿进行深度审视。

任务：
1. 内容缺失审查：检查未定义的术语、未介绍的人物/背景。
2. 深度不足诊断：指出缺乏论证的观点。
3. 知识拓展建议：补充关联知识和延伸阅读。

## 输出格式 (Markdown)
# 审查报告
## 需要补充解释的概念
1. **[概念]** — 理由 + 搜索关键词
## 需要补充的背景信息
...
## 建议补充的关联知识
...
## 具体搜索任务清单
1. 搜索: "[关键词]" — 用于补充 [内容]
...
"""

STAGE3_SYSTEM = """你是一位顶级知识编辑。请将初稿重写为一份完整、深入、样式精美的最终版笔记。

## 核心原则
1. **结构第一**：直接输出笔记，无废话。
2. **样式规范**：
   - 严禁正文使用引用块。
   - 数学公式：行内 $...$ (中文环境禁止 LaTeX)，块级 $$...$$。
3. **内容深度**：解释专业名词，补充背景。

## 输出结构
# [标题]
> **核心摘要**：...
> **视频作者**：...
## 1. [小节]
...
## 延伸阅读
...
"""


# ======================== Stage 1: Gemini ========================

async def stage1_transcribe_and_draft(audio_path, video_title="", video_author="", user_requirement="") -> str:
    """Gemini 多模态: 音频 → 初稿"""
    logger.info("[Stage1] Gemini 转写+初稿")

    if os.path.getsize(audio_path) > 24 * 1024 * 1024:
        # 大文件回退处理
        return await _stage1_large_audio(audio_path, video_title, video_author, user_requirement)

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
        return await _chat(GEMINI_MODEL, messages, GEMINI_API_KEY, timeout=240)
    except Exception as e:
        logger.warning(f"[Stage1] 失败，回退: {e}")
        return await _stage1_fallback(audio_path, video_title, video_author, user_requirement)


async def _stage1_fallback(audio_path, title, author, req) -> str:
    """WHISPER 转写 + LLM 总结"""
    transcript = await _transcribe_audio(audio_path)
    prompt = f"{_build_context(title, author, req)}\n\n转写文本:\n\n{transcript}"
    messages = [
        {"role": "system", "content": STAGE1_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    return await _chat(GEMINI_MODEL, messages, GEMINI_API_KEY)


async def _stage1_large_audio(audio_path, title, author, req) -> str:
    """大文件分段转写"""
    # 省略具体实现细节，保持原有逻辑但简化代码结构
    # 这里为了保持功能完整性，保留核心逻辑但简化注释
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
    return await _chat(GEMINI_MODEL, [{"role": "system", "content": STAGE1_SYSTEM}, {"role": "user", "content": prompt}], GEMINI_API_KEY)


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


# ======================== Stage 2: DeepSeek ========================

async def stage2_critical_review(draft_markdown: str) -> str:
    """DeepSeek 深度审视"""
    logger.info("[Stage2] DeepSeek 深度审视")
    messages = [
        {"role": "system", "content": STAGE2_SYSTEM},
        {"role": "user", "content": f"以下是初稿，请审视：\n\n---\n{draft_markdown}\n---\n\n请输出审查报告。"},
    ]
    return await _chat(DEEPSEEK_MODEL, messages, DEEPSEEK_API_KEY, max_tokens=4096, temperature=0.2, timeout=300)


# ======================== Stage 3: Sonnet ========================

async def stage3_enrich_and_finalize(draft_markdown, review_report, user_requirement="") -> str:
    """Sonnet 联网搜索 + 最终版"""
    logger.info("[Stage3] Sonnet 联网搜索")
    user_content = f"## 初稿\n{draft_markdown}\n\n## 审查报告\n{review_report}\n"
    if user_requirement: user_content += f"\n## 用户要求\n{user_requirement}\n"
    user_content += "\n请执行搜索任务并输出最终版笔记。"

    messages = [{"role": "system", "content": STAGE3_SYSTEM}, {"role": "user", "content": user_content}]
    
    # Sonnet 工具调用
    url = f"{API_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {SONNET_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": SONNET_MODEL, "messages": messages, "max_tokens": 12000, "temperature": 0.3,
        "tools": [{"type": "web_search_20250305", "name": "web_search"}]
    }

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"].get("content", "")
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
    return content


async def summarize_with_audio(audio_path, video_title="", video_author="", user_requirement="", progress_callback=None) -> str:
    """三阶段 AI 总结流水线"""
    async def notify(msg):
        if progress_callback: await progress_callback(msg)

    await notify("🔬 [1/3] Gemini 转写生成初稿...")
    draft = await stage1_transcribe_and_draft(audio_path, video_title, video_author, user_requirement)
    
    await notify("🧠 [2/3] DeepSeek 深度审视...")
    review = await stage2_critical_review(draft)
    
    await notify("🌐 [3/3] Sonnet 联网搜索生成终稿...")
    final = await stage3_enrich_and_finalize(draft, review, user_requirement)
    
    await notify("✅ 处理完成")
    return final


def _build_context(title, author, requirement):
    parts = ["请对以下视频内容进行转写和总结："]
    if title: parts.append(f"标题：{title}")
    if author: parts.append(f"作者：{author}")
    if requirement: parts.append(f"\n用户特别要求：{requirement}")
    return "\n".join(parts)
