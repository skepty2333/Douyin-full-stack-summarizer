"""
抖音知识库 MCP Server

通过 Streamable HTTP 或 stdio 向 MCP 客户端提供知识库文字检索和按需取图。
HTTP 模式默认仅监听 127.0.0.1；远程访问必须经过带认证的 HTTPS
反向代理、VPN 或受控隧道，不直接暴露 8090 端口。

经隧道/反代远程访问时，需将对外域名加入 DNS rebinding 防护白名单：
- MCP_ALLOWED_HOSTS：逗号分隔的完整域名（含端口，支持 域名:* 匹配任意端口）
- MCP_ALLOWED_HOST_SUFFIXES：逗号分隔的后缀通配，如 *.trycloudflare.com（快速隧道）
浏览器客户端另需配置 MCP_ALLOWED_ORIGINS。
"""
import logging
import asyncio
import os
from mcp.server.fastmcp import FastMCP, Image
from mcp.server.transport_security import (
    TransportSecurityMiddleware,
    TransportSecuritySettings,
)
from pydantic import BaseModel, Field
from app.database.knowledge_store import KnowledgeStore
from app.database.note_index import NoteIndex, NoteHit
from app.services.aliyun_client import aliyun_client
from app.config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    KNOWLEDGE_DB_PATH,
    MCP_HOST,
    MCP_PORT,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-knowledge")

# 初始化
store = KnowledgeStore(KNOWLEDGE_DB_PATH)


async def _embed_query(texts):
    return await aliyun_client.embed(
        model=EMBEDDING_MODEL,
        texts=texts,
        dimensions=EMBEDDING_DIMENSIONS,
        operation="embedding_query",
    )


# 章节级混合检索索引（派生数据，可用 scripts/build_note_index.py 重建）。
index = NoteIndex(KNOWLEDGE_DB_PATH, embed_fn=_embed_query)

# 默认只监听本机，并保留 FastMCP 的 DNS rebinding 防护。
# 如需远程访问，请通过带认证的反向代理或受控隧道暴露，
# 并将对外域名加入 MCP_ALLOWED_HOSTS / MCP_ALLOWED_HOST_SUFFIXES 白名单。
_default_hosts = [MCP_HOST, f"{MCP_HOST}:{MCP_PORT}"]
_extra_hosts = [h.strip() for h in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
_extra_origins = [o.strip() for o in os.getenv("MCP_ALLOWED_ORIGINS", "").split(",") if o.strip()]
_host_suffixes = [
    s.strip()[1:].lower()
    for s in os.getenv("MCP_ALLOWED_HOST_SUFFIXES", "").split(",")
    if s.strip().startswith("*.") and len(s.strip()) > 2
]


_orig_validate_host = TransportSecurityMiddleware._validate_host


def _validate_host_with_suffix(self, host):
    """在 SDK 原生校验基础上，额外支持 *.后缀 形式的 Host 通配（如快速隧道随机域名）。"""
    if _orig_validate_host(self, host):
        return True
    # 仅匹配主机名部分（忽略端口），后缀本身不算命中
    hostname = (host or "").split(":")[0].lower()
    if any(hostname.endswith(suffix) and hostname != suffix[1:] for suffix in _host_suffixes):
        return True
    logger.warning("Invalid Host header: %s", host)
    return False


if _host_suffixes:
    # SDK 原生白名单只支持完整域名和 域名:* 端口通配，这里通过替换校验方法补充后缀通配能力
    TransportSecurityMiddleware._validate_host = _validate_host_with_suffix


mcp = FastMCP(
    "douyin_knowledge_mcp",
    host=MCP_HOST,
    port=MCP_PORT,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_default_hosts + _extra_hosts,
        allowed_origins=_extra_origins,
    ),
)


# ======================== Tool: Search ========================

def _note_date(item) -> str:
    stamp = getattr(item, "timestamp", None) or getattr(item, "created_at", "")
    return (stamp or "")[:10]


def _format_note_hits(query: str, hits: list[NoteHit], *, semantic: bool, heading: str) -> str:
    channel_note = "语义 + 关键词" if semantic else "仅关键词（语义通道暂不可用）"
    lines = [f"## {heading}: \"{query}\"（{len(hits)} 条 · {channel_note}）\n"]
    for position, hit in enumerate(hits, 1):
        extras = [f"入库 {_note_date(hit)}", f"命中 {hit.matched_chunks} 段"]
        if hit.best_cosine is not None:
            extras.append(f"相似 {hit.best_cosine:.2f}")
        lines.append(
            f"{position}. `{hit.video_code}` **{hit.title[:60]}** — {hit.author} · " + " · ".join(extras)
        )
        section = hit.best_heading or "概述"
        lines.append(f"   ▸ {section} — {hit.best_snippet}")
    lines.append(
        "\n> 读相关段落用 `collect_sections`；读整篇用 `get_note_by_code`（视频码）。"
    )
    return "\n".join(lines)


def _index_ready() -> bool:
    try:
        return index.stats()["chunks"] > 0
    except Exception:
        logger.exception("读取章节索引状态失败")
        return False


def _legacy_search_text(query: str, rows: list[dict], heading: str) -> str:
    lines = [f"## {heading}: \"{query}\"（{len(rows)} 条 · 章节索引未建立，使用旧版匹配）\n"]
    for r in rows:
        lines.append(
            f"- `{r['video_code']}` **{r['title'][:60]}** — {r['author']} · "
            f"{(r.get('timestamp') or r['created_at'])[:10]} · 标签: {r['tags'][:80]}"
        )
    lines.append("\n> 运行 scripts/build_note_index.py 建立章节索引后可获得相关性排序。")
    return "\n".join(lines)


class SearchInput(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="自然语言问题或关键词，中英文均可；多个关键词用空格分隔。语义与关键词两路召回后融合排序。",
    )
    limit: int = Field(default=10, ge=1, le=30, description="返回笔记数量上限（每条笔记只出现一次）")


@mcp.tool(name="search_notes")
async def search_notes(params: SearchInput) -> str:
    """按相关性搜索视频笔记，返回紧凑列表（视频码、标题、命中的章节和片段）。

    适合回答"哪些视频讲了 X"。要把所有讲 X 的段落一次读完，改用 collect_sections；
    要读某一篇全文，用 get_note_by_code。
    """
    if not await asyncio.to_thread(_index_ready):
        rows = await asyncio.to_thread(store.search, params.query, params.limit)
        if not rows:
            return f"未找到与 \"{params.query}\" 相关的笔记。"
        return _legacy_search_text(params.query, rows, "搜索结果")
    result = await index.search(params.query, limit=params.limit)
    if not result.notes:
        return f"未找到与 \"{params.query}\" 相关的笔记。"
    return _format_note_hits(
        params.query, result.notes, semantic=result.semantic_available, heading="搜索结果"
    )


# ======================== Tool: Precise Search ========================

class PreciseSearchInput(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="空格分隔的关键词；每个关键词都必须在同一条笔记的标题、标签或正文中出现。",
    )
    limit: int = Field(default=10, ge=1, le=30, description="返回笔记数量上限")


@mcp.tool(name="search_notes_precise")
async def search_notes_precise(params: PreciseSearchInput) -> str:
    """精确搜索：所有关键词都必须命中同一条笔记（AND 逻辑），命中后按相关性排序。适合已知产品名、术语时缩小范围。"""
    if not await asyncio.to_thread(_index_ready):
        rows = await asyncio.to_thread(store.search_precise, params.query, params.limit)
        if not rows:
            return f"未找到同时包含所有关键词 \"{params.query}\" 的笔记。"
        return _legacy_search_text(params.query, rows, "精确搜索")
    result = await index.search(params.query, limit=params.limit, require_all_terms=True)
    if not result.notes:
        return f"未找到同时包含所有关键词 \"{params.query}\" 的笔记。"
    return _format_note_hits(
        params.query, result.notes, semantic=result.semantic_available, heading="精确搜索"
    )


# ======================== Tool: Collect Sections ========================

class CollectSectionsInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=200, description="要汇总的问题或主题")
    max_chars: int = Field(
        default=12000, ge=1000, le=40000, description="返回正文总字数预算（默认 12000 字，约 8K token）"
    )
    max_per_note: int = Field(default=3, ge=1, le=8, description="每条笔记最多贡献几个段落")


@mcp.tool(name="collect_sections")
async def collect_sections(params: CollectSectionsInput) -> str:
    """把知识库中与问题最相关的章节正文按预算汇集起来，跨笔记去重，供一次性通读或综合。

    结果按笔记分组，每个段落标注视频码、章节标题，可回溯到 get_note_by_code。
    """
    if not await asyncio.to_thread(_index_ready):
        return "章节索引未建立，无法汇集段落；请先运行 scripts/build_note_index.py。"
    result = await index.collect(
        params.query, max_chars=params.max_chars, max_per_note=params.max_per_note
    )
    if not result.sections:
        return f"未找到与 \"{params.query}\" 相关的段落。"
    channel_note = "语义 + 关键词" if result.semantic_available else "仅关键词（语义通道暂不可用）"
    lines = [
        f"## 相关段落: \"{params.query}\"（{len(result.sections)} 段 · {result.note_count} 条笔记 · "
        f"{result.total_chars} 字 · {channel_note}）\n"
    ]
    current = None
    for section in result.sections:
        chunk = section.chunk
        if chunk.knowledge_id != current:
            current = chunk.knowledge_id
            lines.append(
                f"\n### `{chunk.video_code}` {chunk.title[:60]} — {chunk.author} · 入库 {_note_date(chunk)}"
            )
        lines.append(f"\n#### {chunk.heading_path or '概述'}\n\n{chunk.text}")
    lines.append("\n\n> 段落按相关性挑选并去重，不是笔记全文；需要上下文时用 `get_note_by_code`。")
    return "\n".join(lines)


# ======================== Tool: Get Note ========================

class GetNoteInput(BaseModel):
    note_id: int = Field(..., ge=1, description="笔记 ID")


@mcp.tool(name="get_note")
async def get_note(params: GetNoteInput) -> str:
    """获取完整笔记内容。"""
    entry = await asyncio.to_thread(store.get_by_id, params.note_id)
    if not entry:
        return f"❌ 未找到 ID 为 {params.note_id} 的笔记。"

    header = (
        f"# {entry['title']}\n\n"
        f"- **作者**: {entry['author']}\n"
        f"- **来源**: {entry['source_url']}\n"
        f"- **标签**: {entry['tags']}\n"
        f"- **创建时间**: {entry.get('timestamp') or entry['created_at']}\n"
    )
    if entry.get('user_requirement'):
        header += f"- **用户要求**: {entry['user_requirement']}\n"
    assets = await asyncio.to_thread(store.list_assets, entry["id"])
    if assets:
        header += (
            f"- **视频截图**: {len(assets)} 张；正文中的 `knowledge-asset://` "
            "引用可用 `get_note_image` 按需读取\n"
        )
    header += "\n---\n\n"
    return header + entry['summary_markdown']


@mcp.tool(name="get_note_by_code")
async def get_note_by_code(video_code: str) -> str:
    """通过视频码获取笔记。"""
    entry = await asyncio.to_thread(store.get_by_video_code, video_code)
    if not entry:
        return f"❌ 未找到视频码为 {video_code} 的笔记。"

    header = (
        f"# {entry['title']}\n\n"
        f"- **视频码**: `{entry['video_code']}`\n"
        f"- **作者**: {entry['author']}\n"
        f"- **来源**: {entry['source_url']}\n"
        f"- **标签**: {entry['tags']}\n"
        f"- **创建时间**: {entry.get('timestamp') or entry['created_at']}\n"
    )
    if entry.get('user_requirement'):
        header += f"- **用户要求**: {entry['user_requirement']}\n"
    assets = await asyncio.to_thread(store.list_assets, entry["id"])
    if assets:
        header += (
            f"- **视频截图**: {len(assets)} 张；正文中的 `knowledge-asset://` "
            "引用可用 `get_note_image` 按需读取\n"
        )
    header += "\n---\n\n"
    return header + entry['summary_markdown']


# ======================== Tools: Reviewed Images ========================

class ListNoteImagesInput(BaseModel):
    note_id: int = Field(..., ge=1, description="笔记 ID")


@mcp.tool(name="list_note_images")
async def list_note_images(params: ListNoteImagesInput) -> str:
    """列出笔记中已审核并持久化的视频截图，不暴露服务器文件路径。"""
    entry = await asyncio.to_thread(store.get_by_id, params.note_id)
    if not entry:
        return f"未找到 ID 为 {params.note_id} 的笔记。"
    assets = await asyncio.to_thread(store.list_assets, params.note_id)
    if not assets:
        return "这条笔记没有持久化视频截图。"
    lines = [f"## {entry['title']} · 视频截图 ({len(assets)} 张)\n"]
    for asset in assets:
        total_seconds = max(0, round(asset["timestamp_ms"] / 1000))
        minutes, seconds = divmod(total_seconds, 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"
        lines.append(
            f"- `{asset['asset_key']}` · {timestamp} · {asset['caption']}\n"
            f"  引用：`knowledge-asset://{entry['video_code']}/{asset['asset_key']}`"
        )
    return "\n".join(lines)


class GetNoteImageInput(BaseModel):
    video_code: str = Field(
        ...,
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9]+$",
        description="正文 knowledge-asset URI 中的视频码",
    )
    asset_id: str = Field(
        ...,
        pattern=r"^V\d{4}$",
        description="正文 knowledge-asset URI 中的图片 ID，例如 V0001",
    )


@mcp.tool(name="get_note_image")
async def get_note_image(params: GetNoteImageInput) -> Image:
    """按 Markdown 逻辑引用读取一张经审核的 JPEG，供多模态模型按需查看。"""
    try:
        payload = await asyncio.to_thread(
            store.read_asset_by_video_code,
            params.video_code,
            params.asset_id,
        )
    except (KeyError, ValueError, OSError) as exc:
        raise ValueError("图片不存在或完整性校验失败") from exc
    return Image(data=payload, format="jpeg")


# ======================== Tool: List ========================

class ListNotesInput(BaseModel):
    limit: int = Field(default=10, ge=1, le=100, description="返回数量")
    offset: int = Field(default=0, ge=0, le=100000, description="跳过前 N 条")


@mcp.tool(name="list_notes")
async def list_notes(params: ListNotesInput) -> str:
    """列出最近笔记。"""
    notes = await asyncio.to_thread(store.list_recent, params.limit, params.offset)
    if not notes:
        return "知识库暂无笔记。"

    lines = [f"## 最近笔记 (第 {params.offset+1}-{params.offset+len(notes)} 条)\n"]
    for n in notes:
        lines.append(
            f"- **[{n['id']}]** `{n['video_code']}` {n['title']} — _{n['author']}_ "
            f"({(n.get('timestamp') or n['created_at'])[:10]})"
        )
        if n['tags']:
            lines.append(f"  标签: {n['tags'][:80]}")
    return "\n".join(lines)


# ======================== Tool: Filter by Tag ========================

class TagFilterInput(BaseModel):
    tag: str = Field(..., min_length=1, max_length=100, description="标签关键词")
    limit: int = Field(default=10, ge=1, le=100)


@mcp.tool(name="list_by_tag")
async def list_by_tag(params: TagFilterInput) -> str:
    """按标签筛选笔记。"""
    notes = await asyncio.to_thread(store.list_by_tag, params.tag, params.limit)
    if not notes:
        return f"未找到包含标签 \"{params.tag}\" 的笔记。"

    lines = [f"## 标签 \"{params.tag}\" 相关笔记 ({len(notes)} 条)\n"]
    for n in notes:
        lines.append(
            f"- **[{n['id']}]** {n['title']} — _{n['author']}_ "
            f"({(n.get('timestamp') or n['created_at'])[:10]})"
        )
    return "\n".join(lines)


# ======================== Tool: Stats ========================

@mcp.tool(name="knowledge_stats")
async def knowledge_stats() -> str:
    """知识库统计。"""
    s = await asyncio.to_thread(store.stats)
    text = (
        f"## 知识库统计\n\n"
        f"- **总笔记数**: {s['total_entries']}\n"
        f"- **最新记录**: {s['latest_entry'] or '无'}\n"
        f"- **数据库路径**: {s['db_path']}\n"
    )
    try:
        i = await asyncio.to_thread(index.stats)
        text += (
            f"- **章节索引**: {i['indexed_notes']} 条笔记 / {i['chunks']} 段 / "
            f"{i['vectors']} 个向量（待向量化 {i['pending_embeddings']}，{i['model']}@{i['dimensions']}）\n"
        )
    except Exception:
        logger.exception("读取章节索引状态失败")
    return text


# ======================== Start ========================

if __name__ == "__main__":
    import sys
    if "--stdio" in sys.argv:
        print("MCP Server 启动 (stdio 模式)", file=sys.stderr)
        mcp.run(transport="stdio")
    else:
        print(f"MCP Server 启动 (Streamable HTTP) → http://{MCP_HOST}:{MCP_PORT}/mcp", file=sys.stderr)
        mcp.run(transport="streamable-http")
