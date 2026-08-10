"""
抖音知识库 MCP Server

通过 Streamable HTTP 或 stdio 向 MCP 客户端提供知识库文字检索和按需取图。
HTTP 模式默认仅监听 127.0.0.1；远程访问必须经过带认证的 HTTPS
反向代理、VPN 或受控隧道，不直接暴露 8090 端口。
"""
import logging
import asyncio
from mcp.server.fastmcp import FastMCP, Image
from pydantic import BaseModel, Field
from app.database.knowledge_store import KnowledgeStore
from app.config import KNOWLEDGE_DB_PATH, MCP_HOST, MCP_PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-knowledge")

# 初始化
store = KnowledgeStore(KNOWLEDGE_DB_PATH)

# 默认只监听本机，并保留 FastMCP 的 DNS rebinding 防护。
# 如需远程访问，请通过带认证的反向代理或受控隧道暴露。
mcp = FastMCP("douyin_knowledge_mcp", host=MCP_HOST, port=MCP_PORT)


# ======================== Tool: Search ========================

class SearchInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=200, description="搜索关键词，支持中文。支持多个关键词空格分隔，将匹配标签、标题和正文。")
    limit: int = Field(default=100, ge=1, le=100, description="返回结果数量上限")


@mcp.tool(name="search_notes")
async def search_notes(params: SearchInput) -> str:
    """在知识库中搜索视频笔记。优先匹配标签，也搜索标题和正文。多个关键词用空格分隔。"""
    results = await asyncio.to_thread(store.search, params.query, params.limit)
    if not results:
        return f"未找到与 \"{params.query}\" 相关的笔记。"

    lines = [f"## 搜索结果: \"{params.query}\" ({len(results)} 条)\n"]
    for r in results:
        lines.append(
            f"### [{r['id']}] {r['title'][:60]}\n"
            f"- **视频码**: `{r['video_code']}`\n"
            f"- **作者**: {r['author']}\n"
            f"- **标签**: {r['tags'][:120]}\n"
            f"- **时间**: {r.get('timestamp') or r['created_at'][:19]}\n"
            f"- **摘要**: {r.get('snippet', '')[:200]}\n"
        )
    lines.append("\n> 使用 `get_note` (ID) 或 `get_note_by_code` (视频码) 获取完整内容。")
    return "\n".join(lines)


# ======================== Tool: Precise Search ========================

class PreciseSearchInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=200, description="搜索关键词，空格分隔。所有关键词必须同时出现才会命中。")
    limit: int = Field(default=20, ge=1, le=100, description="返回结果数量上限")


@mcp.tool(name="search_notes_precise")
async def search_notes_precise(params: PreciseSearchInput) -> str:
    """精确搜索：所有关键词必须同时出现在标签、标题或正文中（AND逻辑）。适合缩小范围、精确定位。"""
    results = await asyncio.to_thread(store.search_precise, params.query, params.limit)
    if not results:
        return f"未找到同时包含所有关键词 \"{params.query}\" 的笔记。"

    lines = [f"## 精确搜索: \"{params.query}\" ({len(results)} 条)\n"]
    for r in results:
        lines.append(
            f"### [{r['id']}] {r['title'][:60]}\n"
            f"- **视频码**: `{r['video_code']}`\n"
            f"- **作者**: {r['author']}\n"
            f"- **标签**: {r['tags'][:120]}\n"
            f"- **时间**: {r.get('timestamp') or r['created_at'][:19]}\n"
            f"- **摘要**: {r.get('snippet', '')[:200]}\n"
        )
    lines.append("\n> 使用 `get_note` (ID) 或 `get_note_by_code` (视频码) 获取完整内容。")
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
    return (
        f"## 知识库统计\n\n"
        f"- **总笔记数**: {s['total_entries']}\n"
        f"- **最新记录**: {s['latest_entry'] or '无'}\n"
        f"- **数据库路径**: {s['db_path']}\n"
    )


# ======================== Start ========================

if __name__ == "__main__":
    import sys
    if "--stdio" in sys.argv:
        print("MCP Server 启动 (stdio 模式)", file=sys.stderr)
        mcp.run(transport="stdio")
    else:
        print(f"MCP Server 启动 (Streamable HTTP) → http://{MCP_HOST}:{MCP_PORT}/mcp", file=sys.stderr)
        mcp.run(transport="streamable-http")
