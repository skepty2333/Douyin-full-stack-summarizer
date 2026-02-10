"""
抖音知识库 MCP Server

暴露知识库搜索/检索工具给 Claude (通过 Connector 功能)
部署后在 claude.ai -> Settings -> Connectors -> Add
填入: http://你的IP:8090/mcp
"""
import os
import logging
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field
from app.database.knowledge_store import KnowledgeStore
from app.config import KNOWLEDGE_DB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-knowledge")

# 初始化
store = KnowledgeStore(KNOWLEDGE_DB_PATH)

# 禁用 DNS rebinding 保护 (允许 Cloudflare Tunnel 访问)
security_settings = TransportSecuritySettings(enable_dns_rebinding_protection=False)
mcp = FastMCP("douyin_knowledge_mcp", host="0.0.0.0", port=8090, transport_security=security_settings)


# ======================== Tool: Search ========================

class SearchInput(BaseModel):
    query: str = Field(..., description="搜索关键词，支持中文。")
    limit: int = Field(default=5, description="返回结果数量上限")


@mcp.tool(name="search_notes")
async def search_notes(params: SearchInput) -> str:
    """在知识库中搜索视频笔记。"""
    results = store.search(params.query, params.limit)
    if not results:
        return f"未找到与 \"{params.query}\" 相关的笔记。"

    lines = [f"## 搜索结果: \"{params.query}\" ({len(results)} 条)\n"]
    for r in results:
        lines.append(
            f"### [{r['id']}] {r['title']}\n"
            f"- **视频码**: `{r['video_code']}`\n"
            f"- **作者**: {r['author']}\n"
            f"- **标签**: {r['tags'][:100]}\n"
            f"- **时间**: {r['created_at'][:10]}\n"
            f"- **摘要**: {r.get('snippet', '')[:200]}\n"
        )
    lines.append("\n> 💡 使用 `get_note` (ID) 或 `get_note_by_code` (视频码) 获取完整内容。")
    return "\n".join(lines)


# ======================== Tool: Get Note ========================

class GetNoteInput(BaseModel):
    note_id: int = Field(..., description="笔记 ID")


@mcp.tool(name="get_note")
async def get_note(params: GetNoteInput) -> str:
    """获取完整笔记内容。"""
    entry = store.get_by_id(params.note_id)
    if not entry:
        return f"❌ 未找到 ID 为 {params.note_id} 的笔记。"

    header = (
        f"# {entry['title']}\n\n"
        f"- **作者**: {entry['author']}\n"
        f"- **来源**: {entry['source_url']}\n"
        f"- **标签**: {entry['tags']}\n"
        f"- **创建时间**: {entry['created_at']}\n"
    )
    if entry.get('user_requirement'):
        header += f"- **用户要求**: {entry['user_requirement']}\n"
    header += "\n---\n\n"
    return header + entry['summary_markdown']


@mcp.tool(name="get_note_by_code")
async def get_note_by_code(video_code: str) -> str:
    """通过视频码获取笔记。"""
    entry = store.get_by_video_code(video_code)
    if not entry:
        return f"❌ 未找到视频码为 {video_code} 的笔记。"

    header = (
        f"# {entry['title']}\n\n"
        f"- **视频码**: `{entry['video_code']}`\n"
        f"- **作者**: {entry['author']}\n"
        f"- **来源**: {entry['source_url']}\n"
        f"- **标签**: {entry['tags']}\n"
        f"- **创建时间**: {entry['created_at']}\n"
    )
    if entry.get('user_requirement'):
        header += f"- **用户要求**: {entry['user_requirement']}\n"
    header += "\n---\n\n"
    return header + entry['summary_markdown']


# ======================== Tool: List ========================

class ListNotesInput(BaseModel):
    limit: int = Field(default=10, description="返回数量")
    offset: int = Field(default=0, description="跳过前 N 条")


@mcp.tool(name="list_notes")
async def list_notes(params: ListNotesInput) -> str:
    """列出最近笔记。"""
    notes = store.list_recent(params.limit, params.offset)
    if not notes:
        return "知识库暂无笔记。"

    lines = [f"## 最近笔记 (第 {params.offset+1}-{params.offset+len(notes)} 条)\n"]
    for n in notes:
        lines.append(
            f"- **[{n['id']}]** `{n['video_code']}` {n['title']} — _{n['author']}_ "
            f"({n['created_at'][:10]})"
        )
        if n['tags']:
            lines.append(f"  标签: {n['tags'][:80]}")
    return "\n".join(lines)


# ======================== Tool: Filter by Tag ========================

class TagFilterInput(BaseModel):
    tag: str = Field(..., description="标签关键词")
    limit: int = Field(default=10)


@mcp.tool(name="list_by_tag")
async def list_by_tag(params: TagFilterInput) -> str:
    """按标签筛选笔记。"""
    notes = store.list_by_tag(params.tag, params.limit)
    if not notes:
        return f"未找到包含标签 \"{params.tag}\" 的笔记。"

    lines = [f"## 标签 \"{params.tag}\" 相关笔记 ({len(notes)} 条)\n"]
    for n in notes:
        lines.append(
            f"- **[{n['id']}]** {n['title']} — _{n['author']}_ "
            f"({n['created_at'][:10]})"
        )
    return "\n".join(lines)


# ======================== Tool: Stats ========================

@mcp.tool(name="knowledge_stats")
async def knowledge_stats() -> str:
    """知识库统计。"""
    s = store.stats()
    return (
        f"## 知识库统计\n\n"
        f"- **总笔记数**: {s['total_entries']}\n"
        f"- **最新记录**: {s['latest_entry'] or '无'}\n"
        f"- **数据库路径**: {s['db_path']}\n"
    )


# ======================== Start ========================

if __name__ == "__main__":
    import sys
    port = int(os.getenv("MCP_PORT", "8090"))

    if "--stdio" in sys.argv:
        print("MCP Server 启动 (stdio 模式)", file=sys.stderr)
        mcp.run(transport="stdio")
    else:
        print(f"MCP Server 启动 (Streamable HTTP) → http://0.0.0.0:{port}/mcp", file=sys.stderr)
        mcp.run(transport="streamable-http")
