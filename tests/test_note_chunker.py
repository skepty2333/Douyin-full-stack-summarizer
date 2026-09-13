"""Offline tests for Markdown section chunking."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-test-key"}, clear=False):
    with patch("dotenv.load_dotenv", return_value=False):
        from app.services.note_chunker import (
            MAX_CHUNK_CHARS,
            chunk_note,
            clean_for_embedding,
            snippet,
        )


SAMPLE = """# 八个学习技巧

> 本文梳理了视频中的学习方法。

视频作者：某人

## 核心技巧

视频列出了以下技巧，每一项都有研究来源支持，这一段足够长以避免被当成碎片合并。

- **费曼技巧**：复述概念
- **间隔重复**：按遗忘曲线复习

![视频画面 00:04：画面展示交替学习法标题](knowledge-asset://czrci/V0003)

### 主动提取

不看材料主动回忆信息，比被动重读更能强化记忆，这是一条被反复验证的结论。

## 代码示例

```python
# 这不是标题
print("## 也不是标题")
```

以上代码只是示例，用来说明围栏内的井号不会被当作标题处理。
"""


class ChunkNoteTests(unittest.TestCase):
    def test_sections_follow_headings_and_keep_h2_context_for_h3(self) -> None:
        chunks = chunk_note("抖音原标题", SAMPLE)
        paths = [chunk.heading_path for chunk in chunks]
        self.assertEqual(paths[0], "")  # preamble
        self.assertIn("核心技巧", paths)
        self.assertIn("核心技巧 > 主动提取", paths)
        self.assertIn("代码示例", paths)
        self.assertEqual([chunk.index for chunk in chunks], list(range(len(chunks))))

    def test_headings_inside_code_fences_are_not_boundaries(self) -> None:
        chunks = chunk_note("t", SAMPLE)
        code_chunk = next(chunk for chunk in chunks if chunk.heading_path == "代码示例")
        self.assertIn('print("## 也不是标题")', code_chunk.text)
        self.assertNotIn("也不是标题", [chunk.heading_path for chunk in chunks])

    def test_embed_text_uses_h1_title_and_cleans_markdown_but_text_keeps_original(self) -> None:
        chunks = chunk_note("抖音原标题 #tag", SAMPLE)
        core = next(chunk for chunk in chunks if chunk.heading_path == "核心技巧")
        self.assertTrue(core.embed_text.startswith("八个学习技巧\n核心技巧\n"))
        self.assertIn("knowledge-asset://czrci/V0003", core.text)
        self.assertNotIn("knowledge-asset://", core.embed_text)
        self.assertIn("[图: 视频画面 00:04：画面展示交替学习法标题]", core.embed_text)
        self.assertNotIn("**", core.embed_text)

    def test_hash_is_stable_and_changes_with_content(self) -> None:
        first = chunk_note("t", SAMPLE)
        second = chunk_note("t", SAMPLE)
        self.assertEqual(
            [chunk.embed_text_hash for chunk in first],
            [chunk.embed_text_hash for chunk in second],
        )
        changed = chunk_note("t", SAMPLE.replace("主动回忆信息", "主动提取信息"))
        self.assertNotEqual(
            [chunk.embed_text_hash for chunk in first],
            [chunk.embed_text_hash for chunk in changed],
        )

    def test_long_sections_are_split_under_the_limit(self) -> None:
        paragraphs = "\n\n".join(f"第{i}段。" + "内容" * 150 for i in range(6))
        chunks = chunk_note("t", f"# 标题\n\n## 长节\n\n{paragraphs}")
        long_chunks = [chunk for chunk in chunks if chunk.heading_path == "长节"]
        self.assertGreater(len(long_chunks), 1)
        self.assertTrue(all(chunk.char_count <= MAX_CHUNK_CHARS for chunk in long_chunks))
        self.assertEqual("".join(c.text for c in long_chunks).count("第"), 6)

    def test_short_headed_section_stays_its_own_chunk(self) -> None:
        markdown = "# 标题\n\n## 正文\n\n" + "这是一段足够长的正文。" * 5 + "\n\n## 小节\n\n完。"
        chunks = chunk_note("t", markdown)
        self.assertEqual([chunk.heading_path for chunk in chunks], ["正文", "小节"])

    def test_stray_fragment_after_split_merges_backwards(self) -> None:
        # A long section that splits into (big, tiny) keeps the tail attached.
        body = "内容" * 590 + "\n\n尾。"
        chunks = chunk_note("t", f"# 标题\n\n## 长节\n\n{body}")
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].text.endswith("尾。"))

    def test_plain_text_without_headings_is_one_chunk(self) -> None:
        chunks = chunk_note("t", "只有一段没有标题的正文，长度也足够作为一个块。" * 2)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].heading_path, "")

    def test_empty_markdown_yields_no_chunks(self) -> None:
        self.assertEqual(chunk_note("t", ""), [])
        self.assertEqual(chunk_note("t", "# 只有标题"), [])


class HelperTests(unittest.TestCase):
    def test_clean_for_embedding_strips_formatting(self) -> None:
        cleaned = clean_for_embedding("- **加粗** [链接](http://x)\n\n> 引用\n\n|---|---|")
        self.assertEqual(cleaned, "加粗 链接\n\n引用")

    def test_snippet_is_single_line_and_bounded(self) -> None:
        text = "第一行**重点**\n\n第二行" + "很长" * 100
        result = snippet(text, limit=30)
        self.assertNotIn("\n", result)
        self.assertLessEqual(len(result), 30)
        self.assertTrue(result.endswith("…"))


if __name__ == "__main__":
    unittest.main()
