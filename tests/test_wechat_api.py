"""Offline payload tests for Enterprise WeChat message chunking."""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-test-key"}, clear=False):
    with patch("dotenv.load_dotenv", return_value=False):
        from app.services import wechat_api


class WeChatMarkdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_markdown_is_split_and_each_chunk_is_sent_offline(self) -> None:
        content = "# heading\n" + ("中" * 700) + "\n"
        sleep = AsyncMock()

        with patch.object(
            wechat_api,
            "_send_payload",
            new_callable=AsyncMock,
        ) as send_payload, patch.object(
            wechat_api,
            "asyncio",
            new=SimpleNamespace(sleep=sleep),
        ):
            await wechat_api.send_markdown_message("offline-user", content)

        payloads = [awaited.args[0] for awaited in send_payload.await_args_list]
        chunks = [payload["markdown"]["content"] for payload in payloads]

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), content)
        self.assertTrue(all(len(chunk.encode("utf-8")) <= 1800 for chunk in chunks))
        self.assertTrue(
            all(
                payload == {
                    "touser": "offline-user",
                    "msgtype": "markdown",
                    "agentid": wechat_api.AGENT_ID,
                    "markdown": {"content": chunk},
                }
                for payload, chunk in zip(payloads, chunks)
            )
        )
        self.assertEqual(send_payload.await_count, len(chunks))
        self.assertGreaterEqual(sleep.await_count, 1)

    async def test_partial_delivery_is_not_misreported_as_success(self) -> None:
        with patch.object(
            wechat_api,
            "get_access_token",
            new_callable=AsyncMock,
            return_value="offline-token",
        ), patch.object(
            wechat_api,
            "_request_json",
            new_callable=AsyncMock,
            return_value={
                "errcode": 0,
                "errmsg": "ok",
                "invaliduser": "offline-user",
            },
        ):
            with self.assertRaisesRegex(
                wechat_api.WeChatAPIError,
                "未投递目标",
            ):
                await wechat_api._send_payload(
                    {
                        "touser": "offline-user",
                        "msgtype": "text",
                        "agentid": wechat_api.AGENT_ID,
                        "text": {"content": "test"},
                    }
                )


if __name__ == "__main__":
    unittest.main()
