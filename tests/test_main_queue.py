"""Offline state-transition tests for the per-user job queue."""
from __future__ import annotations

import logging
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


# Import main without consulting .env, opening the production knowledge DB, or
# requiring real Enterprise WeChat callback credentials.
with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-test-key"}, clear=False):
    with patch("dotenv.load_dotenv", return_value=False):
        import app.database.knowledge_store as knowledge_store_module
        import app.utils.wechat_crypto as wechat_crypto_module

        with patch.object(
            knowledge_store_module,
            "KnowledgeStore",
            return_value=MagicMock(),
        ), patch.object(
            wechat_crypto_module,
            "WXBizMsgCrypt",
            return_value=MagicMock(),
        ), patch.object(logging, "basicConfig"):
            import main as main_module


class MainQueueStateTests(unittest.TestCase):
    def setUp(self) -> None:
        main_module._pending.clear()
        main_module._background_tasks.clear()

    def tearDown(self) -> None:
        main_module._pending.clear()
        main_module._background_tasks.clear()

    @staticmethod
    def task(user_id: str, job_id: str) -> main_module.PendingTask:
        return main_module.PendingTask(
            user_id=user_id,
            share_url="https://v.douyin.com/offline/",
            share_text="offline test",
            job_id=job_id,
        )

    def test_job_can_only_be_claimed_once(self) -> None:
        user_id = "queue-user"
        active = self.task(user_id, "active_job_001")
        main_module._pending[user_id] = main_module.UserTaskQueue(active=active)

        self.assertIsNone(main_module._claim_for_processing(user_id, "stale_job_999"))
        first_claim = main_module._claim_for_processing(user_id, active.job_id)
        duplicate_claim = main_module._claim_for_processing(user_id, active.job_id)

        self.assertIs(first_claim, active)
        self.assertTrue(active.processing)
        self.assertIsNone(duplicate_claim)

    def test_stale_advance_does_not_change_active_or_waiting_jobs(self) -> None:
        user_id = "queue-user"
        active = self.task(user_id, "active_job_001")
        waiting = self.task(user_id, "waiting_job_002")
        queue = main_module.UserTaskQueue(active=active, queue=[waiting])
        main_module._pending[user_id] = queue

        with patch.object(main_module, "_spawn_background") as spawn:
            with self.assertLogs("douyin-bot", level="WARNING") as logs:
                main_module._advance_queue(user_id, "expired_job_000")

        self.assertIs(main_module._pending[user_id], queue)
        self.assertIs(queue.active, active)
        self.assertEqual(queue.queue, [waiting])
        spawn.assert_not_called()
        self.assertTrue(any("忽略过期队列推进" in line for line in logs.output))

    def test_matching_advance_activates_and_schedules_next_job(self) -> None:
        user_id = "queue-user"
        completed = self.task(user_id, "completed_job_001")
        next_task = self.task(user_id, "next_job_002")
        later_task = self.task(user_id, "later_job_003")
        queue = main_module.UserTaskQueue(
            active=completed,
            queue=[next_task, later_task],
        )
        main_module._pending[user_id] = queue
        scheduled = object()
        notify = MagicMock(return_value=scheduled)

        with patch.object(main_module, "_advance_and_notify", new=notify), patch.object(
            main_module,
            "_spawn_background",
        ) as spawn:
            main_module._advance_queue(user_id, completed.job_id)

        self.assertIs(queue.active, next_task)
        self.assertEqual(queue.queue, [later_task])
        self.assertFalse(next_task.processing)
        notify.assert_called_once_with(
            user_id,
            next_task.job_id,
            "开始处理队列中的下一个视频。剩余排队: 1个。",
        )
        spawn.assert_called_once_with(scheduled)


if __name__ == "__main__":
    unittest.main()


class AlbumTaskFlowTests(unittest.IsolatedAsyncioTestCase):
    """图文 posts flow through the same queue with their own summarizer and paths."""

    USER = "album-user"
    JOB = "album_job_0001"

    def setUp(self) -> None:
        main_module._pending.clear()
        main_module._background_tasks.clear()

    def tearDown(self) -> None:
        main_module._pending.clear()
        main_module._background_tasks.clear()

    def _album_task(self, work_dir: str) -> main_module.PendingTask:
        task = main_module.PendingTask(
            user_id=self.USER,
            share_url="https://v.douyin.com/VNv0Dv0R5VA/",
            share_text="6.69 复制打开抖音，看看【新用户1401的图文作品】量化策略思路分享。",
            job_id=self.JOB,
        )
        task.parsed_video_id = "7687268235284090154"
        task.parsed_title = "量化策略思路分享。摘要"
        task.parsed_author = "新用户1401"
        task.parsed_media_kind = "images"
        task.parsed_image_paths = [f"{work_dir}/images/image_01.jpg", f"{work_dir}/images/image_02.jpg"]
        task.parsed_caption = "量化策略思路分享。摘要 交易的直接问题……"
        task.parsed_work_dir = work_dir
        main_module._pending[self.USER] = main_module.UserTaskQueue(active=task)
        return task

    async def test_init_records_album_fields_and_hands_off_to_execution(self) -> None:
        task = main_module.PendingTask(
            user_id=self.USER,
            share_url="https://v.douyin.com/VNv0Dv0R5VA/",
            share_text="share",
            job_id=self.JOB,
        )
        main_module._pending[self.USER] = main_module.UserTaskQueue(active=task)
        resolved = {
            "video_id": "7687268235284090154",
            "title": "量化策略思路分享。摘要",
            "author": "新用户1401",
            "published_at": "2026-09-20T10:26:40+00:00",
            "caption": "量化策略思路分享。摘要 交易的直接问题……",
            "job_id": self.JOB,
            "work_dir": "/tmp/douyin-bot/jobs/album_job_0001",
            "media_kind": "images",
            "video_path": "",
            "video_url": "",
            "image_paths": [
                "/tmp/douyin-bot/jobs/album_job_0001/images/image_01.jpg",
                "/tmp/douyin-bot/jobs/album_job_0001/images/image_02.jpg",
            ],
        }

        with patch.object(
            main_module, "resolve_and_download", new_callable=AsyncMock, return_value=resolved
        ), patch.object(
            main_module.knowledge_db, "get_by_title_and_author", return_value=[]
        ), patch.object(
            main_module, "_execute_summary_task", new_callable=AsyncMock
        ) as execute:
            await main_module._process_task_init(self.USER, self.JOB, already_claimed=True)

        execute.assert_awaited_once()
        handed_off = execute.await_args.args[1]
        self.assertIs(handed_off, task)
        self.assertEqual(task.parsed_media_kind, "images")
        self.assertEqual(task.parsed_image_paths, resolved["image_paths"])
        self.assertEqual(task.parsed_caption, resolved["caption"])
        self.assertEqual(task.parsed_work_dir, resolved["work_dir"])
        self.assertEqual(task.parsed_video_path, "")
        self.assertEqual(task.parsed_media_url, "")
        self.assertEqual(task.parsed_published_at, "2026-09-20T10:26:40+00:00")

    async def test_execution_uses_album_summarizer_work_dir_pdf_and_album_wording(self) -> None:
        from app.services.ai_summarizer import PipelineDiagnostics, SummaryResult

        summary = SummaryResult(
            markdown="# 量化策略思路分享\n\n正文",
            video_frames=(),
            diagnostics=PipelineDiagnostics(
                duration_seconds=None,
                policy_name="compact",
                transcript_char_count=120,
                timestamp_quality="image_sequence",
                visual_annotation_count=2,
                approved_screenshot_count=0,
                research_char_count=10,
                final_char_count=20,
            ),
        )
        messages: list[str] = []
        pdf_paths: list[str] = []
        saved_entries: list[object] = []

        async def record_text(user_id: str, content: str) -> bool:
            messages.append(content)
            return True

        def fake_generate_pdf(markdown: str, path: str) -> bool:
            pdf_paths.append(path)
            return True

        def fake_save(entry, allow_overwrite, assets):
            saved_entries.append(entry)
            return 1

        with tempfile.TemporaryDirectory() as work_dir:
            task = self._album_task(work_dir)
            with patch.object(
                main_module, "summarize_album_with_artifacts", new_callable=AsyncMock, return_value=summary
            ) as album, patch.object(
                main_module, "summarize_with_artifacts", new_callable=AsyncMock
            ) as video, patch.object(
                main_module, "generate_video_code", return_value="ab12c"
            ), patch.object(
                main_module, "classify_note", new_callable=AsyncMock, return_value=None
            ), patch.object(
                main_module, "generate_tags_with_ai", new_callable=AsyncMock, return_value="量化,策略"
            ), patch.object(
                main_module.knowledge_db, "save", side_effect=fake_save
            ), patch.object(
                main_module, "_spawn_background", side_effect=lambda coro: coro.close()
            ), patch.object(
                main_module, "generate_pdf", side_effect=fake_generate_pdf
            ), patch.object(
                main_module, "upload_temp_media", new_callable=AsyncMock, return_value="media-id"
            ), patch.object(
                main_module, "send_file_message", new_callable=AsyncMock
            ) as send_file, patch.object(
                main_module, "_safe_send_text", side_effect=record_text
            ), patch.object(
                main_module, "_cleanup_pending_files"
            ), patch.object(
                main_module, "_advance_queue"
            ):
                await main_module._execute_summary_task(self.USER, task)

            video.assert_not_awaited()
            album.assert_awaited_once()
            self.assertEqual(album.await_args.args[0], task.parsed_image_paths)
            self.assertEqual(album.await_args.args[1], "量化策略思路分享。摘要")
            self.assertEqual(album.await_args.args[2], "新用户1401")
            self.assertEqual(album.await_args.kwargs["caption"], task.parsed_caption)
            self.assertEqual(album.await_args.kwargs["work_dir"], work_dir)
            self.assertEqual(pdf_paths, [os.path.join(work_dir, "summary.pdf")])
            send_file.assert_awaited_once_with(self.USER, "media-id")
            self.assertTrue(messages[0].startswith("图文: 量化策略思路分享。摘要\n作者: 新用户1401\n视频码: ab12c"))
            self.assertEqual(len(saved_entries), 1)
            self.assertEqual(saved_entries[0].duration_seconds, 0.0)
            self.assertEqual(saved_entries[0].video_id, "7687268235284090154")
            self.assertEqual(saved_entries[0].tags, "量化,策略")
