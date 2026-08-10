"""Offline state-transition tests for the per-user job queue."""
from __future__ import annotations

import logging
import os
import unittest
from unittest.mock import MagicMock, patch


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
