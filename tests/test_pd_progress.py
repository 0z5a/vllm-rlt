"""Ready handoffs must progress before potentially blocking compute submission."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_rlt.pd.config import PDConfig
from vllm_rlt.pd.worker import PDWorker, Work
from vllm_rlt.request import Stage


@pytest.fixture
def prefill_worker():
    # Exercise owner-loop ordering without CUDA or a NIXL installation.
    worker = object.__new__(PDWorker)
    worker.role = "prefill"
    worker.config = PDConfig()
    worker.cache = Mock()
    worker.connector = Mock(inflight_bytes=0)
    worker.connector.poll.return_value = ([], [])
    worker.connector.submit.return_value = True
    worker.transfer_cursor = 0
    worker.send = Mock()
    worker.engine = Mock()
    worker.engine.scheduler.queues = {Stage.PREFILL: deque(["next"])}
    # Stop exactly at submission; the test does not execute GPU computation.
    worker.engine.model_runner.submit.side_effect = StopIteration("compute submitted")
    ready = Mock()
    ready.query.return_value = True
    worker.work = {
        "handoff": Work(
            "handoff",
            SimpleNamespace(),
            0,
            "decode",
            chunks=deque((ready, i, [], [], 1024) for i in range(3)),
            next_sequence=3,
            compute_done=True,
            compute_released=True,
            last_event=ready,
        ),
        "next": Work("next", SimpleNamespace(), 1, "decode"),
    }
    return worker


def test_ready_handoff_drains_before_compute_even_with_transfer_backpressure(prefill_worker):
    worker = prefill_worker
    # A rejected submission leaves the ready chunk queued. Returning to the
    # owner loop lets it poll completion/credits and process cancellation.
    worker.connector.submit.return_value = False
    worker.progress()
    worker.prefill_step()
    assert len(worker.work["handoff"].chunks) == 3
    worker.engine.scheduler._take.assert_not_called()

    worker.connector.submit.return_value = True
    for remaining in (2, 1):
        worker.progress()
        worker.prefill_step()
        assert len(worker.work["handoff"].chunks) == remaining
        worker.engine.scheduler._take.assert_not_called()
        worker.send.assert_not_called()

    worker.progress()
    assert not worker.work["handoff"].chunks
    worker.send.assert_called_once_with("commit", tid="handoff", count=3)
    with pytest.raises(StopIteration, match="compute submitted"):
        worker.prefill_step()
    worker.engine.scheduler._take.assert_called_once_with(Stage.PREFILL)


def test_unready_handoff_does_not_prevent_other_prefill(prefill_worker):
    worker = prefill_worker
    worker.work["handoff"].last_event.query.return_value = False
    with pytest.raises(StopIteration, match="compute submitted"):
        worker.prefill_step()
    worker.engine.scheduler._take.assert_called_once_with(Stage.PREFILL)
    assert len(worker.work["handoff"].chunks) == 3
