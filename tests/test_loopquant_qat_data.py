from collections import Counter

import pytest

from loopquant.data import Window, training_order, training_windows


def test_qat_pool_excludes_tails_and_preserves_document_positions():
    windows = [
        Window("a", "a" * 40, "calibration", 512, tuple(range(7)), "b" * 64),
        Window("b", "a" * 40, "calibration", 0, tuple(range(9)), "c" * 64),
    ]
    result = training_windows(windows, 4)
    assert [(row.document_id, row.start, row.token_ids) for row in result] == [
        ("a", 512, (0, 1, 2, 3)),
        ("b", 0, (0, 1, 2, 3)),
        ("b", 4, (4, 5, 6, 7)),
    ]
    assert all(row.split == "qat_train" for row in result)
    with pytest.raises(ValueError, match="training documents"):
        training_windows([Window("dev", "a" * 40, "dev", 0, (1, 2, 3), "b" * 64)], 2)


def test_sampling_resumes_across_epoch_and_update_boundaries():
    expected = training_order(19, 17, 0, 96)
    actual = sum((training_order(19, 17, offset, 32) for offset in (0, 32, 64)), [])
    assert actual == expected
    assert Counter(expected[:95]) == {index: 5 for index in range(19)}
    assert expected[:19] != expected[19:38]
    assert expected != training_order(19, 23, 0, 96)
