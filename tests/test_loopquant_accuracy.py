import pytest

from loopquant.quality import QuestionCorrectness, accuracy_interval


def test_accuracy_interval_retains_question_pairing():
    matched = [QuestionCorrectness(str(i), i % 2 == 0, i % 2 == 0) for i in range(20)]
    assert accuracy_interval(matched) == (0.0, 0.0)
    opposed = [QuestionCorrectness(str(i), i % 2 == 0, i % 2 != 0) for i in range(20)]
    drop, upper = accuracy_interval(opposed)
    assert drop == 0 and upper > 0
    assert accuracy_interval([QuestionCorrectness(str(i), True, False) for i in range(20)]) == (
        1,
        1,
    )


def test_accuracy_interval_rejects_repeated_question_ids():
    with pytest.raises(ValueError, match="original question"):
        accuracy_interval([QuestionCorrectness("same", True, True)] * 2)
