from experiments.loopkv.text_stops import ByteStops


def test_stops_cross_token_boundaries_and_preserve_earliest_offset():
    matcher = ByteStops(
        (b"reasoning", b" Q", b":later</s>", b"\xf0\x9f", b"\x98\x80"), ("Q:", "</s>")
    )
    assert matcher.update([3]) is None
    assert matcher.update([3, 4, 0, 1]) is None
    assert matcher.update([3, 4, 0, 1, 2]) == ("Q:", 14)


def test_repeated_cumulative_output_does_not_append_twice():
    matcher = ByteStops((b"Q", b":"), ("Q:",))
    assert matcher.update([0]) is None
    assert matcher.update([0]) is None
    assert matcher.update([0, 1]) == ("Q:", 0)
