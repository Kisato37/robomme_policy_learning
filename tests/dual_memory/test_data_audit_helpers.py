import numpy as np

from experiments.dual_memory.data_audit import array_summary, parse_grounding


def test_grounding_parser_uses_official_y_x_order():
    assert parse_grounding("grasp the peg at <80, 120>") == [(80, 120)]
    assert parse_grounding("no location") == []


def test_array_summary_records_shape_dtype_range_and_finiteness():
    summary = array_summary(np.array([[1.0, 2.0]], dtype=np.float32))
    assert summary == {
        "shape": [1, 2],
        "dtype": "float32",
        "min": 1.0,
        "max": 2.0,
        "finite": True,
    }
