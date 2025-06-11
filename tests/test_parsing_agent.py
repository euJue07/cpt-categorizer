import pytest
from cpt_categorizer.agents.parsing_agent import parse


def test_basic_cleanup():
    input_text = "2D ECHO (with doppler), code 0004"
    cleaned = parse(input_text)
    assert "echocardiogram" in cleaned.lower()
    assert "code" not in cleaned.lower()
    assert "(" not in cleaned  # should clean punctuation


@pytest.mark.parametrize(
    "raw,expect",
    [
        ("PT, CBC, xray - L arm", "prothrombin"),
        ("ultrasound pelvis, v2 (hospital copy)", "ultrasound of pelvis"),
    ],
)
def test_multiple_cases(raw, expect):
    cleaned = parse(raw)
    assert expect.lower() in cleaned.lower()
