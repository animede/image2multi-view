# Copyright 2026 YUZUKI
# SPDX-License-Identifier: MIT
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viewprompt import build_prompt, normalize_angles, snap  # noqa: E402


def test_snap_wraps_and_rounds():
    assert snap(0) == 0
    assert snap(14) == 10
    assert snap(45) == 50 and snap(-45) == -40
    assert snap(-180) == 180
    assert snap(190) == -170
    assert snap(180) == 180


def test_pitch_folds_over_zenith():
    assert normalize_angles(0, 0, 0) == (0, 0, 0)
    assert normalize_angles(0, 90, 0) == (0, 90, 0)
    assert normalize_angles(0, 100, 0) == (180, 80, 180)
    assert normalize_angles(30, 180, 0) == (-150, 0, 180)
    assert normalize_angles(0, -120, 40) == (180, -60, -140)


def test_prompt_mentions_direction_words():
    text, angles = build_prompt(90, 0, 0)
    assert angles == (90, 0, 0)
    assert "true side profile" in text and "right edge" in text
    text, _ = build_prompt(-45, 0, 0)
    assert "their own right" in text and "left side of the image" in text
    text, _ = build_prompt(0, 60, 0)
    assert "high-angle" in text
    text, _ = build_prompt(0, 0, -30)
    assert "counter-clockwise" in text
    text, _ = build_prompt(180, 0, 0, subject="object")
    assert "rear view" in text


def test_every_step_builds():
    for s in ("person", "object"):
        for y in range(-180, 181, 10):
            for p in range(-180, 181, 10):
                for r in (-180, -90, 0, 90, 180):
                    text, _ = build_prompt(y, p, r, subject=s)
                    assert len(text) < 4000


def test_roll_left_out_when_client_rotates():
    text, angles = build_prompt(0, 0, 90, roll_in_prompt=False)
    assert "level" in text and angles == (0, 0, 90)
