# Copyright 2026 YUZUKI
# SPDX-License-Identifier: MIT
"""Turn (yaw, pitch, roll) in 10-degree steps into a full FLUX 3 edit caption.

The prompt upsampler is bypassed for speed, so the text here must already be a complete
caption: what to keep, how the subject is turned, where the camera sits, how the frame is
rolled, how the subject is framed. Direction conventions follow flux3-klein-studio's
character sheet: a positive yaw turns the subject to *their own* left, so the nose points
to the right edge of the image and the camera sees the right side of the body.

Axes (all in degrees, multiples of 10, -180..180):
  yaw   (Y): subject turned to their own left (+) / right (-). +-180 = seen from behind.
  pitch (X): camera above looking down (+) / below looking up (-). |pitch| > 90 is folded
             over the zenith: the camera ends up on the far side, upside down.
  roll  (Z): the picture rotated clockwise (+) / counter-clockwise (-) in the frame.
"""

from __future__ import annotations

import math

SUBJECTS = {
    "person": {
        "noun": "character",
        "keep": "face, hair, outfit, accessories, body proportions, colors and art style",
    },
    "object": {
        "noun": "object",
        "keep": "shape, proportions, materials, colors, surface details and markings",
    },
}
FRAMINGS = ("same", "full", "closeup")

_STEP = 10


def snap(value: float) -> int:
    """Round to the nearest 10 degrees and wrap into (-180, 180]."""
    v = int(math.floor(float(value) / _STEP + 0.5)) * _STEP  # 45 -> 50, not banker's rounding
    v = ((v + 180) % 360) - 180
    return 180 if v == -180 else v


def normalize_angles(yaw: float, pitch: float, roll: float) -> tuple[int, int, int]:
    """Snap to 10 degrees and fold a pitch beyond +-90 over the zenith / nadir.

    Going past straight-above puts the camera on the opposite side of the subject, upside
    down, so (yaw, 180-pitch, roll) with yaw and roll shifted by 180 describes the same view
    with a pitch the caption can state plainly.
    """
    y, p, r = snap(yaw), snap(pitch), snap(roll)
    if p > 90:
        p, y, r = 180 - p, snap(y + 180), snap(r + 180)
    elif p < -90:
        p, y, r = -180 - p, snap(y + 180), snap(r + 180)
    return y, p, r


# --- yaw -------------------------------------------------------------------------------

_LEFT_ON_IMAGE_RIGHT = "The character's own left side is on the right of the image, as in the first image."
_LEFT_ON_IMAGE_LEFT = ("Seen from behind, the character's own left side is now on the left of the image, so "
                       "one-sided details stay on the same body side as in the first image.")
_BACK_ONLY = ("Show the back of the outfit as it looks from behind; front-only details such as the face, chest "
              "prints, front pockets, bib, buttons and zippers are not visible.")


def _yaw_person(yaw: int) -> str:
    a = abs(yaw)
    side = "left" if yaw > 0 else "right"        # the side the character turns toward
    edge = "right" if yaw > 0 else "left"        # the image edge the nose points to
    seen = "right" if yaw > 0 else "left"        # the side of the body the camera sees
    turned = f"The character has turned {a} degrees to their own {side}"
    if a == 0:
        return ("A straight front view: the character faces the camera directly, both eyes and the whole face "
                "are visible and the shoulders are square to the camera. " + _LEFT_ON_IMAGE_RIGHT)
    if a <= 20:
        return (f"{turned}, only slightly away from a straight front view: the face still points almost at the "
                f"camera and both eyes are fully visible, but the nose points a little toward the {edge} edge of "
                f"the image and the {seen} shoulder comes slightly forward. " + _LEFT_ON_IMAGE_RIGHT)
    if a == 30:
        return (f"{turned}: a shallow three-quarter view, the face and body angled toward the {edge} side of the "
                f"image, both eyes visible, the far side of the face foreshortened, the {seen} shoulder forward and "
                f"the far shoulder behind. " + _LEFT_ON_IMAGE_RIGHT)
    if a <= 50:
        return (f"{turned}, exactly halfway between the front view and the side profile: the face and body are "
                f"turned toward the {edge} side of the image, the shoulders and hips are at a clear diagonal, the "
                f"near shoulder is in front and the far arm is partly hidden behind the body. " + _LEFT_ON_IMAGE_RIGHT)
    if a == 60:
        return (f"{turned}: a deep three-quarter view, closer to a profile than to the front. The face points well "
                f"toward the {edge} edge of the image, the far eye is only just visible at the edge of the face, "
                f"the shoulders are strongly angled and we mostly see the {seen} side of the body. " + _LEFT_ON_IMAGE_RIGHT)
    if a <= 80:
        return (f"{turned}, almost a side profile: the nose points toward the {edge} edge of the image, the far eye "
                f"is hidden and only a thin strip of the far cheek shows beyond the nose; we see the {seen} side of "
                f"the body. " + _LEFT_ON_IMAGE_RIGHT)
    if a == 90:
        return (f"{turned}: a true side profile, the nose points to the {edge} edge of the image and only one eye is "
                f"visible; we see the {seen} side of the body, the shoulders are seen edge-on.")
    if a <= 110:
        return (f"{turned}, just past a side profile: the head is seen slightly from behind, the ear and the back of "
                f"the head are visible, the tip of the nose barely shows at the {edge} edge of the face and the far "
                f"eye is not visible; we see the {seen} side of the body and part of the back. " + _LEFT_ON_IMAGE_LEFT)
    if a <= 150:
        return (f"{turned}: a three-quarter rear view, we mostly see the back and the back of the head, which is "
                f"turned toward the {edge} side of the image; only the edge of the cheek shows on the {edge}, the "
                f"face is not visible. {_BACK_ONLY} {_LEFT_ON_IMAGE_LEFT}")
    if a <= 170:
        return (f"{turned}, almost directly from behind: a rear view showing the back of the head and the back of "
                f"the outfit, the head angled only slightly toward the {edge} side so a sliver of the cheek is just "
                f"visible; the face is not visible. {_BACK_ONLY} {_LEFT_ON_IMAGE_LEFT}")
    return ("The character has turned around completely: a rear view showing the back of the head, the back of "
            f"the outfit and the back of the legs; the face is not visible. {_BACK_ONLY} {_LEFT_ON_IMAGE_LEFT}")


def _yaw_object(yaw: int) -> str:
    a = abs(yaw)
    side = "left" if yaw > 0 else "right"
    edge = "right" if yaw > 0 else "left"
    seen = "right" if yaw > 0 else "left"
    rotated = (f"The object is rotated {a} degrees to its own {side} (the camera has moved {a} degrees around "
               f"the object toward its {seen} side)")
    if a == 0:
        return ("A straight front view: the front of the object faces the camera directly and symmetrically, "
                "neither side is visible.")
    if a <= 20:
        return (f"{rotated}: mostly the front is visible, with a thin slice of the {seen} side appearing at the "
                f"{edge} edge, the front face slightly foreshortened.")
    if a == 30:
        return (f"{rotated}: a three-quarter view in which the front dominates and the {seen} side is clearly "
                f"visible, the front face angled toward the {edge} side of the image.")
    if a <= 50:
        return (f"{rotated}: a classic three-quarter view showing the front and the {seen} side equally, the "
                f"front face angled toward the {edge} side of the image, the nearest corner closest to the camera.")
    if a == 60:
        return (f"{rotated}: a three-quarter view in which the {seen} side dominates and the front is strongly "
                f"foreshortened, pointing toward the {edge} edge of the image.")
    if a <= 80:
        return (f"{rotated}: almost a pure side view, the {seen} side fills the view and the front is only a "
                f"narrow sliver at the {edge} edge.")
    if a == 90:
        return (f"{rotated}: a true side view, the camera sees the {seen} side square on, the front points toward "
                f"the {edge} edge of the image and neither the front nor the back face is visible.")
    if a <= 110:
        return (f"{rotated}: just past a side view, the {seen} side is visible together with a narrow slice of the "
                f"back; the front is hidden and the front end points toward the {edge} edge of the image.")
    if a <= 150:
        return (f"{rotated}: a three-quarter rear view, the back and the {seen} side are visible, the front is "
                f"hidden and the front end points away toward the {edge} side of the image.")
    if a <= 170:
        return (f"{rotated}: almost directly from behind, mainly the back is visible with a thin slice of the "
                f"{seen} side at the {edge} edge.")
    return "A straight rear view: the back of the object faces the camera directly and the front is completely hidden."


# --- pitch -----------------------------------------------------------------------------

def _pitch(pitch: int, subject: str) -> str:
    noun = SUBJECTS[subject]["noun"]
    if subject == "person":
        top, bottom = "the top of the head and the shoulders", "the chin, the underside of the jaw and the soles"
    else:
        top, bottom = "the top surface of the object", "the underside of the object"
    a = abs(pitch)
    if a == 0:
        level = "at eye level with the character" if subject == "person" else "level with the center of the object"
        return f"The camera is {level}, looking straight ahead with no tilt up or down; the horizon is at the middle."
    if pitch > 0:
        if a <= 20:
            return (f"The camera is slightly above, tilted down about {a} degrees: {top} are a little more visible "
                    f"than at eye level and the ground is visible behind the {noun}.")
        if a <= 50:
            return (f"A high-angle shot from about {a} degrees above: the camera looks clearly down on the {noun}, "
                    f"{top} appear large while the lower parts are foreshortened and recede downward; the floor "
                    f"fills the background.")
        if a <= 80:
            return (f"A steep high-angle shot from {a} degrees above, almost overhead: we mainly see {top}, the "
                    f"{noun} is strongly foreshortened vertically and the floor is the whole background.")
        return (f"A direct top-down bird's-eye view, the camera straight above looking down at 90 degrees: only "
                f"{top} are visible, the floor is the entire background and there is no horizon.")
    near, far = ("the feet and legs", "the head") if subject == "person" else ("the base", "the top")
    if a <= 20:
        return (f"A slight low-angle shot: the camera is placed below the {noun}'s middle and tilted up about {a} "
                f"degrees, so {bottom} become visible, {near} are nearest to the camera and look a little larger, "
                f"and the ceiling or sky appears behind the upper part of the {noun}.")
    if a <= 50:
        return (f"A low-angle shot from about {a} degrees below, the camera close to the ground pointing up: strong "
                f"upward perspective, {near} are nearest and largest, {far} is farther and smaller, {bottom} are "
                f"clearly visible, the {noun} towers over the viewer and the ceiling or sky fills the background "
                f"behind it; no floor is visible.")
    if a <= 80:
        return (f"A steep low-angle shot from {a} degrees below, the camera on the ground almost directly "
                f"underneath, looking straight up: extreme upward perspective, we mainly see {bottom} and {near}, "
                f"{far} is tiny and far away, and the ceiling or sky is the whole background.")
    return (f"A direct worm's-eye view from straight below, the camera on the ground looking up at 90 degrees: "
            f"only {bottom} are visible, {far} is hidden behind them, and the ceiling or sky is the entire "
            f"background.")


# --- roll ------------------------------------------------------------------------------

def _roll(roll: int, subject: str) -> str:
    noun = SUBJECTS[subject]["noun"]
    a = abs(roll)
    direction = "clockwise" if roll > 0 else "counter-clockwise"
    headside = "right" if roll > 0 else "left"
    if a == 0:
        return f"The camera is level: the horizon is straight and the {noun} stands upright in the frame."
    if a <= 20:
        return (f"A slight Dutch angle: the camera is rolled so the whole picture is rotated {a} degrees "
                f"{direction}, the {noun}'s vertical axis leans a little toward the {headside} of the frame and "
                f"the horizon slopes.")
    if a <= 60:
        return (f"A strong Dutch angle: the whole picture is rotated {a} degrees {direction}, the {noun} leans "
                f"diagonally with its top toward the {headside} side of the frame and the horizon is steeply tilted.")
    if a <= 110:
        return (f"The picture is rotated {a} degrees {direction}, so the {noun} lies almost sideways across the "
                f"frame with its top pointing toward the {headside} edge of the image.")
    if a <= 170:
        return (f"The picture is rotated {a} degrees {direction}, nearly upside down: the {noun}'s top points "
                f"down and toward the {headside}, and the ground is near the top of the frame.")
    return (f"The whole picture is upside down, rotated 180 degrees: the {noun}'s top is at the bottom of the "
            f"frame and the ground is at the top.")


# --- framing ---------------------------------------------------------------------------

def _framing(framing: str, subject: str) -> str:
    noun = SUBJECTS[subject]["noun"]
    if framing == "full":
        if subject == "person":
            return ("Show the whole character from head to toe, full body, centered and filling most of the frame "
                    "with a small margin all around.")
        return "Show the whole object, centered and filling most of the frame with a small margin all around."
    if framing == "closeup":
        if subject == "person":
            return "A close-up: the head and shoulders fill the frame, centered."
        return "A close-up: the object fills the frame, centered, with its nearest surfaces large and detailed."
    return (f"Keep the same framing, crop and distance from the camera as in the first image, with the {noun} "
            f"centered and at the same size in the frame.")


def build_prompt(yaw: float, pitch: float, roll: float, subject: str = "person", framing: str = "same",
                 extra: str = "", roll_in_prompt: bool = True) -> tuple[str, tuple[int, int, int]]:
    """Return (caption, normalized (yaw, pitch, roll)).

    With roll_in_prompt=False the roll is left out of the caption (the client rotates the
    picture itself) but is still returned normalized for display.
    """
    if subject not in SUBJECTS:
        raise ValueError(f"unknown subject: {subject}")
    if framing not in FRAMINGS:
        raise ValueError(f"unknown framing: {framing}")
    y, p, r = normalize_angles(yaw, pitch, roll)
    info = SUBJECTS[subject]
    parts = [
        f"Show the same {info['noun']} from the first image, exactly one {info['noun']} and nothing else, seen "
        f"from a different camera viewpoint, keeping the {info['keep']} exactly the same as in the first image.",
        _yaw_person(y) if subject == "person" else _yaw_object(y),
        _pitch(p, subject),
        _roll(r if roll_in_prompt else 0, subject),
        _framing(framing, subject),
        "Keep the background and lighting consistent with the first image; add no new objects, text or people.",
    ]
    extra = (extra or "").strip()
    if extra:
        parts.append(extra)
    return " ".join(parts), (y, p, r)
