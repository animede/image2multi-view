# Copyright 2026 YUZUKI
# SPDX-License-Identifier: MIT
"""image2multi-view: orbit one uploaded picture with FLUX 3 Image Klein (flux3-klein-studio).

A thin server in front of flux3-klein-studio's job API. The browser sends angles; this server
builds the caption (viewprompt.py), downsizes the reference, submits the job with the prompt
upsampler bypassed, polls it, stores the result and serves it. Drafts are small and fast while
the user drags; the final render at full size is requested once the angles stop changing.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from viewprompt import FRAMINGS, SUBJECTS, build_prompt

KLEIN_URL = os.getenv("KLEIN_URL", "http://127.0.0.1:7864").rstrip("/")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

DRAFT_SIDES = (256, 320, 384, 448, 512, 640)
DRAFT_STEPS = (4, 6, 8, 12, 16)
DEFAULT_STEPS = 16
_steps_supported: bool | None = None   # whether the studio's /api/generate accepts `steps` (checked lazily)
FINAL_LONG_SIDES = (768, 1024, 1536, 2048)
FINAL_REF_SIDE = 1024          # the studio caps references at 1 MP anyway
POLL_SECONDS = 0.1
BUSY_RETRY_SECONDS = 0.15
BUSY_WAIT_LIMIT = 120.0        # give up submitting after this long of 409s

app = FastAPI(title="image2multi-view", version="0.1.0")

jobs: dict[str, dict] = {}
job_tasks: dict[str, asyncio.Task] = {}
generate_lock = asyncio.Lock()   # the studio runs one generation at a time; keep our queue ordered


# ---------------------------------------------------------------------------- helpers

def _snap32(value: float, lo: int = 256, hi: int = 2688) -> int:
    return max(lo, min(hi, int(round(value / 32)) * 32))


def fit_size(src_w: int, src_h: int, side: int) -> tuple[int, int]:
    """Width/height with the source aspect ratio and about side*side pixels, multiples of 32."""
    ar = src_w / src_h
    w = math.sqrt(side * side * ar)
    return _snap32(w), _snap32(w / ar)


def final_size(src_w: int, src_h: int, choice: str) -> tuple[int, int]:
    """'source' keeps the source size capped at 1 MP; otherwise choice is the long side."""
    ar = src_w / src_h
    if choice == "source":
        w, h = src_w, src_h
        if w * h > 1024 * 1024:
            scale = math.sqrt(1024 * 1024 / (w * h))
            w, h = w * scale, h * scale
    else:
        long_side = int(choice)
        if long_side not in FINAL_LONG_SIDES:
            raise HTTPException(422, f"final_sizeは{FINAL_LONG_SIDES}またはsourceです")
        w, h = (long_side, long_side / ar) if ar >= 1 else (long_side * ar, long_side)
    w, h = _snap32(w), _snap32(h)
    while w * h > 2048 * 2048:  # studio limit
        w, h = _snap32(w * 0.9), _snap32(h * 0.9)
    return w, h


def session_dir(sid: str) -> Path:
    if not sid or not all(c in "0123456789abcdef" for c in sid) or len(sid) != 32:
        raise HTTPException(422, "sessionが不正です")
    d = DATA / sid
    if not (d / "meta.json").exists():
        raise HTTPException(404, "sessionがありません。画像をアップロードし直してください")
    return d


def reference_bytes(src: Path, max_side: int) -> bytes:
    """Downscaled reference: smaller references mean fewer tokens in the transformer, so faster."""
    im = Image.open(src).convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", compress_level=1)
    return buf.getvalue()


def cached_reference(sdir: Path, max_side: int) -> bytes:
    path = sdir / f"ref_{max_side}.png"
    if not path.exists():
        path.write_bytes(reference_bytes(sdir / "source.png", max_side))
    return path.read_bytes()


# ---------------------------------------------------------------------------- upload

@app.post("/api/upload")
async def upload(image: UploadFile = File(...)):
    raw = await image.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "画像は20MB以内にしてください")
    try:
        im = Image.open(io.BytesIO(raw))
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[3])
            im = bg
        else:
            im = im.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"画像を読み込めません: {exc}") from exc
    if im.width * im.height > 16_000_000:
        im.thumbnail((4000, 4000), Image.LANCZOS)
    sid = uuid.uuid4().hex
    sdir = DATA / sid
    (sdir / "views").mkdir(parents=True)
    im.save(sdir / "source.png", "PNG", compress_level=1)
    meta = {"id": sid, "width": im.width, "height": im.height, "created_at": time.time(),
            "filename": image.filename}
    (sdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    return session_info(meta)


def session_info(meta: dict) -> dict:
    w, h, sid = meta["width"], meta["height"], meta["id"]
    return {**meta, "source_url": f"/data/{sid}/source.png",
            "draft_sizes": {str(s): fit_size(w, h, s) for s in DRAFT_SIDES},
            "final_sizes": {**{str(s): final_size(w, h, str(s)) for s in FINAL_LONG_SIDES},
                            "source": final_size(w, h, "source")}}


@app.get("/api/upload-info/{sid}")
async def upload_info(sid: str):
    sdir = session_dir(sid)
    return session_info(json.loads((sdir / "meta.json").read_text()))


# ---------------------------------------------------------------------------- prompt preview

@app.get("/api/prompt")
async def prompt_preview(yaw: float = 0, pitch: float = 0, roll: float = 0, subject: str = "person",
                         framing: str = "same", extra: str = "", roll_in_prompt: bool = True):
    try:
        text, angles = build_prompt(yaw, pitch, roll, subject, framing, extra, roll_in_prompt)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"prompt": text, "yaw": angles[0], "pitch": angles[1], "roll": angles[2]}


# ---------------------------------------------------------------------------- render jobs

class RenderRequest(BaseModel):
    session: str
    yaw: float = 0
    pitch: float = 0
    roll: float = 0
    mode: str = Field("draft", pattern="^(draft|final)$")
    seed: int = Field(42, ge=0, le=4294967295)
    subject: str = "person"
    framing: str = "same"
    extra: str = ""
    roll_in_prompt: bool = True
    draft_side: int = 512
    draft_steps: int = 8          # fewer steps for drafts; needs a studio build that accepts `steps`
    final_size: str = "source"
    upsample: bool = False        # final only: let the studio's prompt upsampler rewrite the caption
    upsampler: str | None = None
    shift: float | None = Field(None, ge=1, le=50)


def public_job(job: dict) -> dict:
    keys = ("id", "status", "mode", "cached", "step", "steps", "message", "prompt", "yaw", "pitch", "roll",
            "width", "height", "seed", "image_url", "seconds", "wall_seconds", "error", "warning", "key",
            "upsample_source", "created_at", "gen_steps")
    return {k: job.get(k) for k in keys}


@app.post("/api/render")
async def render(req: RenderRequest):
    sdir = session_dir(req.session)
    meta = json.loads((sdir / "meta.json").read_text())
    if req.subject not in SUBJECTS or req.framing not in FRAMINGS:
        raise HTTPException(422, "subject / framingが不正です")
    if req.mode == "draft":
        if req.draft_side not in DRAFT_SIDES:
            raise HTTPException(422, f"draft_sideは{DRAFT_SIDES}のいずれかです")
        if req.draft_steps not in DRAFT_STEPS:
            raise HTTPException(422, f"draft_stepsは{DRAFT_STEPS}のいずれかです")
        width, height = fit_size(meta["width"], meta["height"], req.draft_side)
        ref_side = req.draft_side
        upsample = False
        gen_steps = req.draft_steps if await steps_supported() else DEFAULT_STEPS
    else:
        width, height = final_size(meta["width"], meta["height"], req.final_size)
        ref_side = FINAL_REF_SIDE
        upsample = req.upsample
        gen_steps = DEFAULT_STEPS
    prompt, (yaw, pitch, roll) = build_prompt(req.yaw, req.pitch, req.roll, req.subject, req.framing,
                                              req.extra, req.roll_in_prompt)
    key_src = json.dumps({"s": req.session, "p": prompt, "w": width, "h": height, "seed": req.seed,
                          "ref": ref_side, "up": upsample, "upsampler": req.upsampler if upsample else None,
                          "shift": req.shift, "steps": gen_steps}, sort_keys=True)
    key = hashlib.sha256(key_src.encode()).hexdigest()[:24]
    image_path = sdir / "views" / f"{key}.png"
    base = {"session": req.session, "mode": req.mode, "prompt": prompt, "yaw": yaw, "pitch": pitch, "roll": roll,
            "width": width, "height": height, "seed": req.seed, "key": key, "created_at": time.time(),
            "gen_steps": gen_steps,
            "image_url": f"/data/{req.session}/views/{key}.png"}
    if image_path.exists():
        info = {}
        info_path = image_path.with_suffix(".json")
        if info_path.exists():
            info = json.loads(info_path.read_text())
        job = {**base, "id": "cached-" + key, "status": "done", "cached": True, "step": gen_steps, "steps": gen_steps,
               "message": "キャッシュ", "seconds": info.get("seconds"), "wall_seconds": 0.0}
        return public_job(job)
    for existing in jobs.values():
        if existing["key"] == key and existing["status"] not in ("done", "error", "cancelled"):
            return public_job(existing)
    job_id = uuid.uuid4().hex[:12]
    job = {**base, "id": job_id, "status": "queued", "cached": False, "step": 0, "steps": gen_steps,
           "message": "待機中", "cancel": False, "klein_id": None, "ref_side": ref_side, "upsample": upsample,
           "upsampler": req.upsampler, "shift": req.shift}
    jobs[job_id] = job
    if len(jobs) > 300:
        for old in sorted(jobs.values(), key=lambda j: j["created_at"])[:100]:
            if old["status"] in ("done", "error", "cancelled"):
                jobs.pop(old["id"], None)
    job_tasks[job_id] = asyncio.create_task(run_job(job, sdir, image_path))
    return public_job(job)


async def run_job(job: dict, sdir: Path, image_path: Path):
    t0 = time.time()
    try:
        async with generate_lock:
            if job["cancel"]:
                job.update(status="cancelled", message="停止しました")
                return
            await _run_job_inner(job, sdir, image_path)
    except Exception as exc:  # noqa: BLE001
        job.update(status="error", error=str(exc), message=f"エラー: {exc}")
    finally:
        job["wall_seconds"] = round(time.time() - t0, 2)
        job_tasks.pop(job["id"], None)


async def _run_job_inner(job: dict, sdir: Path, image_path: Path):
    ref = await asyncio.to_thread(cached_reference, sdir, job["ref_side"])
    data = {"prompt": job["prompt"], "width": str(job["width"]), "height": str(job["height"]),
            "seed": str(job["seed"]), "upsample": "true" if job["upsample"] else "false"}
    if job["upsample"] and job.get("upsampler"):
        data["upsampler"] = job["upsampler"]
    if job.get("shift") is not None:
        data["shift"] = str(job["shift"])
    if job["gen_steps"] != DEFAULT_STEPS:
        data["steps"] = str(job["gen_steps"])
    async with httpx.AsyncClient(base_url=KLEIN_URL, timeout=httpx.Timeout(60.0, read=120.0)) as client:
        waited = 0.0
        while True:
            if job["cancel"]:
                job.update(status="cancelled", message="停止しました")
                return
            r = await client.post("/api/generate", data=data, files=[("image", ("ref.png", ref, "image/png"))])
            if r.status_code == 409 and waited < BUSY_WAIT_LIMIT:
                job.update(status="queued", message="生成サーバーが使用中です。待っています")
                await asyncio.sleep(BUSY_RETRY_SECONDS)
                waited += BUSY_RETRY_SECONDS
                continue
            if r.status_code != 202:
                try:
                    detail = r.json().get("detail")
                except Exception:  # noqa: BLE001
                    detail = r.text
                raise RuntimeError(f"生成サーバー {r.status_code}: {detail}")
            break
        info = r.json()
        job.update(klein_id=info["id"], steps=info.get("steps", 16), status=info.get("status", "queued"))
        while True:
            await asyncio.sleep(POLL_SECONDS)
            if job["cancel"] and job["status"] not in ("done",):
                await client.post(f"/api/jobs/{job['klein_id']}/cancel")
                job.update(status="cancelled", message="停止しました")
                return
            s = (await client.get(f"/api/jobs/{job['klein_id']}")).json()
            # Publish "done" only once the image is on disk, so a poller never sees done without a file.
            job.update(status="saving" if s["status"] == "done" else s["status"], step=s.get("step", 0),
                       steps=s.get("steps", 16), message=s.get("message", ""))
            if s["status"] == "done":
                img = await client.get(s["image_url"])
                img.raise_for_status()
                image_path.write_bytes(img.content)
                metrics = s.get("metrics") or {}
                job.update(seconds=metrics.get("total_seconds") or metrics.get("seconds"),
                           warning=s.get("warning"), upsample_source=s.get("upsample_source"))
                image_path.with_suffix(".json").write_text(json.dumps({
                    "prompt": job["prompt"], "yaw": job["yaw"], "pitch": job["pitch"], "roll": job["roll"],
                    "width": job["width"], "height": job["height"], "seed": job["seed"], "mode": job["mode"],
                    "steps": job["gen_steps"],
                    "seconds": job.get("seconds"), "upsample_source": job.get("upsample_source"),
                    "klein_id": job["klein_id"], "created_at": time.time()}, ensure_ascii=False))
                job.update(status="done", message="完了")
                return
            if s["status"] in ("error", "cancelled"):
                job.update(error=s.get("message"))
                return


@app.get("/api/render/{job_id}")
async def render_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "ジョブがありません")
    return public_job(job)


@app.post("/api/render/{job_id}/cancel")
async def render_cancel(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "ジョブがありません")
    if job["status"] in ("done", "error", "cancelled"):
        return public_job(job)
    job["cancel"] = True
    job["message"] = "停止を要求しました"
    return public_job(job)


@app.get("/api/sessions/{sid}/views")
async def session_views(sid: str):
    sdir = session_dir(sid)
    items = []
    for info_path in (sdir / "views").glob("*.json"):
        png = info_path.with_suffix(".png")
        if not png.exists():
            continue
        info = json.loads(info_path.read_text())
        items.append({**info, "key": png.stem, "image_url": f"/data/{sid}/views/{png.name}"})
    items.sort(key=lambda i: i.get("created_at", 0), reverse=True)
    return {"items": items}


async def steps_supported(force: bool = False) -> bool:
    """True when the running studio's /api/generate lists a `steps` field (added 2026-10-27)."""
    global _steps_supported
    if _steps_supported is None or force:
        try:
            async with httpx.AsyncClient(base_url=KLEIN_URL, timeout=5.0) as client:
                spec = (await client.get("/openapi.json")).json()
            body = spec["paths"]["/api/generate"]["post"]["requestBody"]["content"]
            schema = next(iter(body.values()))["schema"]
            if "$ref" in schema:
                schema = spec["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]]
            _steps_supported = "steps" in (schema.get("properties") or {})
        except Exception:  # noqa: BLE001
            return False   # leave unknown so a later call retries
    return _steps_supported


@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(base_url=KLEIN_URL, timeout=5.0) as client:
            s = (await client.get("/api/health")).json()
        klein = {"ok": True, "loaded": s.get("loaded"), "busy": s.get("busy"),
                 "steps_supported": await steps_supported(force=True),
                 "upsamplers": {k: v.get("available") for k, v in (s.get("upsamplers") or {}).items()}}
    except Exception as exc:  # noqa: BLE001
        klein = {"ok": False, "error": str(exc)}
    return {"status": "ok", "klein_url": KLEIN_URL, "klein": klein, "draft_sides": DRAFT_SIDES,
            "draft_steps": DRAFT_STEPS, "final_long_sides": FINAL_LONG_SIDES, "active_jobs": len(job_tasks)}


app.mount("/data", StaticFiles(directory=str(DATA)), name="data")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-store"})


@app.exception_handler(404)
async def not_found(_request, exc):
    return JSONResponse({"detail": getattr(exc, "detail", "not found")}, status_code=404)
