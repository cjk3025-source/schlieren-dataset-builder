import io
import json
import math
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

APP_TITLE = "BOS Schlieren Label Review Studio"
DEFAULT_PROJECTS_ROOT = str(Path.cwd() / "projects")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def safe_write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def mpm_to_mms(v: float) -> float:
    return v * 1000.0 / 60.0


def mms_to_mpm(v: float) -> float:
    return v * 60.0 / 1000.0


def get_project_paths(project_root: Path) -> Dict[str, Path]:
    return {
        "root": project_root,
        "raw": ensure_dir(project_root / "raw"),
        "processed": ensure_dir(project_root / "processed"),
        "roi": ensure_dir(project_root / "roi"),
        "labels": ensure_dir(project_root / "labels"),
        "exports": ensure_dir(project_root / "exports"),
        "state": project_root / "project_state.json",
        "registry": project_root / "registry.csv",
        "truth": project_root / "ground_truth.csv",
    }


DEFAULT_STATE: Dict[str, Any] = {
    "project_name": "",
    "settings": {
        "repeat_count": 1,
        "naming_mode": "auto",
        "specimen_prefix": "SP",
        "start_number": 1,
        "custom_specimen_names": [],
        "laser_unit": "W",
        "laser_levels": [
            {"label": "reference", "value": 900.0, "enabled": True},
            {"label": "over", "value": 950.0, "enabled": False},
            {"label": "under", "value": 850.0, "enabled": False},
        ],
        "speed_unit": "mpm",
        "speed_levels": [6.0],
        "gap_levels": [],
        "defocusing_levels": [],
        "extra_factors": [],
        "external_conditions": {
            "backlight_power_w": 445.0,
            "pattern_to_backlight_cm": 20.0,
            "pattern_to_weld_center_cm": 50.0,
            "specimen_to_lens_cm": 95.0,
            "aperture": "E",
            "fps": 2000,
            "shutter_speed": "1/5000",
            "resolution": "1280x1024",
            "weld_direction": "right_to_left",
            "material": "STS316L",
            "thickness_t": "0.7t",
            "test_stage": "feasibility",
            "test_date": "",
        },
    },
    "ui": {
        "selected_specimen": "",
        "gallery_page": 1,
        "preprocess_frame_index": 0,
        "label_frame_index": 0,
        "crop_to_roi": True,
        "preview_class": "fume",
        "enable_ignore": False,
        "label_mode": "interval",
        "label_interval": 20,
        "label_selected_frames": "",
        "label_background_source": "preprocessed",
    },
    "preprocess": {
        "brightness": 0,
        "contrast": 1.0,
        "gamma": 1.0,
        "clahe": 2.0,
        "fume_blur": 17,
        "fume_bg_blur": 41,
        "fume_threshold": 22,
        "fume_close": 7,
        "spatter_hp_blur": 17,
        "spatter_threshold": 38,
        "spatter_min_area": 2,
        "spatter_max_area": 150,
        "spatter_open": 3,
    },
    "roi": {},
    "truth": {},
}


@dataclass
class SpecimenAssets:
    specimen_dir: Path
    frames_dir: Path
    raw_video: Optional[Path]
    frame_paths: List[Path]


def merge_defaults(obj: Dict[str, Any], default: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(default))
    def rec(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                rec(dst[k], v)
            else:
                dst[k] = v
    rec(out, obj)
    return out


def load_state(project_root: Path) -> Dict[str, Any]:
    pp = get_project_paths(project_root)
    existing = safe_read_json(pp["state"], {})
    state = merge_defaults(existing, DEFAULT_STATE)
    state["project_name"] = project_root.name
    return state


def save_state(project_root: Path, state: Dict[str, Any]) -> None:
    pp = get_project_paths(project_root)
    safe_write_json(pp["state"], state)
    registry = build_registry_dataframe(state)
    registry.to_csv(pp["registry"], index=False, encoding="utf-8-sig")
    build_truth_dataframe(state).to_csv(pp["truth"], index=False, encoding="utf-8-sig")


def parse_simple_levels(text: str, cast=float) -> List[Any]:
    items = []
    for x in text.replace("\n", ",").split(","):
        x = x.strip()
        if not x:
            continue
        try:
            items.append(cast(x))
        except Exception:
            pass
    return items


def effective_laser_levels(settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    unit = settings["laser_unit"]
    levels = []
    for row in settings["laser_levels"]:
        if row.get("enabled"):
            levels.append({
                "label": row.get("label", "level"),
                "value": float(row.get("value", 0)),
                "unit": unit,
                "display": f"{row.get('label', 'level')}:{row.get('value', 0)}{unit}",
            })
    return levels or [{"label": "reference", "value": 0.0, "unit": unit, "display": f"reference:0{unit}"}]


def effective_speed_levels(settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    unit = settings["speed_unit"]
    vals = [float(v) for v in settings["speed_levels"] if str(v).strip() != ""]
    if not vals:
        vals = [6.0 if unit == "mpm" else 100.0]
    out = []
    for v in vals:
        if unit == "mpm":
            out.append({"value": v, "unit": unit, "display": f"{v:g} mpm", "mm_per_s": mpm_to_mms(v)})
        else:
            out.append({"value": v, "unit": unit, "display": f"{v:g} mm/s", "mm_per_s": v})
    return out


def build_factor_levels(settings: Dict[str, Any]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    factors: List[Tuple[str, List[Dict[str, Any]]]] = []
    lasers = effective_laser_levels(settings)
    if lasers:
        factors.append(("laser_power", lasers))
    speeds = effective_speed_levels(settings)
    if speeds:
        factors.append(("welding_speed", speeds))
    gap_levels = [g for g in settings.get("gap_levels", []) if str(g).strip()]
    if gap_levels:
        factors.append(("gap", [{"display": g, "value": g} for g in gap_levels]))
    df_levels = [g for g in settings.get("defocusing_levels", []) if str(g).strip()]
    if df_levels:
        factors.append(("defocusing", [{"display": g, "value": g} for g in df_levels]))
    for row in settings.get("extra_factors", []):
        name = str(row.get("name", "")).strip()
        levels = [x for x in row.get("levels", []) if str(x).strip()]
        if name and levels:
            factors.append((name, [{"display": x, "value": x} for x in levels]))
    return factors


def cartesian_product(dict_items: List[Tuple[str, List[Dict[str, Any]]]]) -> List[Dict[str, Dict[str, Any]]]:
    if not dict_items:
        return [{}]
    name, vals = dict_items[0]
    rest = cartesian_product(dict_items[1:])
    out = []
    for v in vals:
        for r in rest:
            x = dict(r)
            x[name] = v
            out.append(x)
    return out


def build_registry_dataframe(state: Dict[str, Any]) -> pd.DataFrame:
    settings = state["settings"]
    factors = build_factor_levels(settings)
    combos = cartesian_product(factors)
    repeats = int(settings.get("repeat_count", 1) or 1)
    names = settings.get("custom_specimen_names", []) or []
    rows = []
    idx = 0
    for repeat_idx in range(1, repeats + 1):
        for combo_idx, combo in enumerate(combos, start=1):
            idx += 1
            if settings.get("naming_mode") == "custom" and idx <= len(names) and names[idx - 1].strip():
                specimen_id = names[idx - 1].strip()
            else:
                prefix = settings.get("specimen_prefix", "SP")
                start = int(settings.get("start_number", 1))
                specimen_id = f"{prefix}{start + idx - 1:03d}"
            row = {
                "specimen_id": specimen_id,
                "repeat_index": repeat_idx,
                "combo_index": combo_idx,
            }
            for key, val in combo.items():
                row[key] = val.get("display", val.get("value"))
                if key == "laser_power":
                    row["laser_power_label"] = val.get("label")
                    row["laser_power_value"] = val.get("value")
                    row["laser_power_unit"] = val.get("unit")
                if key == "welding_speed":
                    row["welding_speed_value"] = val.get("value")
                    row["welding_speed_unit"] = val.get("unit")
                    row["welding_speed_mm_per_s"] = val.get("mm_per_s")
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["specimen_id", "repeat_index", "combo_index"])
    return df


def auto_final_label(visual_result: str, tensile_result: str, fallback: str = "미입력") -> str:
    valid_simple = {"미입력", "정상", "불량"}
    visual = visual_result if visual_result in valid_simple else "미입력"
    tensile = tensile_result if tensile_result in valid_simple else "미입력"
    if visual == "불량" or tensile == "불량":
        return "불량"
    if visual == "정상" and tensile == "정상":
        return "정상"
    if visual == "미입력" and tensile == "미입력":
        return fallback if fallback in {"미입력", "review_needed"} else "미입력"
    return fallback if fallback == "review_needed" else "미입력"


def build_truth_dataframe(state: Dict[str, Any]) -> pd.DataFrame:
    registry = build_registry_dataframe(state)
    truth = state.get("truth", {})
    rows = []
    for specimen_id in registry["specimen_id"].tolist() if not registry.empty else []:
        t = truth.get(specimen_id, {})
        visual_result = t.get("visual_result", "미입력")
        tensile_result = t.get("tensile_result", "미입력")
        final_label = auto_final_label(visual_result, tensile_result, t.get("final_label", "미입력"))
        short = f"{specimen_id} 시편은 {final_label}입니다."
        details = t.get("details", "")
        rows.append({
            "specimen_id": specimen_id,
            "final_label": final_label,
            "visual_result": visual_result,
            "tensile_result": tensile_result,
            "defect_type": t.get("defect_type", ""),
            "notes": t.get("notes", ""),
            "summary_short": short,
            "summary_detail": details,
        })
    return pd.DataFrame(rows)


def get_specimen_display_name(registry_df: pd.DataFrame, specimen_id: str) -> str:
    if not specimen_id:
        return "-"
    if registry_df is None or registry_df.empty or "specimen_id" not in registry_df.columns:
        return specimen_id
    matched = registry_df.loc[registry_df["specimen_id"] == specimen_id]
    if matched.empty:
        return specimen_id
    row = matched.iloc[0]
    laser_name = str(row.get("laser_power_label", "")).strip()
    laser_display = str(row.get("laser_power", "")).strip()
    if laser_name and laser_display and laser_name != laser_display:
        return f"{specimen_id} · {laser_name} ({laser_display})"
    if laser_name:
        return f"{specimen_id} · {laser_name}"
    if laser_display:
        return f"{specimen_id} · {laser_display}"
    return specimen_id


def specimen_assets(project_root: Path, specimen_id: str) -> SpecimenAssets:
    pp = get_project_paths(project_root)
    specimen_dir = ensure_dir(pp["raw"] / specimen_id)
    frames_dir = get_current_frames_dir(specimen_dir)
    raw_video = None
    for p in specimen_dir.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            raw_video = p
            break
    frame_paths = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in IMG_EXTS]) if frames_dir.exists() else []
    return SpecimenAssets(specimen_dir, frames_dir, raw_video, frame_paths)


def save_uploaded_file(uploaded, target: Path) -> None:
    ensure_dir(target.parent)
    with open(target, "wb") as f:
        f.write(uploaded.getbuffer())


def get_current_frames_dir(specimen_dir: Path) -> Path:
    meta_path = specimen_dir / "current_upload.json"
    meta = safe_read_json(meta_path, {})
    current = meta.get("current_frames_dir") if isinstance(meta, dict) else None
    if current and Path(current).exists():
        return Path(current)
    uploads_dir = specimen_dir / "uploads"
    if uploads_dir.exists():
        batches = sorted([p for p in uploads_dir.iterdir() if p.is_dir()])
        if batches:
            latest = batches[-1]
            safe_write_json(meta_path, {"current_frames_dir": str(latest)})
            return latest
    legacy_frames = specimen_dir / "frames"
    ensure_dir(legacy_frames)
    return legacy_frames


def list_frame_batches(specimen_dir: Path) -> List[Path]:
    uploads_dir = specimen_dir / "uploads"
    if not uploads_dir.exists():
        return []
    return sorted([p for p in uploads_dir.iterdir() if p.is_dir()], reverse=True)


def set_current_frames_dir(specimen_dir: Path, batch_dir: Path) -> None:
    safe_write_json(specimen_dir / "current_upload.json", {"current_frames_dir": str(batch_dir)})


def clear_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)



def reset_uploaded_frames(specimen_dir: Path) -> None:
    clear_path(specimen_dir / "uploads")
    clear_path(specimen_dir / "frames")
    clear_path(specimen_dir / "current_upload.json")
    for p in list(specimen_dir.iterdir()) if specimen_dir.exists() else []:
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            p.unlink(missing_ok=True)



def replace_uploaded_frames_from_zip(uploaded_zip: Any, specimen_dir: Path) -> int:
    temp_root = ensure_dir(specimen_dir / "_tmp_import")
    temp_batch = temp_root / pd.Timestamp.now().strftime("upload_%Y%m%d_%H%M%S_%f")
    ensure_dir(temp_batch)
    count = 0
    try:
        with zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue())) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                if not name:
                    continue
                if Path(name).suffix.lower() in IMG_EXTS:
                    with zf.open(info) as src, open(temp_batch / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                        count += 1
        if count == 0:
            shutil.rmtree(temp_batch, ignore_errors=True)
            return 0
        reset_uploaded_frames(specimen_dir)
        uploads_dir = ensure_dir(specimen_dir / "uploads")
        final_batch = uploads_dir / temp_batch.name
        if final_batch.exists():
            shutil.rmtree(final_batch, ignore_errors=True)
        shutil.move(str(temp_batch), str(final_batch))
        set_current_frames_dir(specimen_dir, final_batch)
        return count
    finally:
        if temp_batch.exists():
            shutil.rmtree(temp_batch, ignore_errors=True)
        if temp_root.exists() and not any(temp_root.iterdir()):
            shutil.rmtree(temp_root, ignore_errors=True)


@st.cache_data(show_spinner=False)
def load_frame_image(path_str: str) -> np.ndarray:
    path = Path(path_str)
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.array(img)


def try_load_frame_image(path_str: str) -> Optional[np.ndarray]:
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return None
    try:
        return load_frame_image(path_str)
    except FileNotFoundError:
        return None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def sample_video_frames(video_path_str: str, max_frames: int = 200) -> List[np.ndarray]:
    video_path = Path(video_path_str)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, math.ceil(total / max_frames)) if total else 1
    frames: List[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def auto_roi_polygon(img: np.ndarray) -> List[List[int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (31, 31), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = gray.shape
        return [[int(w * 0.2), int(h * 0.1)], [int(w * 0.8), int(h * 0.1)], [int(w * 0.8), int(h * 0.75)], [int(w * 0.2), int(h * 0.75)]]
    cnt = max(cnts, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    pts = approx[:, 0, :].tolist()
    if len(pts) < 3:
        x, y, w, h = cv2.boundingRect(cnt)
        pts = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
    return [[int(x), int(y)] for x, y in pts]


def polygon_to_mask(shape: Tuple[int, int], points: List[List[int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) >= 3:
        arr = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [arr], 255)
    return mask


def apply_preprocess(img: np.ndarray, params: Dict[str, Any], roi_points: Optional[List[List[int]]] = None) -> Dict[str, np.ndarray]:
    work = img.copy().astype(np.float32)
    work = work * float(params["contrast"]) + float(params["brightness"])
    work = np.clip(work, 0, 255).astype(np.uint8)

    gamma = max(0.1, float(params["gamma"]))
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype(np.uint8)
    work = cv2.LUT(work, lut)

    lab = cv2.cvtColor(work, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=max(0.1, float(params["clahe"])), tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    work = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)

    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    roi_mask = np.ones_like(gray, dtype=np.uint8) * 255
    if roi_points and len(roi_points) >= 3:
        roi_mask = polygon_to_mask(gray.shape, roi_points)

    def odd(v: int) -> int:
        v = max(1, int(v))
        return v if v % 2 == 1 else v + 1

    fume_blur = odd(params["fume_blur"])
    bg_blur = odd(params["fume_bg_blur"])
    fume = cv2.GaussianBlur(gray, (fume_blur, fume_blur), 0)
    bg = cv2.GaussianBlur(gray, (bg_blur, bg_blur), 0)
    fume_diff = cv2.absdiff(bg, fume)
    _, fume_mask = cv2.threshold(fume_diff, int(params["fume_threshold"]), 255, cv2.THRESH_BINARY)
    close_k = max(1, int(params["fume_close"]))
    fume_mask = cv2.morphologyEx(fume_mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    fume_mask = cv2.bitwise_and(fume_mask, roi_mask)

    hp_blur = odd(params["spatter_hp_blur"])
    low = cv2.GaussianBlur(gray, (hp_blur, hp_blur), 0)
    hp = cv2.subtract(gray, low)
    _, sp_mask = cv2.threshold(hp, int(params["spatter_threshold"]), 255, cv2.THRESH_BINARY)
    open_k = max(1, int(params["spatter_open"]))
    sp_mask = cv2.morphologyEx(sp_mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(sp_mask, 8)
    filtered = np.zeros_like(sp_mask)
    min_area = int(params["spatter_min_area"])
    max_area = int(params["spatter_max_area"])
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            filtered[labels == i] = 255
    sp_mask = cv2.bitwise_and(filtered, roi_mask)

    # Separate small bright particles from the broad fume region so that
    # fume_only is less likely to contain spatter highlights.
    fume_mask = cv2.bitwise_and(fume_mask, cv2.bitwise_not(sp_mask))

    overlay = work.copy()
    overlay[fume_mask > 0] = [0, 255, 255]
    overlay[sp_mask > 0] = [255, 120, 0]

    fume_only = np.zeros_like(work)
    fume_only[fume_mask > 0] = work[fume_mask > 0]
    spatter_only = np.zeros_like(work)
    spatter_only[sp_mask > 0] = work[sp_mask > 0]

    return {
        "base": work,
        "overlay": overlay,
        "fume_mask": fume_mask,
        "spatter_mask": sp_mask,
        "fume_only": fume_only,
        "spatter_only": spatter_only,
        "roi_mask": roi_mask,
    }


def crop_to_roi(img: np.ndarray, roi_points: Optional[List[List[int]]]) -> np.ndarray:
    if not roi_points or len(roi_points) < 3:
        return img
    pts = np.array(roi_points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    return img[y:y+h, x:x+w]


def write_video(frames: List[np.ndarray], out_path: Path, fps: int = 20) -> None:
    if not frames:
        return
    ensure_dir(out_path.parent)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def load_specimen_frames(project_root: Path, specimen_id: str) -> List[Path]:
    assets = specimen_assets(project_root, specimen_id)
    if assets.frame_paths:
        return assets.frame_paths
    return []


def get_preview_frame_count(project_root: Path, specimen_id: str, max_video_frames: int = 200) -> int:
    assets = specimen_assets(project_root, specimen_id)
    if assets.frame_paths:
        return len(assets.frame_paths)
    if assets.raw_video:
        return len(sample_video_frames(str(assets.raw_video), max_video_frames))
    return 0


def get_preview_frame(project_root: Path, specimen_id: str, frame_idx: int, max_video_frames: int = 200) -> Optional[np.ndarray]:
    assets = specimen_assets(project_root, specimen_id)
    if assets.frame_paths:
        if 0 <= frame_idx < len(assets.frame_paths):
            return try_load_frame_image(str(assets.frame_paths[frame_idx]))
        return None
    if assets.raw_video:
        sampled = sample_video_frames(str(assets.raw_video), max_video_frames)
        if 0 <= frame_idx < len(sampled):
            return sampled[frame_idx]
    return None



def get_label_background_image(project_root: Path, specimen_id: str, frame_idx: int, params: Dict[str, Any], source: str, crop_to_roi_flag: bool = True) -> Optional[np.ndarray]:
    base_frame = get_preview_frame(project_root, specimen_id, frame_idx)
    if base_frame is None:
        return None
    roi_points = load_roi(project_root, specimen_id)
    results = apply_preprocess(base_frame, params, roi_points)
    source_map = {
        "raw": base_frame,
        "preprocessed": results["base"],
        "overlay": results["overlay"],
        "fume_only": results["fume_only"],
        "spatter_only": results["spatter_only"],
    }
    img = source_map.get(source, results["base"])
    if crop_to_roi_flag:
        return crop_to_roi(img, roi_points)
    return img


def mask_to_polygons(mask: np.ndarray, min_area: int = 20, epsilon_ratio: float = 0.01, max_polygons: int = 30) -> List[List[List[int]]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    polys: List[List[List[int]]] = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < float(min_area):
            continue
        epsilon = max(1.0, epsilon_ratio * cv2.arcLength(cnt, True))
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = approx[:, 0, :].tolist()
        if len(pts) >= 3:
            polys.append([[int(x), int(y)] for x, y in pts])
        if len(polys) >= int(max_polygons):
            break
    return polys


def save_label_polygons(project_root: Path, specimen_id: str, frame_idx: int, cls: str, polygons: List[List[List[int]]], replace_class: bool = False, source: str = "manual") -> int:
    label_dir = ensure_dir(get_project_paths(project_root)["labels"] / specimen_id)
    path = label_dir / f"frame_{frame_idx:05d}.json"
    data = safe_read_json(path, {"polygons": []})
    existing = data.get("polygons", []) if isinstance(data, dict) else []
    if replace_class:
        existing = [x for x in existing if x.get("class") != cls]
    for pts in polygons:
        existing.append({"class": cls, "points": pts, "source": source})
    data = {"polygons": existing}
    safe_write_json(path, data)
    return len(polygons)

def save_roi(project_root: Path, specimen_id: str, points: List[List[int]]) -> None:
    pp = get_project_paths(project_root)
    safe_write_json(pp["roi"] / f"{specimen_id}.json", {"points": points})


def load_roi(project_root: Path, specimen_id: str) -> Optional[List[List[int]]]:
    pp = get_project_paths(project_root)
    d = safe_read_json(pp["roi"] / f"{specimen_id}.json", {})
    return d.get("points") if isinstance(d, dict) else None


def save_label_polygon(project_root: Path, specimen_id: str, frame_idx: int, cls: str, points: List[List[int]]) -> None:
    label_dir = ensure_dir(get_project_paths(project_root)["labels"] / specimen_id)
    data = safe_read_json(label_dir / f"frame_{frame_idx:05d}.json", {"polygons": []})
    data.setdefault("polygons", []).append({"class": cls, "points": points})
    safe_write_json(label_dir / f"frame_{frame_idx:05d}.json", data)


def label_manifest(project_root: Path, specimen_id: str) -> pd.DataFrame:
    label_dir = ensure_dir(get_project_paths(project_root)["labels"] / specimen_id)
    rows = []
    for p in sorted(label_dir.glob("frame_*.json")):
        data = safe_read_json(p, {"polygons": []})
        rows.append({
            "frame_file": p.name,
            "polygon_count": len(data.get("polygons", [])),
            "classes": ", ".join(sorted(set(x.get("class", "") for x in data.get("polygons", []))))
        })
    return pd.DataFrame(rows)


CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 255, 255),   # fume
    2: (255, 140, 0),   # spatter
    3: (255, 0, 255),   # ignore
}
CLASS_NAMES: Dict[int, str] = {0: "background", 1: "fume", 2: "spatter", 3: "ignore"}
ALLOWED_MASK_VALUES = {0, 1, 2, 3}


def specimen_mask_dir(project_root: Path, specimen_id: str) -> Path:
    return ensure_dir(get_project_paths(project_root)["labels"] / specimen_id / "masks_png")


def specimen_prediction_dir(project_root: Path, specimen_id: str) -> Path:
    return ensure_dir(get_project_paths(project_root)["processed"] / specimen_id / "predictions")


def save_uploaded_png_sets(uploaded_files: List[Any], target_dir: Path) -> int:
    ensure_dir(target_dir)
    count = 0
    for up in uploaded_files:
        suffix = Path(up.name).suffix.lower()
        if suffix == ".png":
            save_uploaded_file(up, target_dir / Path(up.name).name)
            count += 1
        elif suffix == ".zip":
            with zipfile.ZipFile(io.BytesIO(up.getvalue())) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    name = Path(info.filename).name
                    if Path(name).suffix.lower() == ".png":
                        with zf.open(info) as src, open(target_dir / name, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                            count += 1
    return count


@st.cache_data(show_spinner=False)
def load_mask_image(path_str: str) -> np.ndarray:
    return np.array(Image.open(path_str).convert("L"), dtype=np.uint8)


@st.cache_data(show_spinner=False)
def mask_stats(path_str: str) -> Dict[str, Any]:
    mask = load_mask_image(path_str)
    uniques = sorted(int(v) for v in np.unique(mask).tolist())
    counts = {int(v): int((mask == v).sum()) for v in uniques}
    total = int(mask.size) if mask.size else 0
    invalid = [v for v in uniques if v not in ALLOWED_MASK_VALUES]
    return {
        "shape": [int(mask.shape[0]), int(mask.shape[1])],
        "uniques": uniques,
        "counts": counts,
        "invalid": invalid,
        "has_fume": 1 in counts,
        "has_spatter": 2 in counts,
        "has_ignore": 3 in counts,
        "empty_mask": total == 0 or counts.get(1, 0) + counts.get(2, 0) + counts.get(3, 0) == 0,
        "ignore_only": counts.get(3, 0) > 0 and counts.get(1, 0) == 0 and counts.get(2, 0) == 0,
    }


def render_mask_rgb(mask: np.ndarray, show_ignore: bool = True) -> np.ndarray:
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for value, color in CLASS_COLORS.items():
        if value == 3 and not show_ignore:
            continue
        rgb[mask == value] = color
    return rgb


def blend_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.35, show_ignore: bool = True) -> np.ndarray:
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    color_mask = render_mask_rgb(mask, show_ignore=show_ignore)
    valid = mask > 0
    if not show_ignore:
        valid = np.logical_and(valid, mask != 3)
    out = image.copy()
    out[valid] = ((1.0 - alpha) * out[valid] + alpha * color_mask[valid]).astype(np.uint8)
    return out


def build_dataset_rows(project_root: Path, specimen_id: str, state: Dict[str, Any]) -> pd.DataFrame:
    assets = specimen_assets(project_root, specimen_id)
    image_map = {p.stem: p for p in assets.frame_paths}
    mask_map = {p.stem: p for p in specimen_mask_dir(project_root, specimen_id).glob("*.png")}
    pred_map = {p.stem: p for p in specimen_prediction_dir(project_root, specimen_id).glob("*.png")}
    all_keys = sorted(set(image_map.keys()) | set(mask_map.keys()) | set(pred_map.keys()))
    truth = state.get("truth", {}).get(specimen_id, {})
    rows: List[Dict[str, Any]] = []
    for key in all_keys:
        image_path = image_map.get(key)
        mask_path = mask_map.get(key)
        pred_path = pred_map.get(key)
        if image_path is not None and not image_path.exists():
            image_path = None
        if mask_path is not None and not mask_path.exists():
            mask_path = None
        if pred_path is not None and not pred_path.exists():
            pred_path = None
        row: Dict[str, Any] = {
            "item_id": key,
            "image_name": image_path.name if image_path else "",
            "mask_name": mask_path.name if mask_path else "",
            "prediction_name": pred_path.name if pred_path else "",
            "has_image": image_path is not None,
            "has_mask": mask_path is not None,
            "has_prediction": pred_path is not None,
            "status": "ready" if image_path and mask_path else ("missing_image" if mask_path and not image_path else "missing_mask" if image_path and not mask_path else "orphan_prediction"),
            "quality_label": auto_final_label(truth.get("visual_result", "미입력"), truth.get("tensile_result", "미입력"), truth.get("final_label", "미입력")),
            "reviewed": truth.get("reviewed", False),
        }
        if image_path:
            img = try_load_frame_image(str(image_path))
            if img is not None:
                row["image_size"] = f"{img.shape[1]}x{img.shape[0]}"
            else:
                row["image_size"] = ""
                row["has_image"] = False
                row["image_name"] = ""
                row["status"] = "missing_image" if mask_path else "orphan_prediction"
                image_path = None
        else:
            row["image_size"] = ""
        if mask_path:
            stats = mask_stats(str(mask_path))
            row["mask_size"] = f"{stats['shape'][1]}x{stats['shape'][0]}"
            row["mask_values"] = ",".join(str(v) for v in stats["uniques"])
            row["invalid_values"] = ",".join(str(v) for v in stats["invalid"])
            row["has_fume"] = bool(stats["has_fume"])
            row["has_spatter"] = bool(stats["has_spatter"])
            row["has_ignore"] = bool(stats["has_ignore"])
            row["empty_mask"] = bool(stats["empty_mask"])
            row["ignore_only"] = bool(stats["ignore_only"])
            row["size_match"] = row["image_size"] == row["mask_size"] if row["image_size"] else False
        else:
            row.update({
                "mask_size": "",
                "mask_values": "",
                "invalid_values": "",
                "has_fume": False,
                "has_spatter": False,
                "has_ignore": False,
                "empty_mask": False,
                "ignore_only": False,
                "size_match": False,
            })
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[
            "item_id", "image_name", "mask_name", "prediction_name", "has_image", "has_mask", "has_prediction",
            "status", "image_size", "mask_size", "mask_values", "invalid_values", "has_fume", "has_spatter",
            "has_ignore", "empty_mask", "ignore_only", "size_match", "quality_label", "reviewed"
        ])
    return pd.DataFrame(rows)


def save_truth_from_dataframe(project_root: Path, df: pd.DataFrame, state: Dict[str, Any]) -> None:
    truth = {}
    for _, row in df.iterrows():
        sid = str(row["specimen_id"])
        visual_result = row.get("visual_result", "")
        tensile_result = row.get("tensile_result", "")
        truth[sid] = {
            "final_label": auto_final_label(visual_result, tensile_result, row.get("final_label", "미입력")),
            "visual_result": visual_result,
            "tensile_result": tensile_result,
            "defect_type": row.get("defect_type", ""),
            "notes": row.get("notes", ""),
            "details": row.get("summary_detail", ""),
        }
    state["truth"] = truth
    save_state(project_root, state)



def export_handoff(project_root: Path, state: Dict[str, Any]) -> Path:
    pp = get_project_paths(project_root)
    registry = build_registry_dataframe(state)
    truth_df = build_truth_dataframe(state)
    export_dir = ensure_dir(pp["exports"] / f"handoff_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
    registry.to_csv(export_dir / "registry.csv", index=False, encoding="utf-8-sig")
    truth_df.to_csv(export_dir / "ground_truth.csv", index=False, encoding="utf-8-sig")
    safe_write_json(export_dir / "external_conditions.json", state["settings"]["external_conditions"])
    dataset_summary_rows: List[pd.DataFrame] = []
    for specimen_id in registry["specimen_id"].tolist():
        out = ensure_dir(export_dir / specimen_id)
        raw_assets = specimen_assets(project_root, specimen_id)
        if raw_assets.frame_paths:
            fd = ensure_dir(out / "frames")
            for p in raw_assets.frame_paths:
                shutil.copy2(p, fd / p.name)
        mask_dir = specimen_mask_dir(project_root, specimen_id)
        if mask_dir.exists():
            shutil.copytree(mask_dir, out / "masks_png", dirs_exist_ok=True)
        pred_dir = specimen_prediction_dir(project_root, specimen_id)
        if pred_dir.exists():
            shutil.copytree(pred_dir, out / "predictions", dirs_exist_ok=True)
        dataset_df = build_dataset_rows(project_root, specimen_id, state)
        if not dataset_df.empty:
            dataset_df.insert(0, "specimen_id", specimen_id)
            dataset_df.to_csv(out / "dataset_manifest.csv", index=False, encoding="utf-8-sig")
            dataset_summary_rows.append(dataset_df)
    if dataset_summary_rows:
        pd.concat(dataset_summary_rows, ignore_index=True).to_csv(export_dir / "dataset_manifest_all.csv", index=False, encoding="utf-8-sig")
    zip_path = export_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in export_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(export_dir))
    return zip_path


def persist_state(project_root: Path, state: Dict[str, Any]) -> None:
    save_state(project_root, state)



st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("최종 이미지·mask 업로드 · 검토 · 품질 라벨 · handoff export")

# Sidebar project management
with st.sidebar:
    st.header("프로젝트")
    projects_root = Path(st.text_input("프로젝트 폴더", value=st.session_state.get("projects_root", DEFAULT_PROJECTS_ROOT), key="projects_root_input"))
    ensure_dir(projects_root)
    st.session_state["projects_root"] = str(projects_root)

    existing_projects = sorted([p.name for p in projects_root.iterdir() if p.is_dir()])
    project_mode = st.radio("작업 방식", ["기존 프로젝트 열기", "새 프로젝트 만들기"], key="project_mode")
    if project_mode == "기존 프로젝트 열기" and existing_projects:
        selected_project_name = st.selectbox("프로젝트 선택", existing_projects, key="selected_project_name")
    else:
        selected_project_name = st.text_input("새 프로젝트 이름", value=st.session_state.get("new_project_name", "schlieren_project"), key="new_project_name")

    if st.button("프로젝트 열기/생성", type="primary"):
        project_name = selected_project_name.strip()
        project_root = projects_root / project_name
        if not project_name:
            st.error("프로젝트 이름을 입력해주세요.")
        elif project_mode == "새 프로젝트 만들기" and project_root.exists():
            st.error("같은 이름의 프로젝트가 이미 존재합니다. 다른 이름을 사용해주세요.")
        else:
            ensure_dir(project_root)
            state = load_state(project_root)
            st.session_state["active_project_root"] = str(project_root)
            st.session_state["project_state"] = state
            save_state(project_root, state)
            st.success(f"프로젝트 준비 완료: {project_root.name}")

    active_project_root = st.session_state.get("active_project_root")
    if active_project_root:
        st.info(f"현재 프로젝트: {Path(active_project_root).name}")

if "active_project_root" not in st.session_state:
    st.warning("왼쪽 사이드바에서 프로젝트를 먼저 열어주세요.")
    st.stop()

project_root = Path(st.session_state["active_project_root"])
state = st.session_state.get("project_state", load_state(project_root))
st.session_state["project_state"] = state
pp = get_project_paths(project_root)
registry_df = build_registry_dataframe(state)
specimen_ids = registry_df["specimen_id"].tolist() if not registry_df.empty else []
if specimen_ids and not state["ui"].get("selected_specimen"):
    state["ui"]["selected_specimen"] = specimen_ids[0]
selected_specimen = state["ui"].get("selected_specimen", specimen_ids[0] if specimen_ids else "")
selected_specimen_display = get_specimen_display_name(registry_df, selected_specimen)

st.markdown("### 현재 작업 시편")
if specimen_ids:
    default_idx = max(0, specimen_ids.index(selected_specimen)) if selected_specimen in specimen_ids else 0
    if "selected_specimen_widget" not in st.session_state or st.session_state.get("selected_specimen_widget") not in specimen_ids:
        st.session_state["selected_specimen_widget"] = specimen_ids[default_idx]
    new_specimen = st.selectbox("작업할 시편 선택", specimen_ids, key="selected_specimen_widget")
    if new_specimen != state["ui"].get("selected_specimen", ""):
        state["ui"]["selected_specimen"] = new_specimen
        selected_specimen = new_specimen
        persist_state(project_root, state)
    else:
        selected_specimen = new_specimen
    selected_specimen_display = get_specimen_display_name(registry_df, selected_specimen)
    st.caption(f"선택된 시편: {selected_specimen_display}")
else:
    st.info("실험표를 먼저 저장하면 시편 목록이 만들어집니다.")

setup_tab, upload_tab, review_tab, quality_tab, tools_tab, export_tab = st.tabs([
    "실험표/설정", "업로드", "Review", "Quality Label", "Tools", "Export"
])

with setup_tab:
    st.subheader("실험표 설정")
    st.info("이 앱에서는 전처리·ROI·웹 내 라벨링을 수행하지 않습니다. OpenCV/Python rule-based 전처리 → LabelMe 라벨링 → U-Net 자동 라벨링/수정 후 최종 PNG mask를 업로드해주세요.")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("**실험 반복 횟수**")
        state["settings"]["repeat_count"] = st.number_input("같은 실험 반복 횟수", min_value=1, step=1, value=int(state["settings"]["repeat_count"]))

        st.markdown("**시편 이름 설정**")
        naming_mode = st.radio("이름 방식", ["auto", "custom"], horizontal=True, format_func=lambda x: "자동 생성" if x == "auto" else "직접 입력", index=0 if state["settings"]["naming_mode"] == "auto" else 1)
        state["settings"]["naming_mode"] = naming_mode
        if naming_mode == "auto":
            a, b = st.columns(2)
            with a:
                state["settings"]["specimen_prefix"] = st.text_input("접두어", value=state["settings"]["specimen_prefix"])
            with b:
                state["settings"]["start_number"] = st.number_input("시작 번호", min_value=1, step=1, value=int(state["settings"]["start_number"]))
        else:
            custom_text = st.text_area("시편 이름 목록", value="\n".join(state["settings"].get("custom_specimen_names", [])), help="한 줄에 하나씩 입력")
            state["settings"]["custom_specimen_names"] = [x.strip() for x in custom_text.splitlines() if x.strip()]

        st.markdown("**Laser 출력**")
        laser_unit = st.radio("출력 단위", ["W", "kW"], horizontal=True, index=0 if state["settings"]["laser_unit"] == "W" else 1)
        state["settings"]["laser_unit"] = laser_unit
        for i, row in enumerate(state["settings"]["laser_levels"]):
            cols = st.columns([0.8, 1.2, 1.2])
            row["enabled"] = cols[0].checkbox(f"사용 {i+1}", value=bool(row.get("enabled", True)), key=f"laser_enabled_{i}")
            row["label"] = cols[1].text_input(f"Laser 이름 {i+1}", value=row.get("label", f"level{i+1}"), key=f"laser_label_{i}")
            row["value"] = cols[2].number_input(f"Laser 값 {i+1}", value=float(row.get("value", 0.0)), step=0.1 if laser_unit == "kW" else 1.0, key=f"laser_value_{i}")

        st.markdown("**용접 속도**")
        speed_unit = st.radio("속도 입력 단위", ["mpm", "mm/s"], horizontal=True, index=0 if state["settings"]["speed_unit"] == "mpm" else 1)
        state["settings"]["speed_unit"] = speed_unit
        speed_text = st.text_input("속도 값들", value=", ".join(str(x) for x in state["settings"]["speed_levels"]), help="쉼표로 여러 개 입력")
        state["settings"]["speed_levels"] = parse_simple_levels(speed_text, float)
        if state["settings"]["speed_levels"]:
            if speed_unit == "mpm":
                st.caption("자동 변환(mm/s): " + ", ".join(f"{mpm_to_mms(v):.3f}" for v in state["settings"]["speed_levels"]))
            else:
                st.caption("자동 변환(mpm): " + ", ".join(f"{mms_to_mpm(v):.3f}" for v in state["settings"]["speed_levels"]))

        st.markdown("**추가 인자**")
        gap_text = st.text_input("Gap 수준", value=", ".join(state["settings"].get("gap_levels", [])))
        def_text = st.text_input("Defocusing 수준", value=", ".join(state["settings"].get("defocusing_levels", [])))
        state["settings"]["gap_levels"] = [x.strip() for x in gap_text.split(",") if x.strip()]
        state["settings"]["defocusing_levels"] = [x.strip() for x in def_text.split(",") if x.strip()]

    with right:
        st.markdown("**촬영/실험 외부 조건**")
        ec = state["settings"]["external_conditions"]
        ec["material"] = st.text_input("소재", value=ec.get("material", "STS316L"))
        ec["thickness_t"] = st.text_input("두께", value=ec.get("thickness_t", "0.7t"))
        ec["test_stage"] = st.selectbox("실험 단계", ["feasibility", "main_experiment"], index=0 if ec.get("test_stage") == "feasibility" else 1)
        ec["test_date"] = st.text_input("실험 날짜", value=ec.get("test_date", ""))
        ec["backlight_power_w"] = st.number_input("백라이트 세기 (W)", value=float(ec.get("backlight_power_w", 445.0)))
        ec["pattern_to_backlight_cm"] = st.number_input("패턴-백라이트 거리 (cm)", value=float(ec.get("pattern_to_backlight_cm", 20.0)))
        ec["pattern_to_weld_center_cm"] = st.number_input("패턴-용접부 중앙 거리 (cm)", value=float(ec.get("pattern_to_weld_center_cm", 50.0)))
        ec["specimen_to_lens_cm"] = st.number_input("시편-카메라렌즈 거리 (cm)", value=float(ec.get("specimen_to_lens_cm", 95.0)))
        ec["aperture"] = st.text_input("조리개", value=ec.get("aperture", "E"))
        ec["fps"] = st.number_input("프레임 (fps)", min_value=1, value=int(ec.get("fps", 2000)))
        ec["shutter_speed"] = st.text_input("셔터 스피드", value=ec.get("shutter_speed", "1/5000"))
        ec["resolution"] = st.text_input("해상도", value=ec.get("resolution", "1280x1024"))
        ec["weld_direction"] = st.selectbox("용접 진행 방향", ["right_to_left", "left_to_right"], index=0 if ec.get("weld_direction") == "right_to_left" else 1)

    registry_df = build_registry_dataframe(state)
    st.markdown("**실험표 미리보기**")
    st.dataframe(registry_df, use_container_width=True, height=300)
    if st.button("설정 저장 / 실험표 갱신", type="primary"):
        persist_state(project_root, state)
        st.success("저장되었습니다.")

with upload_tab:
    st.subheader(f"업로드 · {selected_specimen_display or '-'}")
    if not selected_specimen:
        st.info("실험표를 먼저 저장해주세요.")
    else:
        assets = specimen_assets(project_root, selected_specimen)
        mask_dir = specimen_mask_dir(project_root, selected_specimen)
        pred_dir = specimen_prediction_dir(project_root, selected_specimen)
        st.caption("권장 포맷: 이미지 파일명과 mask 파일명 stem 일치 (예: sample_001.jpg ↔ sample_001.png). Mask PNG 값은 background=0, fume=1, spatter=2, ignore=3 을 권장합니다.")
        st.info("최종 이미지 업로드는 원본/검토 기준 이미지(raw 성격의 최종 이미지) ZIP 1개만 받습니다. 새 ZIP을 적용하면 기존 이미지 업로드는 전부 삭제되고 교체됩니다.")

        flash_key = f"upload_flash_{selected_specimen}"
        flash_message = st.session_state.pop(flash_key, None)
        if flash_message:
            st.success(flash_message)

        frame_nonce_key = f"frame_zip_nonce_{selected_specimen}"
        mask_nonce_key = f"mask_upload_nonce_{selected_specimen}"
        pred_nonce_key = f"pred_upload_nonce_{selected_specimen}"
        st.session_state.setdefault(frame_nonce_key, 0)
        st.session_state.setdefault(mask_nonce_key, 0)
        st.session_state.setdefault(pred_nonce_key, 0)

        img_col, mask_col, pred_col = st.columns(3)
        with img_col:
            up_frames_zip = st.file_uploader(
                "최종 이미지 ZIP 업로드",
                accept_multiple_files=False,
                type=["zip"],
                key=f"frames_zip_{selected_specimen}_{st.session_state[frame_nonce_key]}",
                help="2000장 이상처럼 많은 이미지도 ZIP 1개로 업로드하세요. 적용 시 기존 이미지는 누적되지 않고 전체 교체됩니다.",
            )
            if up_frames_zip is not None:
                st.caption(f"선택됨: {up_frames_zip.name} · {up_frames_zip.size / (1024 * 1024):.1f} MB")
                if st.button("최종 이미지 ZIP 적용", type="primary", key=f"apply_frames_zip_{selected_specimen}_{st.session_state[frame_nonce_key]}"):
                    with st.spinner("이미지 ZIP 압축 해제 및 교체 중..."):
                        cnt = replace_uploaded_frames_from_zip(up_frames_zip, assets.specimen_dir)
                        st.cache_data.clear()
                    st.session_state[flash_key] = f"이미지 교체 완료: {cnt}장"
                    st.session_state[frame_nonce_key] += 1
                    st.rerun()
        with mask_col:
            up_masks = st.file_uploader(
                "최종 Mask PNG 업로드",
                accept_multiple_files=True,
                type=["png", "zip"],
                key=f"masks_{selected_specimen}_{st.session_state[mask_nonce_key]}",
            )
            if up_masks:
                st.caption(f"선택된 파일 수: {len(up_masks)}")
                if st.button("Mask 업로드 적용", key=f"apply_masks_{selected_specimen}_{st.session_state[mask_nonce_key]}"):
                    with st.spinner("Mask 업로드 중..."):
                        cnt = save_uploaded_png_sets(up_masks, mask_dir)
                        st.cache_data.clear()
                    st.session_state[flash_key] = f"Mask 저장 완료: {cnt}장"
                    st.session_state[mask_nonce_key] += 1
                    st.rerun()
        with pred_col:
            up_preds = st.file_uploader(
                "예측 Mask 업로드(선택)",
                accept_multiple_files=True,
                type=["png", "zip"],
                key=f"preds_{selected_specimen}_{st.session_state[pred_nonce_key]}",
            )
            if up_preds:
                st.caption(f"선택된 파일 수: {len(up_preds)}")
                if st.button("예측 Mask 업로드 적용", key=f"apply_preds_{selected_specimen}_{st.session_state[pred_nonce_key]}"):
                    with st.spinner("예측 mask 업로드 중..."):
                        cnt = save_uploaded_png_sets(up_preds, pred_dir)
                        st.cache_data.clear()
                    st.session_state[flash_key] = f"예측 mask 저장 완료: {cnt}장"
                    st.session_state[pred_nonce_key] += 1
                    st.rerun()

        m1, m2, m3, m4 = st.columns(4)
        mask_count = len(list(mask_dir.glob("*.png")))
        pred_count = len(list(pred_dir.glob("*.png")))
        m1.metric("현재 이미지 수", len(assets.frame_paths))
        m2.metric("최종 mask 수", mask_count)
        m3.metric("예측 mask 수", pred_count)
        dataset_df = build_dataset_rows(project_root, selected_specimen, state)
        ready_count = int((dataset_df["status"] == "ready").sum()) if not dataset_df.empty else 0
        m4.metric("매칭 완료", ready_count)

        action1, action2, action3 = st.columns(3)
        if action1.button("현재 시편 이미지 비우기"):
            reset_uploaded_frames(assets.specimen_dir)
            st.cache_data.clear()
            st.session_state[flash_key] = "현재 시편 이미지 업로드를 비웠습니다."
            st.session_state[frame_nonce_key] += 1
            st.rerun()
        if action2.button("현재 시편 mask 비우기"):
            shutil.rmtree(mask_dir, ignore_errors=True)
            ensure_dir(mask_dir)
            st.cache_data.clear()
            st.session_state[flash_key] = "현재 시편 mask를 비웠습니다."
            st.session_state[mask_nonce_key] += 1
            st.rerun()
        if action3.button("현재 시편 prediction 비우기"):
            shutil.rmtree(pred_dir, ignore_errors=True)
            ensure_dir(pred_dir)
            st.cache_data.clear()
            st.session_state[flash_key] = "현재 시편 prediction을 비웠습니다."
            st.session_state[pred_nonce_key] += 1
            st.rerun()

        if not dataset_df.empty:
            st.markdown("**업로드 후 자동 매칭 요약**")
            summary = {
                "총 항목": len(dataset_df),
                "정상 매칭": int((dataset_df["status"] == "ready").sum()),
                "mask 누락": int((dataset_df["status"] == "missing_mask").sum()),
                "이미지 누락": int((dataset_df["status"] == "missing_image").sum()),
                "해상도 불일치": int((~dataset_df["size_match"] & dataset_df["has_image"] & dataset_df["has_mask"]).sum()),
            }
            st.json(summary)
            st.dataframe(dataset_df[["item_id", "image_name", "mask_name", "prediction_name", "status", "image_size", "mask_size", "mask_values", "invalid_values"]], use_container_width=True, height=280)

with review_tab:
    st.subheader(f"Review · {selected_specimen_display or '-'}")
    if not selected_specimen:
        st.info("실험표를 먼저 저장해주세요.")
    else:
        dataset_df = build_dataset_rows(project_root, selected_specimen, state)
        if dataset_df.empty:
            st.info("먼저 이미지와 mask를 업로드해주세요.")
        else:
            with st.expander("필터", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                status_filter = c1.multiselect("상태", sorted(dataset_df["status"].unique().tolist()), default=sorted(dataset_df["status"].unique().tolist()))
                need_ignore = c2.checkbox("ignore 포함만", value=False)
                need_unreviewed = c3.checkbox("품질 미입력/미검토만", value=False)
                show_only_invalid = c4.checkbox("문제 있는 항목만", value=False)
            filtered = dataset_df[dataset_df["status"].isin(status_filter)].copy()
            if need_ignore:
                filtered = filtered[filtered["has_ignore"]]
            if need_unreviewed:
                filtered = filtered[(filtered["quality_label"] == "미입력") | (~filtered["reviewed"])]
            if show_only_invalid:
                filtered = filtered[(filtered["status"] != "ready") | (~filtered["size_match"]) | (filtered["invalid_values"] != "") | (filtered["empty_mask"]) | (filtered["ignore_only"])]

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("필터 결과", len(filtered))
            s2.metric("매칭 완료", int((filtered["status"] == "ready").sum()))
            s3.metric("문제 항목", int(((filtered["status"] != "ready") | (~filtered["size_match"]) | (filtered["invalid_values"] != "") | (filtered["empty_mask"]) | (filtered["ignore_only"])).sum()))
            s4.metric("미검토", int(((filtered["quality_label"] == "미입력") | (~filtered["reviewed"])).sum()))
            st.caption("Review 탭에서는 선택한 프레임만 간단히 확인합니다. 상세 자동 점검표(item_id / has_fume / has_spatter 등)는 Tools 탭에서 확인할 수 있습니다.")

            item_ids = filtered["item_id"].tolist() or dataset_df["item_id"].tolist()
            item_id = st.selectbox("검토할 항목", item_ids)
            row = dataset_df[dataset_df["item_id"] == item_id].iloc[0].to_dict()
            image_path = None
            mask_path = None
            pred_path = None
            assets = specimen_assets(project_root, selected_specimen)
            for p in assets.frame_paths:
                if p.stem == item_id:
                    image_path = p
                    break
            candidate_mask = specimen_mask_dir(project_root, selected_specimen) / f"{item_id}.png"
            if candidate_mask.exists():
                mask_path = candidate_mask
            candidate_pred = specimen_prediction_dir(project_root, selected_specimen) / f"{item_id}.png"
            if candidate_pred.exists():
                pred_path = candidate_pred

            badge_parts = [f"status={row['status']}"]
            if row["quality_label"] != "미입력":
                badge_parts.append(f"quality={row['quality_label']}")
            badge_parts.append(f"reviewed={bool(row['reviewed'])}")
            if row["has_prediction"]:
                badge_parts.append("prediction=Y")
            st.markdown(" · ".join(badge_parts))

            with st.expander("보기 옵션", expanded=True):
                opt1, opt2, opt3, opt4 = st.columns(4)
                show_mask = opt1.checkbox("Mask 보기", value=False)
                show_overlay = opt2.checkbox("Overlay 보기", value=False)
                show_ignore = opt3.checkbox("Ignore 표시", value=False)
                show_prediction = opt4.checkbox("Prediction 보기", value=False)
                opacity = st.slider("Overlay 투명도", min_value=0.1, max_value=0.9, value=0.35, step=0.05)

            preview_cols = []
            if image_path and image_path.exists():
                image_preview = try_load_frame_image(str(image_path))
                if image_preview is not None:
                    preview_cols.append(("원본 이미지", image_preview))
            if mask_path and mask_path.exists() and show_mask:
                mask = load_mask_image(str(mask_path))
                preview_cols.append(("Mask", render_mask_rgb(mask, show_ignore=show_ignore)))
            if image_path and mask_path and show_overlay:
                image = try_load_frame_image(str(image_path))
                mask = load_mask_image(str(mask_path))
                if image is not None:
                    preview_cols.append(("Overlay", blend_overlay(image, mask, alpha=opacity, show_ignore=show_ignore)))
            if pred_path and pred_path.exists() and show_prediction:
                pred_mask = load_mask_image(str(pred_path))
                preview_cols.append(("Prediction", render_mask_rgb(pred_mask, show_ignore=show_ignore)))
                if image_path:
                    pred_base = try_load_frame_image(str(image_path))
                    if pred_base is not None:
                        preview_cols.append(("Prediction Overlay", blend_overlay(pred_base, pred_mask, alpha=opacity, show_ignore=show_ignore)))

            if preview_cols:
                cols = st.columns(min(4, len(preview_cols)))
                for idx, (title, img) in enumerate(preview_cols):
                    cols[idx % len(cols)].image(img, caption=title, use_container_width=True)
            else:
                st.info("현재 보기 옵션으로 표시할 미리보기가 없습니다. Mask/Overlay 토글을 켜보세요.")

            with st.expander("선택 항목의 자동 분석 정보", expanded=False):
                meta1, meta2 = st.columns(2)
                with meta1:
                    st.markdown("**파일/매칭 정보**")
                    st.json({
                        "item_id": row["item_id"],
                        "status": row["status"],
                        "image_size": row["image_size"],
                        "mask_size": row["mask_size"],
                        "mask_values": row["mask_values"],
                        "invalid_values": row["invalid_values"],
                    })
                with meta2:
                    st.markdown("**mask 자동 판독 요약**")
                    st.json({
                        "fume": bool(row["has_fume"]),
                        "spatter": bool(row["has_spatter"]),
                        "ignore": bool(row["has_ignore"]),
                        "empty_mask": bool(row["empty_mask"]),
                        "ignore_only": bool(row["ignore_only"]),
                        "prediction": bool(row["has_prediction"]),
                    })

with quality_tab:
    st.subheader("Quality Label")
    if not selected_specimen:
        st.info("실험표를 먼저 저장해주세요.")
    else:
        current_truth = state.get("truth", {}).get(selected_specimen, {})
        reviewed = st.checkbox("검토 완료", value=bool(current_truth.get("reviewed", False)), key=f"label_reviewed_{selected_specimen}")
        visual_result = st.radio("외관 검사 결과", ["미입력", "정상", "불량"], index=["미입력", "정상", "불량"].index(current_truth.get("visual_result", "미입력") if current_truth.get("visual_result", "미입력") in ["미입력", "정상", "불량"] else "미입력"), horizontal=True, key=f"label_visual_{selected_specimen}")
        tensile_result = st.radio("인장 시험 결과", ["미입력", "정상", "불량"], index=["미입력", "정상", "불량"].index(current_truth.get("tensile_result", "미입력") if current_truth.get("tensile_result", "미입력") in ["미입력", "정상", "불량"] else "미입력"), horizontal=True, key=f"label_tensile_{selected_specimen}")
        final_label = auto_final_label(visual_result, tensile_result, current_truth.get("final_label", "미입력"))
        st.markdown(f"**자동 최종 판정:** `{final_label}`")
        st.caption("규칙: 외관 검사 결과 또는 인장 시험 결과 중 하나라도 불량이면 최종 판정은 자동으로 불량입니다.")
        defect_type = st.text_input("불량 유형", value=current_truth.get("defect_type", ""), key=f"label_defect_{selected_specimen}")
        notes = st.text_area("메모", value=current_truth.get("notes", ""), key=f"label_notes_{selected_specimen}")
        detail_auto = f"{selected_specimen} 시편은 최종 {final_label}입니다. 외관 검사 결과는 {visual_result}, 인장 시험 결과는 {tensile_result}입니다."
        summary_detail = st.text_area("상세 문장", value=current_truth.get("details", detail_auto), key=f"label_detail_{selected_specimen}")
        if st.button("현재 시편 품질 라벨 저장", type="primary"):
            state.setdefault("truth", {})[selected_specimen] = {
                "final_label": final_label,
                "visual_result": visual_result,
                "tensile_result": tensile_result,
                "defect_type": defect_type,
                "notes": notes,
                "details": summary_detail,
                "reviewed": reviewed,
            }
            persist_state(project_root, state)
            st.success(f"시편 품질 라벨이 저장되었습니다. 자동 최종 판정: {final_label}")

        with st.expander("표 형태 일괄 편집", expanded=False):
            truth_df = build_truth_dataframe(state)
            if not truth_df.empty:
                edit_df = truth_df.copy()
                edit_df["final_label"] = [
                    auto_final_label(v, t, f)
                    for v, t, f in zip(
                        edit_df.get("visual_result", pd.Series(["미입력"] * len(edit_df))),
                        edit_df.get("tensile_result", pd.Series(["미입력"] * len(edit_df))),
                        edit_df.get("final_label", pd.Series(["미입력"] * len(edit_df))),
                    )
                ]
                if "reviewed" not in edit_df.columns:
                    edit_df["reviewed"] = [bool(state.get("truth", {}).get(sid, {}).get("reviewed", False)) for sid in edit_df["specimen_id"]]
                st.caption("final_label은 외관 검사 결과 / 인장 시험 결과에 따라 자동 계산됩니다. 저장 시 수동 입력값이 있더라도 자동 규칙이 우선합니다.")
                edited = st.data_editor(edit_df, use_container_width=True, num_rows="fixed")
                if st.button("표 편집 내용 저장"):
                    truth = {}
                    for _, row in edited.iterrows():
                        sid = str(row["specimen_id"])
                        visual_result = row.get("visual_result", "")
                        tensile_result = row.get("tensile_result", "")
                        truth[sid] = {
                            "final_label": auto_final_label(visual_result, tensile_result, row.get("final_label", "미입력")),
                            "visual_result": visual_result,
                            "tensile_result": tensile_result,
                            "defect_type": row.get("defect_type", ""),
                            "notes": row.get("notes", ""),
                            "details": row.get("summary_detail", ""),
                            "reviewed": bool(row.get("reviewed", False)),
                        }
                    state["truth"] = truth
                    persist_state(project_root, state)
                    st.success("일괄 저장되었습니다.")

with tools_tab:
    st.subheader(f"Tools · {selected_specimen_display or '-'}")
    if not selected_specimen:
        st.info("실험표를 먼저 저장해주세요.")
    else:
        dataset_df = build_dataset_rows(project_root, selected_specimen, state)
        if dataset_df.empty:
            st.info("먼저 이미지와 mask를 업로드해주세요.")
        else:
            st.caption("이 탭의 값들은 사람이 프레임마다 입력하는 항목이 아니라, 업로드된 이미지와 mask를 앱이 자동 매칭·자동 판독해서 만든 점검 정보입니다.")
            t1, t2, t3, t4, t5 = st.columns(5)
            t1.metric("총 항목", len(dataset_df))
            t2.metric("mask 누락", int((dataset_df["status"] == "missing_mask").sum()))
            t3.metric("이미지 누락", int((dataset_df["status"] == "missing_image").sum()))
            t4.metric("해상도 불일치", int((~dataset_df["size_match"] & dataset_df["has_image"] & dataset_df["has_mask"]).sum()))
            t5.metric("잘못된 mask 값", int((dataset_df["invalid_values"] != "").sum()))

            with st.expander("문제 항목 보기", expanded=True):
                st.dataframe(
                    dataset_df[(dataset_df["status"] != "ready") | (~dataset_df["size_match"]) | (dataset_df["invalid_values"] != "") | (dataset_df["empty_mask"]) | (dataset_df["ignore_only"])][[
                        "item_id", "status", "image_size", "mask_size", "invalid_values", "empty_mask", "ignore_only"
                    ]],
                    use_container_width=True,
                    height=240,
                )

            with st.expander("자동 매칭 상세표", expanded=False):
                st.dataframe(
                    dataset_df[["item_id", "status", "has_fume", "has_spatter", "has_ignore", "size_match", "invalid_values", "quality_label", "reviewed"]],
                    use_container_width=True,
                    height=260,
                )

            with st.expander("클래스 분포 / 필터용 요약", expanded=False):
                class_summary = {
                    "fume 포함": int(dataset_df["has_fume"].sum()),
                    "spatter 포함": int(dataset_df["has_spatter"].sum()),
                    "ignore 포함": int(dataset_df["has_ignore"].sum()),
                    "empty mask": int(dataset_df["empty_mask"].sum()),
                    "ignore only": int(dataset_df["ignore_only"].sum()),
                    "prediction 존재": int(dataset_df["has_prediction"].sum()),
                }
                st.json(class_summary)

            with st.expander("데이터셋 CSV 다운로드", expanded=False):
                csv_bytes = dataset_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button("현재 시편 dataset CSV 다운로드", data=csv_bytes, file_name=f"{selected_specimen}_dataset_manifest.csv", mime="text/csv")

with export_tab:
    st.subheader("Export")
    truth_df = build_truth_dataframe(state)
    if truth_df.empty:
        st.info("실험표를 먼저 저장해주세요.")
    else:
        st.caption("검토/추출 전용입니다. 입력/수정은 Quality Label 탭에서 진행하세요.")
        for _, row in truth_df.iterrows():
            sid = row["specimen_id"]
            reviewed = bool(state.get("truth", {}).get(sid, {}).get("reviewed", False))
            with st.expander(f"{sid} · {row['summary_short']} · 검토완료={reviewed}", expanded=False):
                st.write(f"**최종 판정:** {row['final_label']}")
                st.write(f"**외관 검사:** {row.get('visual_result', '') or '미입력'}")
                st.write(f"**인장 시험:** {row.get('tensile_result', '') or '미입력'}")
                st.write(f"**결함 유형:** {row.get('defect_type', '') or '-'}")
                st.write(f"**메모:** {row.get('notes', '') or '-'}")
                st.write(f"**상세 문장:** {row.get('summary_detail', '') or '-'}")
                specimen_dataset = build_dataset_rows(project_root, sid, state)
                st.write(f"**매칭 완료:** {int((specimen_dataset['status'] == 'ready').sum()) if not specimen_dataset.empty else 0} / {len(specimen_dataset)}")
        if st.button("handoff 패키지 추출"):
            zip_path = export_handoff(project_root, state)
            st.success(f"추출 완료: {zip_path.name}")
            with open(zip_path, "rb") as f:
                st.download_button("handoff zip 다운로드", f, file_name=zip_path.name, mime="application/zip")
