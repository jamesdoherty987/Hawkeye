"""
hawkeye_live.py — Product loop: Align → Play → optional instant replay.

Composes existing modules (does not replace them):
  stereo_config          rig geometry
  stereo_calibrate       projection / triangulation
  sky_motion             motion wake for YOLO
  stereo_plane_calib     plane draw, click Align, ball call UI helpers

Modes
─────
  ALIGN  Click BOTTOM then TOP of the FAR post on LEFT, then RIGHT.
         Overlay snaps; Enter / P when both cams are locked.
  PLAY   Motion-gated YOLO + stereo 3D + GOAL / WIDE / IN FIELD.
  REPLAY Scrub buffer with ←/→ or A/D. Enter / P or I back to Play.

Keys
────
  A           Align mode
  Enter / P   Play
  I           Instant replay toggle
  ←/→ or A/D  Scrub replay (in REPLAY only for A; A alone = Align in other modes)
  B           Toggle ball detection
  D           Toggle verbose dbg line
  C / K       Clear clicks / re-snap (Align)
  O / R / S / L   rotate / reset / save / load
  Q / Esc     Quit

Usage
─────
  python capture/hawkeye_live.py --left 2 --right 1
  python capture/hawkeye_live.py --left 2 --right 1 --imgsz 320
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from enum import Enum, auto

import cv2
import numpy as np
from ultralytics import YOLO

import stereo_config as cfg
import stereo_plane_calib as plane
from sky_motion import SkyMotionDetector
from stereo_calibrate import make_proj_matrix


WINDOW = "Hawkeye Live"
REPLAY_S = 8.0
REPLAY_SCALE = 0.5  # store smaller frames in the ring buffer (CPU)


class Mode(Enum):
    ALIGN = auto()
    PLAY = auto()
    REPLAY = auto()


def window_client_size(name: str = WINDOW) -> tuple[int, int] | None:
    try:
        rect = cv2.getWindowImageRect(name)
        if rect is not None and len(rect) >= 4 and int(rect[2]) > 16 and int(rect[3]) > 16:
            return int(rect[2]), int(rect[3])
    except cv2.error:
        pass
    return None


def _click_to_panel(
    mx: int,
    my: int,
    layout: dict,
    view_l: np.ndarray,
    view_r: np.ndarray,
) -> tuple[str, tuple[int, int]] | None:
    nw, nh = layout["nw"], layout["nh"]
    if nw <= 0 or nh <= 0:
        return None
    x0, y0 = layout["x0"], layout["y0"]
    if not (x0 <= mx < x0 + nw and y0 <= my < y0 + nh):
        return None
    sx = (mx - x0) * layout["src_w"] / nw
    sy = (my - y0) * layout["src_h"] / nh
    pair_h = layout["pair_h"]
    split_x = layout["split_x"]
    if not (0 <= sy < pair_h):
        return None
    if sx < split_x and split_x > 0:
        x = max(0, min(view_l.shape[1] - 1, int(sx * view_l.shape[1] / split_x)))
        y = max(0, min(view_l.shape[0] - 1, int(sy * view_l.shape[0] / pair_h)))
        return "L", (x, y)
    if layout["src_w"] > split_x:
        lx = sx - split_x
        rw = layout["src_w"] - split_x
        x = max(0, min(view_r.shape[1] - 1, int(lx * view_r.shape[1] / rw)))
        y = max(0, min(view_r.shape[0] - 1, int(sy * view_r.shape[0] / pair_h)))
        return "R", (x, y)
    return None


def _try_snap(
    side: str,
    clicks: plane.ClickState,
    P1: np.ndarray,
    P2: np.ndarray,
    baseline: float,
    extend_up: float,
    snap_l: np.ndarray | None,
    snap_r: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """Returns (snap_l, snap_r, changed)."""
    half = baseline / 2.0
    changed = False
    if side == "L" and len(clicks.far_l) >= 2:
        aff, _m, note = plane.overlay_snap_to_clicks(
            P1, clicks.far_l, +half, 0.0, extend_up,
        )
        if aff is None:
            print(f"  Left snap skipped — {note}")
        else:
            snap_l = aff
            changed = True
            print(f"  Left overlay locked  {note}")
    if side == "R" and len(clicks.far_r) >= 2:
        aff, _m, note = plane.overlay_snap_to_clicks(
            P2, clicks.far_r, -half, 0.0, extend_up,
        )
        if aff is None:
            print(f"  Right snap skipped — {note}")
        else:
            snap_r = aff
            changed = True
            print(f"  Right overlay locked  {note}")
    return snap_l, snap_r, changed


def _shrink(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return frame.copy()
    h, w = frame.shape[:2]
    return cv2.resize(
        frame,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _grow_to(frame: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    w, h = wh
    if frame.shape[1] == w and frame.shape[0] == h:
        return frame
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)


def _mode_badge(img: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    """Top-left mode chip on the combined view."""
    pad_x, pad_y = 10, 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.7, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x1, y1 = pad_x, pad_y
    x2, y2 = pad_x + tw + 16, pad_y + th + 14
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    cv2.putText(img, text, (x1 + 8, y2 - 8), font, scale, color, thick, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hawkeye live: Align → Play → Replay")
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--right", type=int, default=1)
    parser.add_argument("--baseline", type=float, default=cfg.BASELINE_M)
    parser.add_argument("--behind", type=float, default=cfg.CAM_BEHIND_POST_M)
    parser.add_argument("--h-angle", type=float, default=cfg.H_ANGLE_DEG)
    parser.add_argument("--v-angle", type=float, default=cfg.V_ANGLE_DEG)
    parser.add_argument("--extend", type=float, default=4.0)
    parser.add_argument("--focal", type=float, default=None)
    parser.add_argument("--rotate", type=int, default=None, choices=(0, 90, 270))
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--conf", type=float, default=plane.CONFIDENCE)
    parser.add_argument("--imgsz", type=int, default=plane.IMAGE_SIZE)
    parser.add_argument("--yolo-stride", type=int, default=plane.IDLE_YOLO_STRIDE)
    parser.add_argument("--replay-s", type=float, default=REPLAY_S)
    parser.add_argument("--no-motion-gate", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Probing cameras 0–5:")
        for i in range(6):
            try:
                cap = plane.open_camera(i)
                print(f"  → {i} OK")
                cap.release()
                time.sleep(0.3)
            except RuntimeError as e:
                print(f"  → {i} FAIL: {e}")
        return 0

    if args.left == args.right:
        print("ERROR: --left and --right must differ", file=sys.stderr)
        return 1

    baseline = float(args.baseline)
    behind = float(args.behind)
    extend_up = float(args.extend)
    h_left = float(args.h_angle)
    h_right = -float(args.h_angle)
    v_angle = float(args.v_angle)
    rotate_deg = int(args.rotate) if args.rotate is not None else int(cfg.SIDEWAYS_ROTATE_DEG)
    if rotate_deg not in (0, 90, 270):
        rotate_deg = 90
    yolo_imgsz = max(160, int(args.imgsz))
    motion_gate = not bool(args.no_motion_gate)
    replay_s = max(2.0, float(args.replay_s))

    clicks = plane.ClickState()
    clicks.mode = "far"
    snap_l: np.ndarray | None = None
    snap_r: np.ndarray | None = None
    mode = Mode.ALIGN
    ball_on = True
    verbose_dbg = False

    model: YOLO | None = None
    ball_classes: list[int] | None = None
    ball_dets: tuple = (None, None)
    ball_sched = plane.BallDetectScheduler(idle_stride=int(args.yolo_stride))
    motion_l = SkyMotionDetector(warmup_frames=8)
    motion_r = SkyMotionDetector(warmup_frames=8)
    ball_latch: dict = {"prev_z": None, "latch": None, "latch_until": 0.0}

    # (small_l, small_r, call, t) — downscaled to keep PLAY smooth
    ring: deque[tuple[np.ndarray, np.ndarray, str, float]] = deque()
    replay_idx = 0
    last_event_call = ""
    raw_wh_l = (640, 480)
    raw_wh_r = (640, 480)
    fps_ema = 0.0
    last_t = time.time()
    frame_i = 0

    mouse = {"x": 0, "y": 0, "clicked": False}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["x"] = x
            mouse["y"] = y
            mouse["clicked"] = True

    layout = {
        "x0": 0, "y0": 0, "nw": 1, "nh": 1,
        "src_w": 1, "src_h": 1, "split_x": 0, "bar_h": 0, "pair_h": 1,
    }

    cap_l = cap_r = None
    try:
        print(f"\nOpening left camera ({args.left})...")
        cap_l = plane.open_camera(args.left)
        time.sleep(0.4)
        print(f"Opening right camera ({args.right})...")
        cap_r = plane.open_camera(args.right)
        ok_l, probe_l = plane.read_cam(cap_l)
        ok_r, probe_r = plane.read_cam(cap_r)
        if not ok_l or probe_l is None or not ok_r or probe_r is None:
            print("ERROR: could not read initial frames.", file=sys.stderr)
            return 1
        raw_wh_l = (probe_l.shape[1], probe_l.shape[0])
        raw_wh_r = (probe_r.shape[1], probe_r.shape[0])

        view_l = plane.orient_frame(probe_l, rotate_deg)
        view_r = plane.orient_frame(probe_r, rotate_deg)
        img_w, img_h = view_l.shape[1], view_l.shape[0]
        img_w_r, img_h_r = view_r.shape[1], view_r.shape[0]
        if args.focal is not None:
            focal_px = float(args.focal)
            focal_r = focal_px * (img_w_r / img_w) if img_w else focal_px
            focal_src = "cli"
        else:
            focal_px, focal_src = cfg.get_focal_px(img_w)
            focal_r, _ = cfg.get_focal_px(img_w_r)
        cfg.print_summary(focal_px, focal_src, behind_m=behind)
        print(
            "ALIGN: BOT→TOP far post on L, then R → Enter PLAY.\n"
            f"imgsz={yolo_imgsz}  replay={replay_s:.0f}s@{REPLAY_SCALE:.0%}  "
            "D=verbose dbg\n"
        )

        xyz_l, xyz_r = cfg.camera_positions(baseline, behind)
        pos_l = np.array(xyz_l)
        pos_r = np.array(xyz_r)

        def build_projs():
            return (
                make_proj_matrix(pos_l, h_left, v_angle, focal_px, img_w, img_h),
                make_proj_matrix(pos_r, h_right, v_angle, focal_r, img_w_r, img_h_r),
            )

        data = plane.load_plane()
        if data is not None:
            h_left = float(data.get("h_left_deg", data.get("h_angle_deg", h_left)))
            h_right = float(data.get("h_right_deg", -abs(h_left)))
            v_angle = float(data.get("v_angle_deg", v_angle))
            extend_up = float(data.get("extend_up_m", extend_up))
            behind = float(data.get("cam_behind_post_m", behind))
            xyz_l, xyz_r = cfg.camera_positions(baseline, behind)
            pos_l = np.array(xyz_l)
            pos_r = np.array(xyz_r)
            rd = int(data.get("rotate_deg", rotate_deg))
            if rd in (0, 90, 270):
                rotate_deg = rd
            clicks.load_dict(data.get("clicks", {}))
            sl = data.get("overlay_snap_left")
            sr = data.get("overlay_snap_right")
            snap_l = None if sl is None else np.array(sl, dtype=np.float64)
            snap_r = None if sr is None else np.array(sr, dtype=np.float64)
            if snap_l is not None and snap_r is not None:
                mode = Mode.PLAY
                print(f"  Loaded snaps → PLAY  ({plane.PLANE_PATH.name})")
            else:
                print(f"  Loaded {plane.PLANE_PATH.name} — finish Align")

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1400, 900)
        cv2.setMouseCallback(WINDOW, on_mouse)

        while True:
            if mode != Mode.REPLAY:
                ok_l, next_l = plane.read_cam(cap_l)
                ok_r, next_r = plane.read_cam(cap_r)
                if not ok_l or next_l is None or not ok_r or next_r is None:
                    time.sleep(0.02)
                    continue
                frame_l, frame_r = next_l, next_r
                raw_wh_l = (frame_l.shape[1], frame_l.shape[0])
                raw_wh_r = (frame_r.shape[1], frame_r.shape[0])
            else:
                if not ring:
                    mode = Mode.PLAY
                    print("  Replay empty → PLAY")
                    continue
                replay_idx = int(np.clip(replay_idx, 0, len(ring) - 1))
                small_l, small_r, replay_call, _ts = ring[replay_idx]
                frame_l = _grow_to(small_l, raw_wh_l)
                frame_r = _grow_to(small_r, raw_wh_r)

            view_l = plane.orient_frame(frame_l, rotate_deg)
            view_r = plane.orient_frame(frame_r, rotate_deg)
            img_w, img_h = view_l.shape[1], view_l.shape[0]
            img_w_r, img_h_r = view_r.shape[1], view_r.shape[0]
            if args.focal is not None:
                focal_px = float(args.focal)
                focal_r = focal_px * (img_w_r / img_w) if img_w else focal_px
            else:
                focal_px, _ = cfg.get_focal_px(img_w)
                focal_r, _ = cfg.get_focal_px(img_w_r)
            P1, P2 = build_projs()

            now = time.time()
            dt = max(now - last_t, 1e-6)
            last_t = now
            fps_ema = (1.0 / dt) if frame_i == 0 else (0.9 * fps_ema + 0.1 / dt)
            frame_i += 1

            disp_l = view_l.copy()
            disp_r = view_r.copy()
            _, y_vis, _ = cfg.lowest_visible_on_far_post(baseline, behind, v_angle)
            y_vis = max(0.0, float(y_vis))
            compact = mode != Mode.ALIGN
            plane.draw_goal_plane(
                disp_l, P1, baseline, extend_up, pos_l, pos_r, "L",
                show_point_labels=False, y_vis_min=y_vis, affine=snap_l, compact=compact,
            )
            plane.draw_goal_plane(
                disp_r, P2, baseline, extend_up, pos_l, pos_r, "R",
                show_point_labels=False, y_vis_min=y_vis, affine=snap_r, compact=compact,
            )
            if mode == Mode.ALIGN:
                clicks.draw_on(disp_l, disp_r)

            ball_line = ""
            ball_call = "none"
            if mode == Mode.REPLAY:
                ball_call = replay_call if replay_call in ("goal", "wide", "approaching", "wide_field") else "none"
            elif mode == Mode.PLAY and ball_on:
                if model is None:
                    model, ball_classes = plane.load_ball_model(
                        args.model or plane.default_ball_model(),
                    )
                mr_l = motion_l.process(view_l)
                mr_r = motion_r.process(view_r)
                has_motion = (not motion_gate) or bool(mr_l.blobs) or bool(mr_r.blobs)
                run_yolo, _ym = ball_sched.should_run(has_motion, now)
                if run_yolo:
                    ball_dets = plane.detect_both(
                        model, view_l, view_r, args.conf, ball_classes, imgsz=yolo_imgsz,
                    )
                    ball_sched.note_yolo(ball_dets, now)
                    xyz = plane.estimate_ball_xyz(
                        ball_dets[0], ball_dets[1], P1, P2, pos_l, pos_r,
                    )
                    ball_sched.note_track(xyz, now)
                else:
                    ball_dets = ball_sched.held_dets()
                    if ball_sched.last_mode == "idle":
                        ball_dets = (None, None)
                ball_line, ball_call = plane.paint_ball_pair(
                    disp_l, disp_r, ball_dets[0], ball_dets[1],
                    P1, P2, pos_l, pos_r, baseline, now, ball_latch,
                )
                if ball_call in ("goal", "wide"):
                    last_event_call = ball_call

            show_n = mode == Mode.ALIGN
            plane.draw_camera_hud(disp_l, "L", rotate_deg, show_north=show_n)
            plane.draw_camera_hud(disp_r, "R", rotate_deg, show_north=show_n)
            combo, panel_split_x = plane.side_by_side_full(disp_l, disp_r)

            color, banner, pulse = plane.call_style(ball_call)
            if banner and mode in (Mode.PLAY, Mode.REPLAY):
                plane.draw_flash_border(combo, color, now, pulse and mode == Mode.PLAY)

            badge_col = {
                Mode.ALIGN: (0, 220, 255),
                Mode.PLAY: (0, 255, 120),
                Mode.REPLAY: (255, 180, 80),
            }[mode]
            _mode_badge(combo, mode.name, badge_col)

            if mode == Mode.PLAY:
                ring.append((
                    _shrink(frame_l, REPLAY_SCALE),
                    _shrink(frame_r, REPLAY_SCALE),
                    ball_call,
                    now,
                ))
                max_n = max(30, int(round(replay_s * max(fps_ema, 15.0))))
                while len(ring) > max_n:
                    ring.popleft()

            aligned = snap_l is not None and snap_r is not None
            if mode == Mode.ALIGN:
                hint = (
                    f"L {len(clicks.far_l)}/2  R {len(clicks.far_r)}/2   "
                    f"snap {'L' if snap_l is not None else '·'}"
                    f"{'R' if snap_r is not None else '·'}   "
                    + ("Enter = PLAY" if aligned else "BOT→TOP far post each side")
                )
            elif mode == Mode.REPLAY:
                hint = (
                    f"{replay_idx + 1}/{len(ring)}   "
                    f"{(last_event_call or replay_call or '—').upper()}   "
                    f"←/→ scrub   Enter = PLAY"
                )
            else:
                short = plane.compact_ball_status(
                    ball_line, ball_call, ball_sched.last_mode if ball_on else "off",
                )
                hint = f"{short}   I replay   A align"

            lines = [
                f"{hint}   {fps_ema:.0f} fps",
            ]
            if verbose_dbg and ball_line and mode == Mode.PLAY:
                lines.append(ball_line)

            packed, bar_h = plane.attach_status_bar(
                combo, lines,
                accent_color=color if banner and mode != Mode.ALIGN else None,
                accent_line=lines[0] if banner else "",
            )
            win = window_client_size() or (1400, 900)
            canvas, x0, y0, nw, nh = plane.letterbox(packed, win[0], win[1])
            layout.update({
                "x0": x0, "y0": y0, "nw": nw, "nh": nh,
                "src_w": packed.shape[1], "src_h": packed.shape[0],
                "split_x": panel_split_x, "bar_h": bar_h, "pair_h": combo.shape[0],
            })
            cv2.imshow(WINDOW, canvas)

            if mouse["clicked"]:
                mouse["clicked"] = False
                if mode == Mode.ALIGN:
                    hit = _click_to_panel(mouse["x"], mouse["y"], layout, view_l, view_r)
                    if hit is not None:
                        side, xy = hit
                        n = clicks.add(side, xy)
                        if n >= 2:
                            snap_l, snap_r, _ = _try_snap(
                                side, clicks, P1, P2, baseline, extend_up, snap_l, snap_r,
                            )
                            if snap_l is not None and snap_r is not None:
                                print("  Both cams locked — press Enter for PLAY")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            # --- mode switches ---
            if mode == Mode.REPLAY and key in (ord("a"), ord("A"), 81, 2):  # A or left arrow
                step = max(1, int(round(fps_ema * 0.35)))
                replay_idx = max(0, replay_idx - step)
            elif mode == Mode.REPLAY and key in (ord("d"), ord("D"), 83, 3):  # D or right
                step = max(1, int(round(fps_ema * 0.35)))
                replay_idx = min(len(ring) - 1, replay_idx + step)
            elif key in (ord("a"), ord("A")):
                mode = Mode.ALIGN
                print("  → ALIGN")
            elif key in (13, ord("p"), ord("P")):
                if mode == Mode.ALIGN and not aligned:
                    print("  Lock BOTH cameras first (or L load plane.json)")
                else:
                    mode = Mode.PLAY
                    ball_sched.reset()
                    motion_l.reset()
                    motion_r.reset()
                    print("  → PLAY")
            elif key in (ord("i"), ord("I")):
                if mode == Mode.REPLAY:
                    mode = Mode.PLAY
                    print("  → PLAY")
                elif len(ring) > 0:
                    mode = Mode.REPLAY
                    replay_idx = len(ring) - 1
                    print(f"  → REPLAY  ({len(ring)} frames)")
                else:
                    print("  Replay empty — play first")
            elif key in (ord("b"), ord("B")) and mode == Mode.PLAY:
                ball_on = not ball_on
                ball_dets = (None, None)
                ball_sched.reset()
                motion_l.reset()
                motion_r.reset()
                print(f"  Ball {'ON' if ball_on else 'OFF'}")
            elif key in (ord("d"), ord("D")) and mode != Mode.REPLAY:
                verbose_dbg = not verbose_dbg
                print(f"  Verbose dbg {'ON' if verbose_dbg else 'OFF'}")
            elif key in (ord("c"), ord("C")) and mode == Mode.ALIGN:
                clicks.clear()
                print("  Clicks cleared")
            elif key in (ord("k"), ord("K")) and mode == Mode.ALIGN:
                if len(clicks.far_l) >= 2:
                    snap_l, snap_r, _ = _try_snap(
                        "L", clicks, P1, P2, baseline, extend_up, snap_l, snap_r,
                    )
                if len(clicks.far_r) >= 2:
                    snap_l, snap_r, _ = _try_snap(
                        "R", clicks, P1, P2, baseline, extend_up, snap_l, snap_r,
                    )
            elif key in (ord("o"), ord("O")):
                rotate_deg = {90: 270, 270: 0, 0: 90}.get(rotate_deg, 90)
                clicks.clear()
                snap_l = snap_r = None
                ball_dets = (None, None)
                ball_sched.reset()
                motion_l.reset()
                motion_r.reset()
                mode = Mode.ALIGN
                print(f"  {plane.rotate_label(rotate_deg)} → ALIGN")
            elif key in (ord("r"), ord("R")):
                h_left = float(cfg.H_ANGLE_DEG)
                h_right = -float(cfg.H_ANGLE_DEG)
                v_angle = float(cfg.V_ANGLE_DEG)
                snap_l = snap_r = None
                clicks.clear()
                mode = Mode.ALIGN
                print("  Reset → ALIGN")
            elif key in (ord("s"), ord("S")):
                plane.save_plane(
                    baseline, behind, abs(h_left), v_angle, extend_up,
                    focal_px, img_w, img_h, clicks, rotate_deg,
                    h_left=h_left, h_right=h_right,
                    v_left=v_angle, v_right=v_angle,
                    snap_l=snap_l, snap_r=snap_r,
                )
            elif key in (ord("l"), ord("L")):
                data = plane.load_plane()
                if data is None:
                    print("  No plane.json")
                else:
                    h_left = float(data.get("h_left_deg", data.get("h_angle_deg", h_left)))
                    h_right = float(data.get("h_right_deg", -abs(h_left)))
                    v_angle = float(data.get("v_angle_deg", v_angle))
                    extend_up = float(data.get("extend_up_m", extend_up))
                    behind = float(data.get("cam_behind_post_m", behind))
                    xyz_l, xyz_r = cfg.camera_positions(baseline, behind)
                    pos_l = np.array(xyz_l)
                    pos_r = np.array(xyz_r)
                    rd = int(data.get("rotate_deg", rotate_deg))
                    if rd in (0, 90, 270):
                        rotate_deg = rd
                    clicks.load_dict(data.get("clicks", {}))
                    sl = data.get("overlay_snap_left")
                    sr = data.get("overlay_snap_right")
                    snap_l = None if sl is None else np.array(sl, dtype=np.float64)
                    snap_r = None if sr is None else np.array(sr, dtype=np.float64)
                    print(f"  Loaded {plane.PLANE_PATH.name}")

    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    finally:
        if cap_l is not None:
            cap_l.release()
        if cap_r is not None:
            cap_r.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
