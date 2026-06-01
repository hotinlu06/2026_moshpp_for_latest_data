import os
import cv2
import json
import pickle
import argparse
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Suitable for server / headless environment
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ==============================================================
# 1. SMPL 24 joints definition
# ==============================================================

SMPL_24_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]

SMPL_24_CONN = [
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),                  # Spine
    (0, 1), (1, 4), (4, 7), (7, 10),                            # Left leg
    (0, 2), (2, 5), (5, 8), (8, 11),                            # Right leg
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),             # Left arm
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23)              # Right arm
]

# OpenCV uses BGR
COLOR_SPINE_2D = (0, 255, 0)     # Green
COLOR_LEFT_2D = (0, 0, 255)      # Red
COLOR_RIGHT_2D = (255, 0, 0)     # Blue

BONE_COLORS_2D = (
    [COLOR_SPINE_2D] * 5 +
    [COLOR_LEFT_2D] * 4 +
    [COLOR_RIGHT_2D] * 4 +
    [COLOR_LEFT_2D] * 5 +
    [COLOR_RIGHT_2D] * 5
)

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]


# ==============================================================
# 2. Utility functions
# ==============================================================

def load_motion_pkl(pkl_file):
    print(f"📦 Loading PKL: {pkl_file}")

    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    data = np.asarray(data)

    if not (data.ndim == 3 and data.shape[1] == 24 and data.shape[2] == 3):
        raise ValueError(f"Expected shape (*, 24, 3), got {data.shape}")

    nan_count = np.isnan(data).sum()
    if nan_count > 0:
        raise ValueError(f"PKL contains NaN values: {nan_count}")

    print(f"✅ Motion shape: {data.shape}")
    return data


def infer_camera_view(pkl_file, video_file=None, camera_view="auto"):
    if camera_view != "auto":
        return camera_view

    names = [os.path.basename(pkl_file)]

    if video_file is not None:
        names.append(os.path.basename(video_file))

    joined = " ".join(names)

    if "_L_" in joined or "_L." in joined:
        return "left"

    if "_C_" in joined or "_C." in joined:
        return "middle"

    if "_R_" in joined or "_R." in joined:
        return "right"

    print("⚠️ Could not infer camera view from filename. Defaulting to middle.")
    return "middle"


def load_camera_params(camera_param_dir, view):
    extrinsic_file = os.path.join(camera_param_dir, f"extrinsics_{view}.json")
    intrinsic_file = os.path.join(camera_param_dir, f"intrinsic_{view}.json")

    if not os.path.exists(extrinsic_file):
        raise FileNotFoundError(f"Extrinsic file not found: {extrinsic_file}")

    if not os.path.exists(intrinsic_file):
        raise FileNotFoundError(f"Intrinsic file not found: {intrinsic_file}")

    print(f"   -> Reading extrinsic: {extrinsic_file}")
    print(f"   -> Reading intrinsic: {intrinsic_file}")

    with open(extrinsic_file, "r") as f:
        extr = json.load(f)
        extrinsic = np.array(extr["best_extrinsic"], dtype=np.float32)
        R = extrinsic[:, :3]
        t = extrinsic[:, 3]
        rvec, _ = cv2.Rodrigues(R)

    with open(intrinsic_file, "r") as f:
        intr = json.load(f)
        camera_matrix = np.array(intr["camera_matrix"], dtype=np.float32)
        dist_coeffs = np.array(intr["dist_coeffs"], dtype=np.float32).flatten()

    return camera_matrix, dist_coeffs, rvec, t


def check_output_path(path, overwrite=False):
    parent = os.path.dirname(path)

    if parent:
        os.makedirs(parent, exist_ok=True)

    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"Output already exists and overwrite=False:\n{path}\n"
            f"Use --overwrite if you want to replace it."
        )


def draw_skeleton_2d(frame, points_2d):
    h, w = frame.shape[:2]

    for idx, (start, end) in enumerate(SMPL_24_CONN):
        pt1 = points_2d[start]
        pt2 = points_2d[end]

        if not (np.all(np.isfinite(pt1)) and np.all(np.isfinite(pt2))):
            continue

        pt1 = tuple(pt1.astype(int))
        pt2 = tuple(pt2.astype(int))

        if (
            0 <= pt1[0] < w and 0 <= pt1[1] < h and
            0 <= pt2[0] < w and 0 <= pt2[1] < h
        ):
            cv2.line(
                frame,
                pt1,
                pt2,
                BONE_COLORS_2D[idx],
                thickness=3,
                lineType=cv2.LINE_AA
            )

    for pt in points_2d:
        if not np.all(np.isfinite(pt)):
            continue

        pt = tuple(pt.astype(int))

        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(frame, pt, radius=4, color=(0, 255, 255), thickness=-1)

    return frame


def index_video_folder(video_root):
    """
    Recursively index all videos under video_root by basename without extension.

    Example:
        /videos/01_boss_C/01_boss_C_raising-hand_row18_rep4.mp4
    becomes:
        key = 01_boss_C_raising-hand_row18_rep4
    """
    video_root = Path(video_root)
    video_map = {}

    for ext in VIDEO_EXTENSIONS:
        for video_path in video_root.rglob(f"*{ext}"):
            key = video_path.stem

            if key in video_map:
                print(
                    f"⚠️ Duplicate video basename found: {key}\n"
                    f"   Existing: {video_map[key]}\n"
                    f"   Ignored:   {video_path}"
                )
                continue

            video_map[key] = str(video_path)

    print(f"🎞️ Indexed {len(video_map)} video files from: {video_root}")
    return video_map


def find_matching_video_for_pkl(pkl_file, video_map):
    base_name = Path(pkl_file).stem
    return video_map.get(base_name, None)


def build_output_subdir_for_pkl(pkl_file, input_root, output_root):
    """
    In folder mode, preserve the relative folder structure and create one folder per pkl.

    Example:
        input_root:
            /data/smpl_24_sliced_Boss_test_Actor01_0528

        pkl:
            /data/smpl_24_sliced_Boss_test_Actor01_0528/01_boss_C_3d/xxx.pkl

        output:
            /output_root/01_boss_C_3d/xxx/
    """
    pkl_path = Path(pkl_file)
    input_root = Path(input_root)
    output_root = Path(output_root)

    try:
        relative_parent = pkl_path.parent.relative_to(input_root)
    except ValueError:
        relative_parent = Path()

    return output_root / relative_parent / pkl_path.stem


# ==============================================================
# 3. 2D overlay video
# ==============================================================

def render_2d_overlay(
    motion,
    pkl_file,
    video_file,
    camera_param_dir,
    output_video,
    video_start_frame=0,
    camera_view="auto",
    overwrite=False
):
    print("\n===================================================")
    print("🎬 Rendering 2D overlay video")
    print("===================================================")

    view = infer_camera_view(pkl_file, video_file, camera_view)
    print(f"📷 Camera view: {view}")

    cam_mtx, dist_coef, rvec, tvec = load_camera_params(camera_param_dir, view)

    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Video file not found: {video_file}")

    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 0:
        print("⚠️ Video FPS is invalid. Using 30 FPS.")
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)

    check_output_path(output_video, overwrite=overwrite)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        raise ValueError(f"Cannot create output video: {output_video}")

    num_frames = motion.shape[0]
    rendered = 0

    print(f"Input video: {video_file}")
    print(f"Output video: {output_video}")
    print(f"Frames to render: {num_frames}")

    for i in range(num_frames):
        ret, frame = cap.read()

        if not ret:
            print(f"⚠️ Video ended early at frame {i}.")
            break

        points_3d = motion[i].astype(np.float32)

        projected_2d, _ = cv2.projectPoints(
            points_3d,
            rvec,
            tvec,
            cam_mtx,
            dist_coef
        )

        points_2d = projected_2d.squeeze()
        frame = draw_skeleton_2d(frame, points_2d)

        text = f"SMPL 24 | Frame {i + 1}/{num_frames} | View: {view.upper()}"

        cv2.putText(
            frame,
            text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3
        )

        cv2.putText(
            frame,
            text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        out.write(frame)
        rendered += 1

        if i > 0 and i % 100 == 0:
            print(f"   Rendered {i}/{num_frames} frames...")

    cap.release()
    out.release()

    print(f"✅ 2D overlay saved: {output_video}")
    print(f"Rendered frames: {rendered}")


# ==============================================================
# 4. 3D GIF visualization
# ==============================================================

def render_3d_gif(
    motion,
    pkl_file,
    output_gif,
    view_name,
    elev,
    azim,
    invert_x=False,
    use_ortho=False,
    max_frames=300,
    fps=30,
    overwrite=False
):
    print("\n===================================================")
    print(f"🧍 Rendering 3D GIF: {view_name}")
    print("===================================================")

    check_output_path(output_gif, overwrite=overwrite)

    if max_frames is not None and max_frames > 0:
        motion_vis = motion[:max_frames]
    else:
        motion_vis = motion

    base_name = Path(pkl_file).stem

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    if use_ortho:
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass

    lines = []

    for start, end in SMPL_24_CONN:
        end_joint_name = SMPL_24_JOINT_NAMES[end]

        if end_joint_name.startswith("L_"):
            color = "red"
        elif end_joint_name.startswith("R_"):
            color = "blue"
        else:
            color = "gray"

        line = ax.plot([], [], [], color=color, lw=2.5)[0]
        lines.append(line)

    points = ax.scatter([], [], [], c="black", s=18, zorder=10)

    # Data coordinate: [X, Y, Z]
    # Plot coordinate:
    #   plot X = data X
    #   plot Y = data Z / depth
    #   plot Z = data Y / height
    x_min, x_max = np.min(motion_vis[:, :, 0]), np.max(motion_vis[:, :, 0])
    y_min, y_max = np.min(motion_vis[:, :, 2]), np.max(motion_vis[:, :, 2])
    z_min, z_max = np.min(motion_vis[:, :, 1]), np.max(motion_vis[:, :, 1])

    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2

    max_range = max(
        x_max - x_min,
        y_max - y_min,
        z_max - z_min
    )

    if max_range <= 0:
        max_range = 1.0

    radius = max_range / 2.0 * 1.15

    ax.set_xlim([x_center - radius, x_center + radius])
    ax.set_ylim([y_center - radius, y_center + radius])
    ax.set_zlim([z_center - radius, z_center + radius])

    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass

    ax.view_init(elev=elev, azim=azim)

    if invert_x:
        ax.invert_xaxis()

    ax.set_xlabel("X")
    ax.set_ylabel("Z / Depth")
    ax.set_zlabel("Y / Height")

    def update(frame):
        for i, (start, end) in enumerate(SMPL_24_CONN):
            x = [
                motion_vis[frame, start, 0],
                motion_vis[frame, end, 0]
            ]

            y = [
                motion_vis[frame, start, 2],
                motion_vis[frame, end, 2]
            ]

            z = [
                motion_vis[frame, start, 1],
                motion_vis[frame, end, 1]
            ]

            lines[i].set_data(x, y)
            lines[i].set_3d_properties(z)

        points._offsets3d = (
            motion_vis[frame, :, 0],
            motion_vis[frame, :, 2],
            motion_vis[frame, :, 1]
        )

        ax.set_title(
            f"SMPL 24 Motion Visualization - {view_name}\n"
            f"Left: Red | Right: Blue | Frame {frame + 1}/{len(motion_vis)}\n"
            f"{base_name}",
            fontsize=10
        )

        return lines + [points]

    print(f"Output GIF: {output_gif}")
    print(f"Frames to render: {len(motion_vis)}")
    print(f"View angle: elev={elev}, azim={azim}, invert_x={invert_x}")

    ani = FuncAnimation(
        fig,
        update,
        frames=len(motion_vis),
        interval=1000 / fps,
        blit=False
    )

    ani.save(output_gif, writer="pillow", fps=fps)
    plt.close(fig)

    print(f"✅ 3D GIF saved: {output_gif}")


# ==============================================================
# 5. Process one PKL
# ==============================================================

def process_one_pkl(
    pkl_file,
    video_file,
    camera_dir,
    output_dir,
    camera_view="auto",
    video_start_frame=0,
    max_3d_frames=300,
    fps_3d=30,
    side_elev=15,
    side_azim=-160,
    side_invert_x=True,
    three_elev=10,
    three_azim=-125,
    three_invert_x=True,
    front_elev=0,
    front_azim=-90,
    front_invert_x=False,
    overwrite=False
):
    pkl_file = str(pkl_file)
    video_file = str(video_file)
    output_dir = str(output_dir)

    print("\n\n###################################################")
    print(f"Processing PKL: {pkl_file}")
    print(f"Matched video:  {video_file}")
    print(f"Output dir:     {output_dir}")
    print("###################################################")

    os.makedirs(output_dir, exist_ok=True)

    base_name = Path(pkl_file).stem

    output_2d = os.path.join(output_dir, f"{base_name}_2d_overlay.mp4")
    output_3d_side = os.path.join(output_dir, f"{base_name}_3d_side.gif")
    output_3d_three = os.path.join(output_dir, f"{base_name}_3d_three_quarter.gif")
    output_3d_front = os.path.join(output_dir, f"{base_name}_3d_front.gif")

    motion = load_motion_pkl(pkl_file)

    render_2d_overlay(
        motion=motion,
        pkl_file=pkl_file,
        video_file=video_file,
        camera_param_dir=camera_dir,
        output_video=output_2d,
        video_start_frame=video_start_frame,
        camera_view=camera_view,
        overwrite=overwrite
    )

    render_3d_gif(
        motion=motion,
        pkl_file=pkl_file,
        output_gif=output_3d_side,
        view_name="Side View",
        elev=side_elev,
        azim=side_azim,
        invert_x=side_invert_x,
        use_ortho=False,
        max_frames=max_3d_frames,
        fps=fps_3d,
        overwrite=overwrite
    )

    render_3d_gif(
        motion=motion,
        pkl_file=pkl_file,
        output_gif=output_3d_three,
        view_name="Three-quarter View",
        elev=three_elev,
        azim=three_azim,
        invert_x=three_invert_x,
        use_ortho=False,
        max_frames=max_3d_frames,
        fps=fps_3d,
        overwrite=overwrite
    )

    render_3d_gif(
        motion=motion,
        pkl_file=pkl_file,
        output_gif=output_3d_front,
        view_name="Front View",
        elev=front_elev,
        azim=front_azim,
        invert_x=front_invert_x,
        use_ortho=True,
        max_frames=max_3d_frames,
        fps=fps_3d,
        overwrite=overwrite
    )

    print("\n✅ Completed one PKL:")
    print(f"2D overlay:       {output_2d}")
    print(f"3D side:          {output_3d_side}")
    print(f"3D three-quarter: {output_3d_three}")
    print(f"3D front:         {output_3d_front}")


# ==============================================================
# 6. Arguments
# ==============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate 2D overlay video and 3D SMPL-24 visualizations.\n"
            "Supports single PKL mode and folder recursive mode."
        )
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["single", "folder"],
        help="Processing mode: single or folder."
    )

    # Single mode
    parser.add_argument(
        "--pkl",
        default=None,
        help="Single mode: input action PKL path."
    )

    parser.add_argument(
        "--video",
        default=None,
        help="Single mode: corresponding video path."
    )

    # Folder mode
    parser.add_argument(
        "--pkl_dir",
        default=None,
        help="Folder mode: root folder containing PKL files, recursively."
    )

    parser.add_argument(
        "--video_dir",
        default=None,
        help="Folder mode: root folder containing videos, recursively. Videos are matched by basename."
    )

    # Shared
    parser.add_argument(
        "--camera_dir",
        required=True,
        help="Camera parameter directory containing intrinsic_xxx.json and extrinsics_xxx.json."
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output root directory."
    )

    parser.add_argument(
        "--camera_view",
        default="auto",
        choices=["auto", "middle", "left", "right"],
        help="Camera view. Default auto: infer from filename _C_, _L_, or _R_."
    )

    parser.add_argument(
        "--video_start_frame",
        type=int,
        default=0,
        help="Start frame in the video. Default: 0."
    )

    parser.add_argument(
        "--max_3d_frames",
        type=int,
        default=300,
        help="Maximum frames for 3D GIF. Use 0 or negative to render all frames."
    )

    parser.add_argument(
        "--fps_3d",
        type=int,
        default=30,
        help="FPS for 3D GIF. Default: 30."
    )

    parser.add_argument("--side_elev", type=float, default=15)
    parser.add_argument("--side_azim", type=float, default=-160)
    parser.add_argument("--no_side_invert_x", action="store_true")

    parser.add_argument("--three_elev", type=float, default=10)
    parser.add_argument("--three_azim", type=float, default=-125)
    parser.add_argument("--no_three_invert_x", action="store_true")

    parser.add_argument("--front_elev", type=float, default=0)
    parser.add_argument(
        "--front_azim",
        type=float,
        default=-90,
        help="If front view looks like back view, try --front_azim 90."
    )
    parser.add_argument("--front_invert_x", action="store_true")

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files."
    )

    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Folder mode: continue processing remaining PKLs if one file fails."
    )

    return parser.parse_args()


# ==============================================================
# 7. Main
# ==============================================================

def run_single_mode(args):
    if args.pkl is None:
        raise ValueError("single mode requires --pkl")

    if args.video is None:
        raise ValueError("single mode requires --video")

    if not os.path.exists(args.pkl):
        raise FileNotFoundError(f"Input PKL not found: {args.pkl}")

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Input video not found: {args.video}")

    if not os.path.isdir(args.camera_dir):
        raise FileNotFoundError(f"Camera dir not found: {args.camera_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    process_one_pkl(
        pkl_file=args.pkl,
        video_file=args.video,
        camera_dir=args.camera_dir,
        output_dir=args.output_dir,
        camera_view=args.camera_view,
        video_start_frame=args.video_start_frame,
        max_3d_frames=args.max_3d_frames,
        fps_3d=args.fps_3d,
        side_elev=args.side_elev,
        side_azim=args.side_azim,
        side_invert_x=not args.no_side_invert_x,
        three_elev=args.three_elev,
        three_azim=args.three_azim,
        three_invert_x=not args.no_three_invert_x,
        front_elev=args.front_elev,
        front_azim=args.front_azim,
        front_invert_x=args.front_invert_x,
        overwrite=args.overwrite
    )


def run_folder_mode(args):
    if args.pkl_dir is None:
        raise ValueError("folder mode requires --pkl_dir")

    if args.video_dir is None:
        raise ValueError("folder mode requires --video_dir")

    if not os.path.isdir(args.pkl_dir):
        raise FileNotFoundError(f"PKL dir not found: {args.pkl_dir}")

    if not os.path.isdir(args.video_dir):
        raise FileNotFoundError(f"Video dir not found: {args.video_dir}")

    if not os.path.isdir(args.camera_dir):
        raise FileNotFoundError(f"Camera dir not found: {args.camera_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    pkl_root = Path(args.pkl_dir)
    output_root = Path(args.output_dir)

    pkl_files = sorted(pkl_root.rglob("*.pkl"))

    print("===================================================")
    print("Folder mode")
    print("===================================================")
    print(f"PKL root:     {pkl_root}")
    print(f"Video root:   {args.video_dir}")
    print(f"Output root:  {output_root}")
    print(f"PKL files found: {len(pkl_files)}")
    print("===================================================")

    if len(pkl_files) == 0:
        print("⚠️ No PKL files found. Nothing to process.")
        return

    video_map = index_video_folder(args.video_dir)

    processed = 0
    skipped_no_video = 0
    failed = 0

    for idx, pkl_file in enumerate(pkl_files, start=1):
        print("\n\n===================================================")
        print(f"[{idx}/{len(pkl_files)}] Checking PKL:")
        print(pkl_file)
        print("===================================================")

        video_file = find_matching_video_for_pkl(pkl_file, video_map)

        if video_file is None:
            print(f"⚠️ No matching video found for PKL: {pkl_file}")
            print(f"   Expected video basename: {pkl_file.stem}")
            skipped_no_video += 1
            continue

        output_subdir = build_output_subdir_for_pkl(
            pkl_file=pkl_file,
            input_root=pkl_root,
            output_root=output_root
        )

        try:
            process_one_pkl(
                pkl_file=str(pkl_file),
                video_file=video_file,
                camera_dir=args.camera_dir,
                output_dir=str(output_subdir),
                camera_view=args.camera_view,
                video_start_frame=args.video_start_frame,
                max_3d_frames=args.max_3d_frames,
                fps_3d=args.fps_3d,
                side_elev=args.side_elev,
                side_azim=args.side_azim,
                side_invert_x=not args.no_side_invert_x,
                three_elev=args.three_elev,
                three_azim=args.three_azim,
                three_invert_x=not args.no_three_invert_x,
                front_elev=args.front_elev,
                front_azim=args.front_azim,
                front_invert_x=args.front_invert_x,
                overwrite=args.overwrite
            )
            processed += 1

        except Exception as e:
            failed += 1
            print(f"❌ Failed to process: {pkl_file}")
            print(f"   Error: {repr(e)}")

            if not args.continue_on_error:
                raise

            print("   continue_on_error=True, moving to next PKL...")

    print("\n\n===================================================")
    print("Folder mode summary")
    print("===================================================")
    print(f"Total PKL files:       {len(pkl_files)}")
    print(f"Processed:             {processed}")
    print(f"Skipped no video:      {skipped_no_video}")
    print(f"Failed:                {failed}")
    print(f"Output root:           {output_root}")
    print("===================================================")


def main():
    args = parse_args()

    if args.mode == "single":
        run_single_mode(args)
    elif args.mode == "folder":
        run_folder_mode(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()