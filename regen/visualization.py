from importlib import import_module
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
plt = import_module("matplotlib.pyplot")


def add_label(image, text, label_height=34):
    labeled = Image.new("RGB", (image.width, image.height + label_height), "white")
    labeled.paste(image.convert("RGB"), (0, label_height))
    draw = ImageDraw.Draw(labeled)
    draw.text((8, 10), text, fill="black")
    return labeled


def render_recovery_3d(
    current,
    predicted_damage,
    added_mask=None,
    original_shape=None,
    size=(12, 4),
    render_edges=False,
):
    predicted_mask = (predicted_damage > 0) & (current > 0)
    if added_mask is None:
        added_mask = np.zeros_like(current, dtype=bool)
    else:
        added_mask = added_mask > 0

    visible = (current > 0) | added_mask
    facecolors = np.zeros(visible.shape + (4,), dtype=np.float32)
    facecolors[current > 0] = [0.0, 0.05, 0.85, 1.0]
    facecolors[predicted_mask] = [1.0, 0.72, 0.0, 1.0]
    facecolors[added_mask] = [0.0, 0.72, 0.25, 1.0]

    if original_shape is None:
        original_shape = current.shape

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection="3d")
    voxel_kwargs = {"facecolors": facecolors}
    if render_edges:
        voxel_kwargs.update({"edgecolor": "k", "linewidth": 0.04})
    else:
        voxel_kwargs.update({"edgecolor": "none", "linewidth": 0.0})
    ax.voxels(visible, **voxel_kwargs)
    ax.set_xlim(0, original_shape[0])
    ax.set_ylim(0, original_shape[1])
    ax.set_zlim(0, original_shape[2])
    ax.set_box_aspect(original_shape)
    ax.view_init(elev=24, azim=-58)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.patch.set_facecolor("white")
    plt.tight_layout(pad=0)

    buf = BytesIO()
    plt.savefig(
        buf,
        format="png",
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_recovery_gif(
    trajectory,
    output_path,
    frame_duration=250,
    size=(12, 4),
    show_progress=True,
    frame_stride=1,
    render_edges=False,
    skip_noop_prefix=True,
):
    frames = []
    sampled_steps = sample_recovery_steps(
        trajectory.steps,
        frame_stride,
        skip_noop_prefix=skip_noop_prefix,
    )
    steps = recovery_frame_iterator(sampled_steps, show_progress)
    for step_idx, step in steps:
        predicted_count = int(((step.predicted_damage > 0) & (step.voxels > 0)).sum())
        added_count = int((step.added_mask > 0).sum())
        missing_text = "unknown"
        if step.missing_count is not None:
            missing_text = str(step.missing_count)
        extra_text = "unknown"
        if step.extra_count is not None:
            extra_text = str(step.extra_count)

        frame = render_recovery_3d(
            current=step.voxels,
            predicted_damage=step.predicted_damage,
            added_mask=step.added_mask,
            original_shape=step.voxels.shape,
            size=size,
            render_edges=render_edges,
        )
        frames.append(
            add_label(
                frame,
                (
                    f"step {step_idx:02d} | missing={missing_text} | "
                    f"extra={extra_text} | predicted={predicted_count} | "
                    f"added_last={added_count} | added_total={step.total_added_count}"
                ),
            )
        )

    if not frames:
        raise ValueError("Cannot save recovery GIF with no frames.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
    )
    return output_path


def sample_recovery_steps(steps, frame_stride, skip_noop_prefix=True):
    frame_stride = max(1, int(frame_stride))
    indexed_steps = list(enumerate(steps))
    if skip_noop_prefix:
        original_steps = indexed_steps
        indexed_steps = skip_initial_noop_steps(indexed_steps)
        skipped_prefix = (
            len(indexed_steps) > 1
            and len(original_steps) > 1
            and indexed_steps[1][0] != original_steps[1][0]
        )
    else:
        skipped_prefix = False
    if skipped_prefix:
        sampled_steps = [indexed_steps[0]] + indexed_steps[1::frame_stride]
    else:
        sampled_steps = indexed_steps[::frame_stride]
    if indexed_steps and sampled_steps[-1][0] != indexed_steps[-1][0]:
        sampled_steps.append(indexed_steps[-1])
    return sampled_steps


def skip_initial_noop_steps(indexed_steps):
    if len(indexed_steps) <= 2:
        return indexed_steps

    first_active_position = None
    for position, (_, step) in enumerate(indexed_steps[1:], start=1):
        if int((step.added_mask > 0).sum()) > 0:
            first_active_position = position
            break

    if first_active_position is None or first_active_position <= 1:
        return indexed_steps
    return [indexed_steps[0]] + indexed_steps[first_active_position:]


def recovery_frame_iterator(steps, show_progress):
    if not show_progress:
        return steps

    from tqdm import tqdm

    return tqdm(steps, desc="Rendering recovery GIF", unit="frame")
