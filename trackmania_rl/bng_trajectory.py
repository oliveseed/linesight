"""Virtual-checkpoint handling for BeamNG trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class VirtualCheckpoints:
    """A resampled trajectory with enough points before and after the track."""

    centers: FloatArray
    finish_idx: int
    zone_transitions: FloatArray
    distance_between_zone_transitions: FloatArray
    distance_from_start_track_to_prev_zone_transition: FloatArray
    normalized_vector_along_track_axis: FloatArray


def _validate_trajectory(trajectory: npt.ArrayLike) -> FloatArray:
    points = np.asarray(trajectory, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Trajectory must have shape (n, 3), got {points.shape!r}.")
    if len(points) < 2:
        raise ValueError("Trajectory must contain at least two distinct points.")
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]
    points = points[keep]
    if len(points) < 2:
        raise ValueError("Trajectory needs at least two distinct points.")
    return points


def generate_virtual_checkpoints(trajectory: npt.ArrayLike, spacing: float) -> FloatArray:
    """Resample a recorded trajectory at approximately equally spaced VCPs.

    The recorded end point is always preserved.  This lets the last real VCP be
    the finish, independent of whether the source sampling interval divides the
    requested spacing exactly.
    """

    if spacing <= 0:
        raise ValueError("Virtual-checkpoint spacing must be positive.")
    points = _validate_trajectory(trajectory)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    sample_distances = np.arange(0.0, cumulative[-1], spacing, dtype=np.float32)
    if sample_distances.size == 0 or cumulative[-1] - sample_distances[-1] > 1e-5:
        sample_distances = np.append(sample_distances, cumulative[-1])

    result = np.empty((len(sample_distances), 3), dtype=np.float32)
    for axis in range(3):
        result[:, axis] = np.interp(sample_distances, cumulative, points[:, axis])
    return result


def load_virtual_checkpoints(
    trajectory_path: Path | str | None = None,
    spacing: float | None = None,
    n_before: int | None = None,
    n_after: int | None = None,
) -> VirtualCheckpoints:
    """Load a ``.npy`` player trajectory, resample it, and extrapolate its ends."""
    from config_files import config_copy

    if trajectory_path is None:
        trajectory_path = Path(__file__).resolve().parents[1] / "path.npy"
    trajectory_path = Path(trajectory_path)
    if spacing is None:
        spacing = float(config_copy.distance_between_checkpoints)
    if n_before is None:
        n_before = int(config_copy.n_zone_centers_extrapolate_before_start_of_map)
    if n_after is None:
        n_after = int(config_copy.n_zone_centers_extrapolate_after_end_of_map)

    raw = np.load(str(trajectory_path))
    centers = generate_virtual_checkpoints(raw, spacing)
    # Extrapolate beyond start and finish so the model always has next-N VCPs.
    before = centers[0] + np.expand_dims(centers[0] - centers[1], axis=0) * np.expand_dims(
        np.arange(n_before, 0, -1, dtype=np.float32), axis=1
    )
    after = centers[-1] + np.expand_dims(centers[-1] - centers[-2], axis=0) * np.expand_dims(
        np.arange(1, 1 + n_after, dtype=np.float32), axis=1
    )
    full = np.vstack((before, centers, after)).astype(np.float32)
    # Light smoothing matching map_loader behavior, keeping ends stable.
    if len(full) > 10:
        full[5:-5] = 0.5 * (full[:-10] + full[10:])
    finish_idx = n_before + len(centers) - 1
    return make_virtual_checkpoints(full, finish_idx)


def get_vcp_centers_from_file(trajectory_path: Path | str | None = None, config = None) -> VirtualCheckpoints:
    """Load a ``.npy`` player trajectory, resample it, and extrapolate its ends."""

    if trajectory_path is None:
        trajectory_path = Path(__file__).resolve().parents[1] / "nord1.npy"

    required_after_finish = (config.n_zone_centers_in_inputs - 1) * config.one_every_n_zone_centers_in_inputs
    if config.n_zone_centers_extrapolate_after_end_of_map < required_after_finish:
        raise ValueError(
            "Not enough post-finish VCP extrapolation for the requested model input: "
            f"need {required_after_finish}, got {config.n_zone_centers_extrapolate_after_end_of_map}."
        )
    trajectory = np.load(Path(trajectory_path), allow_pickle=False)
    real_centers = generate_virtual_checkpoints(trajectory, config.distance_between_checkpoints)
    start_direction = real_centers[1] - real_centers[0]
    end_direction = real_centers[-1] - real_centers[-2]
    start_direction /= np.linalg.norm(start_direction)
    end_direction /= np.linalg.norm(end_direction)

    before = real_centers[0] - np.arange(
        config.n_zone_centers_extrapolate_before_start_of_map, 0, -1, dtype=np.float32
    )[:, None] * config.distance_between_checkpoints * start_direction
    after = real_centers[-1] + np.arange(
        1, config.n_zone_centers_extrapolate_after_end_of_map + 1, dtype=np.float32
    )[:, None] * config.distance_between_checkpoints * end_direction
    centers = np.vstack((before, real_centers, after)).astype(np.float32)
    finish_idx = config.n_zone_centers_extrapolate_before_start_of_map + len(real_centers) - 1
    return centers


def make_virtual_checkpoints(centers: npt.ArrayLike, finish_idx: int) -> VirtualCheckpoints:
    """Precalculate the same segment information used by Linesight rollouts."""

    centers = _validate_trajectory(centers)
    if not 1 <= finish_idx < len(centers) - 1:
        raise ValueError("finish_idx must leave at least one extrapolated VCP after the finish.")
    zone_transitions = (0.5 * (centers[1:] + centers[:-1])).astype(np.float32)
    deltas = zone_transitions[1:] - zone_transitions[:-1]
    segment_lengths = np.linalg.norm(deltas, axis=1).astype(np.float32)
    if np.any(segment_lengths <= 1e-6):
        raise ValueError("Virtual checkpoints must not create zero-length segments.")
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)].astype(np.float32)
    return VirtualCheckpoints(
        centers=centers,
        finish_idx=finish_idx,
        zone_transitions=zone_transitions,
        distance_between_zone_transitions=segment_lengths,
        distance_from_start_track_to_prev_zone_transition=cumulative,
        normalized_vector_along_track_axis=(deltas / segment_lengths[:, None]).astype(np.float32),
    )


def update_current_zone_idx(
    current_zone_idx: int,
    zone_centers: npt.ArrayLike,
    position: npt.ArrayLike,
    max_allowable_distance_to_virtual_checkpoint: float,
    first_zone_idx: int,
    finish_idx: int,
) -> int:
    """Advance or retreat through VCPs only while the vehicle remains on track.

    Progress advances when the next VCP is both closer than the current one and
    within the road-width-derived tolerance.  Retreating uses the symmetric
    rule, preventing an off-track teleport from fabricating progress.
    """

    centers = np.asarray(zone_centers, dtype=np.float32)
    position = np.asarray(position, dtype=np.float32)
    current = int(np.clip(current_zone_idx, first_zone_idx, finish_idx))

    while current < finish_idx:
        current_distance = float(np.linalg.norm(centers[current] - position))
        next_distance = float(np.linalg.norm(centers[current + 1] - position))
        if next_distance > current_distance or next_distance > max_allowable_distance_to_virtual_checkpoint:
            break
        current += 1

    while current > first_zone_idx:
        current_distance = float(np.linalg.norm(centers[current] - position))
        previous_distance = float(np.linalg.norm(centers[current - 1] - position))
        if previous_distance >= current_distance or previous_distance > max_allowable_distance_to_virtual_checkpoint:
            break
        current -= 1
    return current
