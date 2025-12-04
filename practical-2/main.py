from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


class BoundingBox(object):
    def __init__(self, bb_left, bb_top, bb_width, bb_height):
        self.bb_left: int = bb_left
        self.bb_top: int = bb_top
        self.bb_width: int = bb_width
        self.bb_height: int = bb_height

        self.top_left: tuple[int, int] = (bb_left, bb_top)
        self.top_right: tuple[int, int] = (bb_left + bb_width, bb_top)
        self.bot_left: tuple[int, int] = (bb_left, bb_top + bb_height)
        self.bot_right: tuple[int, int] = (bb_left + bb_width, bb_top + bb_height)

        self.area: float = float(bb_width * bb_height)

    @property
    def right(self) -> int:
        return self.bb_left + self.bb_width

    @property
    def bottom(self) -> int:
        return self.bb_top + self.bb_height


class Detection(object):
    def __init__(self, frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z):
        self.frame: int = frame
        self.id: int = id
        self.bb: BoundingBox = BoundingBox(bb_left, bb_top, bb_width, bb_height)
        self.conf: float = conf
        self.x: int = x
        self.y: int = y
        self.z: int = z

    def __str__(self):
        return f"""Detection(frame={self.frame},
          id={self.id},
          conf={self.conf:.2f}, 
          x={self.x},
          y={self.y},
          z={self.z})"""

    def to_gt_line(self) -> str:
        """Formats the detection object into a GT-compatible CSV line."""
        return f"{self.frame},{self.id},{self.bb.bb_left},{self.bb.bb_top},{self.bb.bb_width},{self.bb.bb_height},1,-1,-1,-1"


def load_detections(det_path: Path) -> tuple[list[int], dict[int, list[Detection]]]:
    if not det_path.exists():
        raise ValueError(f"{det_path} doesn't exist!")

    detections = dict()

    sep: str = " "
    with open(det_path, encoding="utf-8") as f:
        while True:
            line: str = f.readline()
            if line == "":
                break

            line = line[:-1] if line[-1] == "\n" else line  # remove newline
            str_tokens: list[str] = line.split(sep=sep)
            # only conf score is float, others are integers
            tokens: list[int | float] = list(
                map(lambda x: float(x) if "." in x else int(x), str_tokens)
            )

            det: Detection = Detection(*tokens)
            frame: int = det.frame
            if frame not in detections:
                detections[frame] = []
            detections[frame].append(det)

    frames: list[int] = sorted(detections.keys())
    return frames, detections


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    inter_left = max(box1.bb_left, box2.bb_left)
    inter_top = max(box1.bb_top, box2.bb_top)
    inter_right = min(box1.right, box2.right)
    inter_bottom = min(box1.bottom, box2.bottom)

    inter_width = max(0, inter_right - inter_left)
    inter_height = max(0, inter_bottom - inter_top)

    area_intersection = inter_width * inter_height
    if area_intersection == 0:
        return 0.0

    area_union = box1.area + box2.area - area_intersection
    if area_union == 0.0:
        return 0.0

    iou = area_intersection / area_union
    return iou


def find_unmatched(nb_tracked, nb_new_obs, row_ind, col_ind):
    all_track_indices = np.arange(nb_tracked)
    all_obs_indices = np.arange(nb_new_obs)

    unmatched_track_indices = np.setdiff1d(all_track_indices, row_ind)
    unmatched_obs_indices = np.setdiff1d(all_obs_indices, col_ind)

    return unmatched_track_indices, unmatched_obs_indices


def assign_tracks(
    frames: list[int], detections: dict[int, list[Detection]]
):  # tracks: dict[int, list[Detection]]):

    max_track_id = 0

    det: Detection
    for det in detections[1]:
        det.id = max_track_id
        max_track_id += 1

    for cur_frame, next_frame in zip(frames[:], frames[1:]):  # last frame is dropped
        tracked = detections[cur_frame]
        nb_tracked = len(tracked)
        new_obs = detections[next_frame]
        nb_new_obs = len(new_obs)
        # print(f"[frame #{cur_frame}]: {nb_tracked} tracked, {nb_new_obs} new obs")

        similarity: np.ndarray = np.zeros((nb_tracked, nb_new_obs), dtype=float)
        for i in range(nb_tracked):
            for j in range(nb_new_obs):
                jaccard_idx = 1 - compute_iou(tracked[i].bb, new_obs[j].bb)
                similarity[i, j] = jaccard_idx

        row_ind, col_ind = linear_sum_assignment(similarity)

        unmatched_tracks, unmatched_obs = find_unmatched(
            nb_tracked, nb_new_obs, row_ind, col_ind
        )

        # Matched -> update existing tracks based on associations
        for track_idx, obs_idx in zip(row_ind, col_ind):
            new_obs[obs_idx].id = tracked[track_idx].id

        # Unmatched detections -> create new tracks
        for obs_idx in unmatched_obs:
            new_obs[obs_idx].id = max_track_id
            max_track_id += 1

        # TODO: Unmatched tracks -> remove tracks that exceed the "maximum missed frames"


def process_and_display_tracks(
    img_folder_path: Path,
    frames: list[int],
    detections: dict[int, list[Detection]],
    save_path: str | None = None,
    fps: int = 30,
) -> None:
    """
    Displays the tracking result and optionally saves it to a video file.
    """
    green = (0, 255, 0)
    video_writer = None

    # Determine frame size from the first image if saving is requested
    if save_path:
        first_img_path = img_folder_path.joinpath(f"{frames[0]:06}.jpg")
        first_img = cv2.imread(str(first_img_path), cv2.IMREAD_COLOR_BGR)
        if first_img is None:
            raise ValueError(
                f"Could not read first image to determine video size: {first_img_path}"
            )

        height, width, _ = first_img.shape

        # CHANGED: Use 'avc1' (H.264) for better compression
        # If this fails on your system, try 'H264' or 'X264'
        try:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        except Exception as e:
            print(f"Failed to initialize avc1 codec: {e}. Falling back to mp4v.")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        print(f"Recording video to: {save_path}")

    frame_num: int
    for frame_num in frames:
        img_name = f"{frame_num:06}.jpg"
        img_path = img_folder_path.joinpath(img_name)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR_BGR)

        if img is None:
            continue

        det: Detection
        for det in detections[frame_num]:
            bb: BoundingBox = det.bb
            cv2.rectangle(img, bb.top_left, bb.bot_right, green, 2)
            anchor = (bb.top_left[0], bb.top_left[1] - 10)
            track_id = str(det.id)
            cv2.putText(img, track_id, anchor, 0, 0.5, green, 2)

        # Display the frame
        cv2.imshow("image", img)

        # Write the frame if writer exists
        if video_writer:
            video_writer.write(img)

        if cv2.waitKey(1) == ord("q"):
            break

    # Cleanup
    if video_writer:
        video_writer.release()
        print("Video saved successfully.")

    cv2.destroyAllWindows()


def save_tracking_results(
    frames: list[int], detections: dict[int, list[Detection]], output_filename: str
) -> None:
    output_path = Path(output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        for frame_num in frames:
            if frame_num in detections:
                for det in detections[frame_num]:
                    f.write(det.to_gt_line() + "\n")

    print(f"Tracking results saved to {output_path.absolute()}")


def main() -> None:
    sequence_name = "ADL-Rundle-6"
    det_path: Path = Path(f"../data/{sequence_name}/det/Yolov5s/det.txt")
    img_folder_path: Path = Path(f"../data/{sequence_name}/img1/")

    frames: list[int]
    detections: dict[int, list[Detection]]
    frames, detections = load_detections(det_path)

    print("Assigning tracks...")
    assign_tracks(frames, detections)

    save_tracking_results(frames, detections, f"{sequence_name}.txt")

    print("Displaying and saving output...")
    process_and_display_tracks(
        img_folder_path, frames, detections, save_path="output.mp4", fps=30
    )


if __name__ == "__main__":
    main()
