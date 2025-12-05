from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from KalmanFilter import KalmanFilter

DT = 1.0 / 30.0
U_X = 0
U_Y = 0
STD_ACC = 5.0
X_STD_MEAS = 1.0
Y_STD_MEAS = 1.0


class BoundingBox(object):
    def __init__(self, bb_left, bb_top, bb_width, bb_height):
        self.bb_left = float(bb_left)
        self.bb_top = float(bb_top)
        self.bb_width = float(bb_width)
        self.bb_height = float(bb_height)

        self.area = self.bb_width * self.bb_height

    @property
    def right(self):
        return self.bb_left + self.bb_width

    @property
    def bottom(self):
        return self.bb_top + self.bb_height

    @property
    def top_left(self):
        return (int(self.bb_left), int(self.bb_top))

    @property
    def bot_right(self):
        return (int(self.bb_left + self.bb_width), int(self.bb_top + self.bb_height))

    @property
    def center(self):
        cx = self.bb_left + (self.bb_width / 2.0)
        cy = self.bb_top + (self.bb_height / 2.0)
        return np.array([[cx], [cy]])


class Detection(object):
    def __init__(self, frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z):
        self.frame = frame
        self.id = id
        self.bb = BoundingBox(bb_left, bb_top, bb_width, bb_height)
        self.conf = conf
        self.x = x
        self.y = y
        self.z = z

    def to_gt_line(self) -> str:
        return f"{self.frame},{self.id},{self.bb.bb_left},{self.bb.bb_top},{self.bb.bb_width},{self.bb.bb_height},1,-1,-1,-1"


class Track(object):
    """
    Represents a single tracked object using a Kalman Filter.
    """

    def __init__(self, track_id, detection):
        self.id = track_id
        self.kf = KalmanFilter(DT, U_X, U_Y, STD_ACC, X_STD_MEAS, Y_STD_MEAS)

        # Initialize state with the first detection centroid
        self.kf.x[0] = detection.bb.center[0]
        self.kf.x[1] = detection.bb.center[1]

        # Store dimensions to reconstruct box after prediction (KF only tracks centroid)
        self.width = detection.bb.bb_width
        self.height = detection.bb.bb_height

        # Current best estimate of the bounding box
        self.bb = detection.bb

    def predict(self):
        """
        Advances the state vector using the Kalman Filter prediction step.
        Returns the predicted BoundingBox.
        """
        predicted_state = self.kf.predict()  # Returns [x, y, vx, vy]

        pred_x = predicted_state[0][0]
        pred_y = predicted_state[1][0]

        # We assume width/height stay roughly the same or use last known
        new_left = pred_x - (self.width / 2.0)
        new_top = pred_y - (self.height / 2.0)

        self.bb = BoundingBox(new_left, new_top, self.width, self.height)
        return self.bb

    def update(self, detection):
        """
        Corrects the state vector using the new detection measurement.
        """
        measurement = detection.bb.center
        self.kf.update(measurement)

        self.width = detection.bb.bb_width
        self.height = detection.bb.bb_height

        self.bb = detection.bb
        detection.id = self.id


def load_detections(det_path: Path):
    if not det_path.exists():
        raise ValueError(f"{det_path} doesn't exist!")

    detections = dict()
    sep = " "
    with open(det_path, encoding="utf-8") as f:
        while True:
            line = f.readline()
            if line == "":
                break
            line = line.strip()
            str_tokens = line.split(sep)
            tokens = list(map(lambda x: float(x) if "." in x else int(x), str_tokens))

            det = Detection(*tokens)
            frame = det.frame
            if frame not in detections:
                detections[frame] = []
            detections[frame].append(det)

    frames = sorted(detections.keys())
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

    return area_intersection / area_union


def assign_tracks(frames, detections):
    """
    Core Logic: Predict -> Match (IoU) -> Update/Create/Delete
    """

    global_track_id = 1
    active_tracks = []  # List of Track objects

    for frame_num in frames:
        current_detections = detections[frame_num]

        # Predict the new position for all currently active tracks
        for track in active_tracks:
            track.predict()

        # Hungarian Algorithm
        nb_tracks = len(active_tracks)
        nb_dets = len(current_detections)

        if nb_tracks == 0:
            # If no tracks exist, all detections are new tracks
            unmatched_track_indices = []
            unmatched_det_indices = list(range(nb_dets))
            matches = []
        else:
            # Compute Cost Matrix (1 - IoU)
            cost_matrix = np.zeros((nb_tracks, nb_dets), dtype=float)
            for t, track in enumerate(active_tracks):
                for d, det in enumerate(current_detections):
                    # Calculate IoU between Predicted Box and Detected Box
                    iou = compute_iou(track.bb, det.bb)
                    cost_matrix[t, d] = 1.0 - iou

            # Apply Hungarian Algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Identify matches and unmatched
            matched_indices = set()
            unmatched_track_indices = []
            unmatched_det_indices = []
            matches = []

            iou_threshold = 0.1

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < (1.0 - iou_threshold):
                    matches.append((r, c))
                    matched_indices.add(r)
                    matched_indices.add(c)
                else:
                    unmatched_track_indices.append(r)
                    unmatched_det_indices.append(c)

            # Find indices that were not in row_ind/col_ind at all
            for t in range(nb_tracks):
                if t not in row_ind:
                    unmatched_track_indices.append(t)
                elif t in row_ind and t not in [m[0] for m in matches]:
                    pass

            for d in range(nb_dets):
                if d not in col_ind:
                    unmatched_det_indices.append(d)
                elif d in col_ind and d not in [m[1] for m in matches]:
                    pass

        new_active_tracks = []

        for t_idx, d_idx in matches:
            track = active_tracks[t_idx]
            det = current_detections[d_idx]

            track.update(det)
            new_active_tracks.append(track)

        for d_idx in unmatched_det_indices:
            det = current_detections[d_idx]
            new_track = Track(global_track_id, det)
            det.id = global_track_id
            global_track_id += 1
            new_active_tracks.append(new_track)

        active_tracks = new_active_tracks


def process_and_display_tracks(
    img_folder_path: Path,
    frames: list[int],
    detections: dict[int, list[Detection]],
    save_path: str | None = None,
    fps: int = 30,
) -> None:
    green = (0, 255, 0)
    video_writer = None

    if save_path:
        first_img_path = img_folder_path.joinpath(f"{frames[0]:06}.jpg")
        first_img = cv2.imread(str(first_img_path), cv2.IMREAD_COLOR_BGR)
        if first_img is None:
            height, width = 1080, 1920
        else:
            height, width, _ = first_img.shape

        try:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")  # type: ignore
            video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        except:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for frame_num in frames:
        img_name = f"{frame_num:06}.jpg"
        img_path = img_folder_path.joinpath(img_name)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR_BGR)

        if img is None:
            continue

        for det in detections[frame_num]:
            # Draw bbox
            cv2.rectangle(img, det.bb.top_left, det.bb.bot_right, green, 2)
            # Draw ID
            anchor = (det.bb.top_left[0], det.bb.top_left[1] - 10)
            cv2.putText(img, str(det.id), anchor, 0, 0.5, green, 2)

        cv2.imshow("image", img)

        if video_writer:
            video_writer.write(img)

        if cv2.waitKey(1) == ord("q"):
            break

    if video_writer:
        video_writer.release()

    cv2.destroyAllWindows()


def save_tracking_results(frames, detections, output_filename):
    output_path = Path(output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        for frame_num in frames:
            if frame_num in detections:
                for det in detections[frame_num]:
                    f.write(det.to_gt_line() + "\n")
    print(f"Results saved to {output_path.absolute()}")


def main():
    sequence_name = "ADL-Rundle-6"
    det_path = Path(f"../data/{sequence_name}/det/Yolov5s/det.txt")
    img_folder_path = Path(f"../data/{sequence_name}/img1/")

    if not det_path.exists():
        print(f"Error: path {det_path} does not exist.")
        return

    frames, detections = load_detections(det_path)

    print("Running Kalman-Guided IoU Tracker...")
    assign_tracks(frames, detections)

    save_tracking_results(frames, detections, f"{sequence_name}.txt")

    print("Displaying...")
    process_and_display_tracks(
        img_folder_path, frames, detections, save_path="output.mp4", fps=30
    )


if __name__ == "__main__":
    main()
