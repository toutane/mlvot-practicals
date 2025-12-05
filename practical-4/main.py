from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch
import torchvision.models as models
import torchvision.transforms as T
from KalmanFilter import KalmanFilter

# --- Configuration ---
DT = 1.0 / 30.0
U_X = 0
U_Y = 0
STD_ACC = 5.0
X_STD_MEAS = 1.0
Y_STD_MEAS = 1.0

# TP4 Configuration: Weights for the Combined Score
ALPHA = 0.5  # Weight for IoU
BETA = 0.5  # Weight for Appearance Similarity
EMA_ALPHA = 0.9  # For updating track features (smooths out noise)


class FeatureExtractor:
    """
    Handles Feature Extraction using a lightweight CNN (MobileNetV2).
    """

    def __init__(self):
        # Using MobileNetV2 as a lightweight alternative to OSNet/EfficientNet
        # Pretrained on ImageNet gives decent generic features for this exercise.
        print("Loading Feature Extractor (MobileNetV2)...")
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # Remove the classifier head to get the feature vector (1280 dimensions)
        self.model.classifier = torch.nn.Identity()  # type: ignore
        self.model.eval()

        # Standard normalization for ImageNet models
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess_patch(self, im_crops):
        """
        Implementation of the specific preprocessing steps requested.
        Ref: Screenshot 2025-12-04 at 13.52.48.jpg
        """
        # 1. Resize to (64, 128) - Note: cv2.resize uses (width, height)
        roi_input = cv2.resize(im_crops, (64, 128))

        # 2. Convert BGR to RGB
        roi_input = cv2.cvtColor(roi_input, cv2.COLOR_BGR2RGB)

        # 3. Normalize
        roi_input = roi_input.astype(np.float32) / 255.0
        roi_input = (roi_input - self.mean) / self.std

        # 4. Channel first (H, W, C) -> (C, H, W) for PyTorch
        roi_input = np.moveaxis(roi_input, -1, 0)

        return roi_input

    def extract(self, img, bb):
        """
        Extracts the feature vector for a specific bounding box.
        """
        x, y, w, h = (
            int(bb.bb_left),
            int(bb.bb_top),
            int(bb.bb_width),
            int(bb.bb_height),
        )

        # Handle edge cases (clamping to image dimensions)
        img_h, img_w = img.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w <= 0 or h <= 0:
            return np.zeros(1280)  # Return zero vector if invalid crop

        # Crop
        im_crop = img[y : y + h, x : x + w]

        # Preprocess
        blob = self.preprocess_patch(im_crop)

        # Convert to Tensor and add batch dimension
        tensor = torch.from_numpy(blob).unsqueeze(0)

        # Inference
        with torch.no_grad():
            feature = self.model(tensor)

        # Return as numpy array (flattened)
        return feature.numpy().flatten()


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
        self.feature = None  # Placeholder for ReID vector

    def to_gt_line(self) -> str:
        return f"{self.frame},{self.id},{self.bb.bb_left},{self.bb.bb_top},{self.bb.bb_width},{self.bb.bb_height},1,-1,-1,-1"


class Track(object):
    def __init__(self, track_id, detection, feature):
        self.id = track_id
        self.kf = KalmanFilter(DT, U_X, U_Y, STD_ACC, X_STD_MEAS, Y_STD_MEAS)
        self.kf.x[0] = detection.bb.center[0]
        self.kf.x[1] = detection.bb.center[1]

        self.width = detection.bb.bb_width
        self.height = detection.bb.bb_height
        self.bb = detection.bb

        # Store appearance feature
        self.feature = feature

    def predict(self):
        predicted_state = self.kf.predict()
        pred_x = predicted_state[0][0]
        pred_y = predicted_state[1][0]

        new_left = pred_x - (self.width / 2.0)
        new_top = pred_y - (self.height / 2.0)
        self.bb = BoundingBox(new_left, new_top, self.width, self.height)
        return self.bb

    def update(self, detection, new_feature):
        self.kf.update(detection.bb.center)
        self.width = detection.bb.bb_width
        self.height = detection.bb.bb_height
        self.bb = detection.bb
        detection.id = self.id

        # Update feature using Exponential Moving Average (EMA)
        # This keeps the track's appearance stable over time
        if new_feature is not None:
            self.feature = EMA_ALPHA * self.feature + (1 - EMA_ALPHA) * new_feature


def load_detections(det_path: Path):
    if not det_path.exists():
        raise ValueError(f"{det_path} doesn't exist!")
    detections = dict()
    with open(det_path, encoding="utf-8") as f:
        for line in f:
            tokens = list(
                map(lambda x: float(x) if "." in x else int(x), line.strip().split(" "))
            )
            det = Detection(*tokens)
            if det.frame not in detections:
                detections[det.frame] = []
            detections[det.frame].append(det)
    return sorted(detections.keys()), detections


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    inter_left = max(box1.bb_left, box2.bb_left)
    inter_top = max(box1.bb_top, box2.bb_top)
    inter_right = min(box1.right, box2.right)
    inter_bottom = min(box1.bottom, box2.bottom)
    inter_w = max(0, inter_right - inter_left)
    inter_h = max(0, inter_bottom - inter_top)
    area_inter = inter_w * inter_h
    if area_inter == 0:
        return 0.0
    area_union = box1.area + box2.area - area_inter
    return area_inter / area_union if area_union > 0 else 0.0


def compute_cosine_similarity(track_features, det_features):
    """
    Computes Cosine Similarity between two lists of vectors.
    Returns a matrix of size (N_tracks, M_detections).
    Score ranges from 0 (opposite) to 1 (identical).
    """
    if len(track_features) == 0 or len(det_features) == 0:
        return np.zeros((len(track_features), len(det_features)))

    # 1 - cosine_distance = cosine_similarity
    # cdist 'cosine' returns distance (0=identical, 2=opposite)
    # We want similarity (1=identical, -1=opposite)
    # But usually in ReID 0 to 1 is sufficient after normalization

    # Using sklearn-style logic manually or via scipy
    dists = cdist(track_features, det_features, metric="cosine")
    sims = 1.0 - dists
    return sims


def run_tracker(frames, detections, img_folder_path):
    global_track_id = 1
    active_tracks = []  # List of Track objects

    # Initialize ReID model
    reid_model = FeatureExtractor()

    for frame_num in frames:
        current_detections = detections[frame_num]

        # Load Image for ReID
        img_name = f"{frame_num:06}.jpg"
        img_path = img_folder_path.joinpath(img_name)
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"Warning: could not read image {img_path}")
            continue

        # --- 0. FEATURE EXTRACTION Step ---
        # Extract features for all detections in current frame
        det_features = []
        valid_detections = []  # Filter out detections that might fail extraction

        for det in current_detections:
            feat = reid_model.extract(img, det.bb)
            det.feature = feat
            det_features.append(feat)
            valid_detections.append(det)

        current_detections = valid_detections
        det_features = np.array(det_features)

        # --- 1. PREDICT Step ---
        track_features = []
        for track in active_tracks:
            track.predict()
            track_features.append(track.feature)
        track_features = np.array(track_features)

        # --- 2. MATCH Step ---
        nb_tracks = len(active_tracks)
        nb_dets = len(current_detections)

        matches = []
        unmatched_track_indices = []
        unmatched_det_indices = []

        if nb_tracks > 0 and nb_dets > 0:
            # A. Geometric Cost (IoU)
            iou_matrix = np.zeros((nb_tracks, nb_dets), dtype=float)
            for t, track in enumerate(active_tracks):
                for d, det in enumerate(current_detections):
                    iou_matrix[t, d] = compute_iou(track.bb, det.bb)

            # B. Appearance Cost (Cosine Similarity)
            # Ref: Normalized Similarity = 1 / (1 + Euclidean) OR directly Cosine Similarity
            # We use Cosine Similarity as requested.
            sim_matrix = compute_cosine_similarity(track_features, det_features)

            # C. Combine Scores
            # S = alpha * IoU + beta * Normalized_Similarity
            # Ensure sim_matrix is non-negative (cosine sim is -1 to 1, we map -1..1 to 0..1 for safety)
            sim_matrix_norm = (sim_matrix + 1) / 2.0

            combined_score_matrix = (ALPHA * iou_matrix) + (BETA * sim_matrix_norm)

            # Convert Score to Cost (Hungarian minimizes cost)
            cost_matrix = 1.0 - combined_score_matrix

            # Hungarian Algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Filter matches
            match_threshold = 0.6  # Combined threshold

            used_rows = set()
            used_cols = set()

            for r, c in zip(row_ind, col_ind):
                # If score is too low (cost too high), reject
                if combined_score_matrix[r, c] < match_threshold:
                    continue

                matches.append((r, c))
                used_rows.add(r)
                used_cols.add(c)

            # Identify unmatched
            for t in range(nb_tracks):
                if t not in used_rows:
                    unmatched_track_indices.append(t)
            for d in range(nb_dets):
                if d not in used_cols:
                    unmatched_det_indices.append(d)

        else:
            unmatched_track_indices = list(range(nb_tracks))
            unmatched_det_indices = list(range(nb_dets))

        # --- 3. UPDATE Step ---
        new_active_tracks = []

        # Update matched tracks
        for t_idx, d_idx in matches:
            track = active_tracks[t_idx]
            det = current_detections[d_idx]
            track.update(det, det.feature)
            new_active_tracks.append(track)

        # Create new tracks
        for d_idx in unmatched_det_indices:
            det = current_detections[d_idx]
            new_track = Track(global_track_id, det, det.feature)
            det.id = global_track_id
            global_track_id += 1
            new_active_tracks.append(new_track)

        active_tracks = new_active_tracks


def process_and_display_tracks(
    img_folder_path: Path, frames: list, detections: dict, save_path: str | None = None
):
    green = (0, 255, 0)
    video_writer = None

    if save_path:
        first_img = cv2.imread(str(img_folder_path.joinpath(f"{frames[0]:06}.jpg")))
        h, w, _ = first_img.shape
        video_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)  # type: ignore
        )

    for frame_num in frames:
        img = cv2.imread(str(img_folder_path.joinpath(f"{frame_num:06}.jpg")))
        if img is None:
            continue

        for det in detections[frame_num]:
            cv2.rectangle(img, det.bb.top_left, det.bb.bot_right, green, 2)
            cv2.putText(
                img,
                str(det.id),
                (det.bb.top_left[0], det.bb.top_left[1] - 5),
                0,
                0.5,
                green,
                2,
            )

        cv2.imshow("Tracking", img)
        if video_writer:
            video_writer.write(img)
        if cv2.waitKey(1) == ord("q"):
            break

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


def main():
    sequence_name = "ADL-Rundle-6"
    det_path = Path(f"../data/{sequence_name}/det/Yolov5s/det.txt")
    img_folder_path = Path(f"../data/{sequence_name}/img1/")

    if not det_path.exists():
        print("Path not found.")
        return

    frames, detections = load_detections(det_path)

    print("Running Appearance-Aware (ReID + IoU + Kalman) Tracker...")
    run_tracker(
        frames, detections, img_folder_path
    )  # Pass img path for feature extraction

    print("Displaying...")
    process_and_display_tracks(
        img_folder_path, frames, detections, save_path="output_reid.mp4"
    )


if __name__ == "__main__":
    main()
