# app/processing.py
"""
Monolithic face image processing pipeline.
Adapted from new_new_test.py to work with base64 inputs for API usage.
"""

import base64
import io
import os
import glob
import html
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from app.logger import console
import cv2
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.spatial import Delaunay

# Configure matplotlib for headless server environment
import matplotlib
matplotlib.use('Agg')


# ==========================
# CONSTANTS
# ==========================

# MediaPipe landmark indices
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 193]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 417]
LEFT_EYE = [33, 133, 159, 145, 160, 161, 246, 7, 163, 144, 145, 153, 154, 155, 157]
RIGHT_EYE = [362, 263, 386, 374, 387, 388, 466, 249, 390, 373, 374, 380, 381, 382, 384]

# Overlay styling
OVERLAY_COLOR = "#b266ff"
OVERLAY_ALPHA = 0.35

# Asset paths
EYE_CONTOUR_PATH = os.getenv("EYE_CONTOUR_PATH", "assets/eye_contours")


# ==========================
# BASE64 / IMAGE HELPERS
# ==========================

def base64_to_pil(b64: str) -> Image.Image:
    """
    Accepts either raw base64 or data URI (data:image/png;base64,...)
    Returns a PIL Image in RGB mode.
    """
    header, _, payload = b64.partition(",")
    if payload == "":
        payload = header
    data = base64.b64decode(payload)
    return Image.open(io.BytesIO(data)).convert("RGB")


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Return H x W x C numpy array."""
    return np.array(img)


def numpy_to_base64_png(image_np: np.ndarray) -> str:
    """Convert numpy image to data:image/png;base64,..."""
    # Handle different channel counts
    if image_np.ndim == 2:
        img = Image.fromarray(image_np.astype("uint8"), "L")
    elif image_np.shape[2] == 3:
        img = Image.fromarray(image_np.astype("uint8"), "RGB")
    else:
        img = Image.fromarray(image_np.astype("uint8"), "RGBA")
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ==========================
# LANDMARK PROCESSING
# ==========================

def extract_landmarks_array(landmarks: List[Dict[str, float]]) -> np.ndarray:
    """Convert landmark dicts to numpy array."""
    if not landmarks:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([[lm["x"], lm["y"]] for lm in landmarks], dtype=np.float32)


def scale_landmarks_to_segmentation(
    landmarks: List[Dict[str, float]], 
    seg_w: int, 
    seg_h: int,
    orig_w: Optional[int] = None,
    orig_h: Optional[int] = None
) -> np.ndarray:
    """
    Scale landmarks from original image space to segmentation space.
    If original dimensions not provided, assumes landmarks already in seg space.
    """
    pts_orig = extract_landmarks_array(landmarks)
    
    if pts_orig.shape[0] == 0:
        return pts_orig
    
    if orig_w is None or orig_h is None:
        return pts_orig
    
    if orig_w == seg_w and orig_h == seg_h:
        return pts_orig
    
    scale_x = seg_w / orig_w
    scale_y = seg_h / orig_h
    
    pts = np.empty_like(pts_orig)
    pts[:, 0] = pts_orig[:, 0] * scale_x
    pts[:, 1] = pts_orig[:, 1] * scale_y
    
    return pts


# ==========================
# PCA ROTATION
# ==========================

def compute_pca_rotation(pts: np.ndarray, seg_w: int, seg_h: int) -> Tuple[float, np.ndarray]:
    """
    Compute rotation angle and matrix using PCA on landmarks.
    Returns: (angle_deg, rotation_matrix)
    """
    if pts.shape[0] < 2:
        # Not enough points for PCA
        center = (seg_w / 2.0, seg_h / 2.0)
        M = cv2.getRotationMatrix2D(center, 0.0, 1.0)
        return 0.0, M
    
    center = np.array([seg_w / 2.0, seg_h / 2.0], dtype=np.float32)
    pts_rel = pts - center
    
    cov = np.cov(pts_rel.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v_major = eigvecs[:, np.argmax(eigvals)]
    v_major /= np.linalg.norm(v_major)
    
    target_down = np.array([0.0, 1.0], dtype=np.float32)
    angle_rad = np.arctan2(
        v_major[0] * target_down[1] - v_major[1] * target_down[0],
        v_major.dot(target_down)
    )
    theta = -angle_rad
    angle_deg = float(np.degrees(theta))
    
    M = cv2.getRotationMatrix2D((center[0], center[1]), angle_deg, 1.0)
    
    return angle_deg, M


def rotate_all(
    image_np: np.ndarray,
    seg_np: np.ndarray,
    pts: np.ndarray,
    M: np.ndarray,
    seg_w: int,
    seg_h: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate image, segmentation, and landmarks using rotation matrix M.
    """
    rot_img = cv2.warpAffine(
        image_np,
        M,
        (seg_w, seg_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    
    rot_seg = cv2.warpAffine(
        seg_np,
        M,
        (seg_w, seg_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE,
    )
    
    # Rotate landmarks
    if pts.shape[0] > 0:
        pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
        pts_rot = (M @ pts_h.T).T
    else:
        pts_rot = pts
    
    return rot_img, rot_seg, pts_rot


# ==========================
# BOUNDARY DETECTION
# ==========================

def find_hair_boundaries(seg_gray: np.ndarray) -> Tuple[float, float]:
    """
    Find the leftmost and rightmost x-coordinates where white (hair) meets the face.
    """
    seg_h, seg_w = seg_gray.shape
    upper_half = seg_gray[:seg_h // 2, :]
    white_mask = (upper_half == 255)
    white_cols = np.any(white_mask, axis=0)
    white_col_indices = np.where(white_cols)[0]
    
    if len(white_col_indices) > 0:
        left_x = float(np.min(white_col_indices))
        right_x = float(np.max(white_col_indices))
    else:
        left_x = seg_w * 0.1
        right_x = seg_w * 0.9
    
    return left_x, right_x


# ==========================
# LINE CONSTRUCTION
# ==========================

def construct_eyebrow_line(
    pts_rot: np.ndarray, 
    seg_w: int, 
    seg_h: int, 
    left_bound: float, 
    right_bound: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs a smooth curve following the eyebrow contour.
    """
    if pts_rot.shape[0] < max(max(LEFT_EYEBROW), max(RIGHT_EYEBROW)) + 1:
        # Not enough landmarks - return straight line
        avg_y = seg_h * 0.3
        x_curve = np.array([left_bound, right_bound])
        y_curve = np.array([avg_y, avg_y])
        return x_curve, y_curve
    
    left_brow_pts = pts_rot[LEFT_EYEBROW]
    right_brow_pts = pts_rot[RIGHT_EYEBROW]
    
    left_brow_top_y = np.min(left_brow_pts[:, 1])
    right_brow_top_y = np.min(right_brow_pts[:, 1])
    
    left_brow_inner_x = left_brow_pts[0, 0]
    right_brow_inner_x = right_brow_pts[0, 0]
    
    left_eyebrow_outer_x = np.min(left_brow_pts[:, 0])
    right_eyebrow_outer_x = np.max(right_brow_pts[:, 0])
    
    eyebrow_width = right_eyebrow_outer_x - left_eyebrow_outer_x
    max_extension = eyebrow_width * 0.2
    
    line_left_x = max(left_bound, left_eyebrow_outer_x - max_extension)
    line_right_x = min(right_bound, right_eyebrow_outer_x + max_extension)
    
    anchor_points = []
    anchor_points.append([line_left_x, left_brow_top_y])
    
    for idx in [70, 63, 105, 66, 107]:
        if idx < len(pts_rot):
            anchor_points.append(pts_rot[idx])
    
    center_x = (left_brow_inner_x + right_brow_inner_x) / 2
    center_y = max(left_brow_top_y, right_brow_top_y) + 10
    anchor_points.append([center_x, center_y])
    
    for idx in [336, 296, 334, 293, 300]:
        if idx < len(pts_rot):
            anchor_points.append(pts_rot[idx])
    
    anchor_points.append([line_right_x, right_brow_top_y])
    
    anchor_points = np.array(anchor_points)
    sort_idx = np.argsort(anchor_points[:, 0])
    anchor_points = anchor_points[sort_idx]
    
    try:
        cs = CubicSpline(anchor_points[:, 0], anchor_points[:, 1], bc_type='natural')
        x_curve = np.linspace(line_left_x, line_right_x, 500)
        y_curve = cs(x_curve)
        y_curve = np.clip(y_curve, 0, seg_h)
        return x_curve, y_curve
    except Exception as e:
        console.log(f"Error fitting eyebrow spline: {e}")
        avg_y = (left_brow_top_y + right_brow_top_y) / 2
        x_curve = np.array([line_left_x, line_right_x])
        y_curve = np.array([avg_y, avg_y])
        return x_curve, y_curve


def construct_nose_line_straight(
    pts_rot: np.ndarray, 
    left_bound: float, 
    right_bound: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a straight horizontal line tangent to the bottom of the nose.
    """
    if pts_rot.shape[0] < max(2, 129, 358) + 1:
        # Not enough landmarks
        x_line = np.linspace(left_bound, right_bound, 500)
        y_line = np.full_like(x_line, 0.0)
        return x_line, y_line
    
    nose_tip_y = pts_rot[2, 1]
    left_nostril_y = pts_rot[129, 1]
    right_nostril_y = pts_rot[358, 1]
    
    nose_bottom_y = max(nose_tip_y, left_nostril_y, right_nostril_y)
    nose_line_y = nose_bottom_y + 10
    
    x_line = np.linspace(left_bound, right_bound, 500)
    y_line = np.full_like(x_line, nose_line_y)
    
    return x_line, y_line


# ==========================
# CONTOUR EXTRACTION
# ==========================

def get_hairline_contour(seg_gray: np.ndarray) -> Optional[np.ndarray]:
    """Extract hairline contour from segmentation."""
    mask = np.uint8(seg_gray == 255)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(cnts) == 0:
        # Try max value as fallback
        unique_vals = np.unique(seg_gray)
        if len(unique_vals) > 0:
            max_val = unique_vals[-1]
            mask = np.uint8(seg_gray == max_val)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(cnts) == 0:
            return None
    
    largest = max(cnts, key=cv2.contourArea)
    return largest.squeeze(1)


def get_nose_contour_from_landmarks(seg_gray: np.ndarray, pts_rot: np.ndarray) -> Optional[np.ndarray]:
    """Extract nose contour using landmark-based value detection."""
    h, w = seg_gray.shape
    
    if pts_rot.shape[0] < max(1, 2, 6, 129, 358) + 1:
        return None
    
    nose_tip = pts_rot[1]
    
    nose_sample_points = [
        pts_rot[1],
        pts_rot[2],
        pts_rot[6],
        pts_rot[129],
        pts_rot[358],
    ]
    
    nose_values = []
    for pt in nose_sample_points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            val = seg_gray[y, x]
            if val > 0:
                nose_values.append(val)
    
    if len(nose_values) == 0:
        return None
    
    nose_val = max(set(nose_values), key=nose_values.count)
    
    mask = np.uint8(seg_gray == nose_val)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(cnts) == 0:
        return None
    
    nose_tip_x, nose_tip_y = int(nose_tip[0]), int(nose_tip[1])
    
    for cnt in cnts:
        if cv2.pointPolygonTest(cnt, (nose_tip_x, nose_tip_y), False) >= 0:
            return cnt.squeeze(1)
    
    largest = max(cnts, key=cv2.contourArea)
    return largest.squeeze(1)


def get_head_outline(seg_gray: np.ndarray) -> Optional[np.ndarray]:
    """Extract overall head outline."""
    mask = np.uint8(seg_gray > 0)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(cnts) == 0:
        return None
    
    largest = max(cnts, key=cv2.contourArea)
    return largest.squeeze(1)
# ==========================
# REGION CONSTRUCTION
# ==========================

def create_region_1(
    hairline_contour: Optional[np.ndarray],
    eyebrow_x: np.ndarray,
    eyebrow_y: np.ndarray,
    left_bound: float,
    right_bound: float
) -> Optional[np.ndarray]:
    """Create forehead region polygon."""
    if hairline_contour is None or len(eyebrow_x) == 0:
        return None
    
    eyebrow_left_x = eyebrow_x[0]
    eyebrow_right_x = eyebrow_x[-1]
    eyebrow_left_y = eyebrow_y[0]
    eyebrow_right_y = eyebrow_y[-1]
    
    eyebrow_width = eyebrow_right_x - eyebrow_left_x
    margin = eyebrow_width * 0.1
    
    inner_left_x = eyebrow_left_x + margin
    inner_right_x = eyebrow_right_x - margin
    
    hairline_filtered = []
    for pt in hairline_contour:
        x, y = pt
        if inner_left_x <= x <= inner_right_x:
            eyebrow_y_at_x = np.interp(x, eyebrow_x, eyebrow_y)
            if y < eyebrow_y_at_x:
                hairline_filtered.append([x, y])
    
    if len(hairline_filtered) < 3:
        return None
    
    hairline_filtered = sorted(hairline_filtered, key=lambda p: p[0])
    hairline_arr = np.array(hairline_filtered)
    
    x_unique = np.unique(hairline_arr[:, 0].astype(int))
    hairline_boundary = []
    
    for x in x_unique:
        matching_points = hairline_arr[np.abs(hairline_arr[:, 0] - x) < 1]
        if len(matching_points) > 0:
            lowest_point = matching_points[np.argmax(matching_points[:, 1])]
            hairline_boundary.append(lowest_point)
    
    hairline_boundary = np.array(hairline_boundary)
    
    if len(hairline_boundary) < 2:
        return None
    
    leftmost_hairline = hairline_boundary[0]
    rightmost_hairline = hairline_boundary[-1]
    
    polygon_points = []
    polygon_points.append([eyebrow_left_x, eyebrow_left_y])
    
    if leftmost_hairline[0] > eyebrow_left_x:
        polygon_points.append([leftmost_hairline[0], eyebrow_left_y])
    
    polygon_points.extend(hairline_boundary.tolist())
    
    if rightmost_hairline[0] < eyebrow_right_x:
        polygon_points.append([rightmost_hairline[0], rightmost_hairline[1]])
        polygon_points.append([rightmost_hairline[0], eyebrow_right_y])
    
    polygon_points.append([eyebrow_right_x, eyebrow_right_y])
    
    eyebrow_points = np.column_stack([eyebrow_x, eyebrow_y])
    polygon_points.extend(eyebrow_points[::-1].tolist())
    
    return np.array(polygon_points)


def create_region_4(
    head_outline: Optional[np.ndarray],
    nose_x: np.ndarray,
    nose_y: np.ndarray,
    left_bound: float,
    right_bound: float
) -> Optional[np.ndarray]:
    """Create lower face region polygon."""
    if head_outline is None or len(nose_y) == 0:
        return None
    
    nose_line_y_avg = np.mean(nose_y)
    
    lower_face_points = []
    for pt in head_outline:
        x, y = pt
        if y > nose_line_y_avg:
            lower_face_points.append([x, y])
    
    if len(lower_face_points) < 3:
        return None
    
    nose_line_points = np.column_stack([nose_x, nose_y])
    polygon_points = np.array(nose_line_points.tolist() + lower_face_points)
    
    return polygon_points


def get_main_face_label(seg_gray: np.ndarray, pts_rot: np.ndarray) -> Optional[int]:
    """Identify the main face segmentation value."""
    h, w = seg_gray.shape
    
    candidate_indices = [1, 2, 4, 6, 9, 10, 33, 133, 362, 263, 61, 291, 199]
    
    vals = []
    for idx in candidate_indices:
        if idx >= len(pts_rot):
            continue
        x, y = int(pts_rot[idx, 0]), int(pts_rot[idx, 1])
        if 0 <= x < w and 0 <= y < h:
            v = int(seg_gray[y, x])
            if v > 0:
                vals.append(v)
    
    if not vals:
        all_vals = seg_gray[seg_gray > 0].astype(np.int32).ravel()
        if all_vals.size == 0:
            return None
        face_label = int(np.bincount(all_vals).argmax())
        return face_label
    
    face_label = max(set(vals), key=vals.count)
    return face_label


def create_face_mesh_mask(pts_rot: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Create face mesh mask using Delaunay triangulation."""
    h, w = shape
    if pts_rot.shape[0] < 3:
        return np.zeros((h, w), dtype=np.uint8)
    
    tri = Delaunay(pts_rot[:, :2])
    mesh_mask = np.zeros((h, w), dtype=np.uint8)
    
    for simplex in tri.simplices:
        tri_pts = np.round(pts_rot[simplex, :2]).astype(np.int32)
        tri_pts[:, 0] = np.clip(tri_pts[:, 0], 0, w - 1)
        tri_pts[:, 1] = np.clip(tri_pts[:, 1], 0, h - 1)
        cv2.fillConvexPoly(mesh_mask, tri_pts, 1)
    
    return mesh_mask


def get_ear_region_contours(
    seg_gray: np.ndarray,
    face_label: Optional[int],
    mesh_mask: np.ndarray,
    min_area: int = 200
) -> List[np.ndarray]:
    """Extract ear/outer-face region contours."""
    if face_label is None:
        return []
    
    face_mask = (seg_gray == face_label).astype(np.uint8)
    outside_mesh = np.logical_and(face_mask == 1, mesh_mask == 0).astype(np.uint8)
    
    kernel = np.ones((3, 3), np.uint8)
    outside_mesh = cv2.morphologyEx(outside_mesh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    cnts, _ = cv2.findContours(outside_mesh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area >= min_area and c.shape[0] >= 3:
            contours.append(c.squeeze(1))
    
    return contours


# ==========================
# EYE BAG CONTOURS
# ==========================

def load_eye_bag_template(template_path: str) -> Optional[np.ndarray]:
    """Load eye bag template from PNG file."""
    if not os.path.exists(template_path):
        png_files = glob.glob(os.path.join(template_path, "*.png"))
        if len(png_files) > 0:
            template_path = png_files[0]
        else:
            return None
    
    img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(cnts) == 0:
        return None
    
    largest = max(cnts, key=cv2.contourArea)
    contour = largest.squeeze(1)
    
    return contour


def get_eye_properties(pts_rot: np.ndarray, eye_indices: List[int]) -> Dict[str, Any]:
    """Calculate eye center, width, height, and bottom-most point."""
    eye_pts = pts_rot[eye_indices]
    
    cx = np.mean(eye_pts[:, 0])
    cy = np.mean(eye_pts[:, 1])
    
    min_x, max_x = np.min(eye_pts[:, 0]), np.max(eye_pts[:, 0])
    min_y, max_y = np.min(eye_pts[:, 1]), np.max(eye_pts[:, 1])
    
    width = max_x - min_x
    height = max_y - min_y
    bottom_y = max_y
    
    return {
        'center': np.array([cx, cy]),
        'width': width,
        'height': height,
        'bottom_y': bottom_y,
        'bounds': (min_x, max_x, min_y, max_y)
    }


def scale_and_position_eye_bag(
    template_contour: Optional[np.ndarray],
    eye_props: Dict[str, Any],
    scale_multiplier: float = 0.85,
    offset_down: float = 2
) -> Optional[np.ndarray]:
    """Scale template to match eye size and position below eye."""
    if template_contour is None or len(template_contour) < 3:
        return None
    
    template_cx = np.mean(template_contour[:, 0])
    template_cy = np.mean(template_contour[:, 1])
    template_width = np.max(template_contour[:, 0]) - np.min(template_contour[:, 0])
    
    if template_width == 0:
        return None
    
    template_top_y = np.min(template_contour[:, 1])
    
    scale_factor = (eye_props['width'] * scale_multiplier * 1.5) / template_width
    
    centered = template_contour - np.array([template_cx, template_cy])
    scaled = centered * scale_factor
    
    scaled_top_offset = np.min(scaled[:, 1])
    
    target_x = eye_props['center'][0]
    target_y = eye_props['bottom_y'] + offset_down - scaled_top_offset
    
    positioned = scaled + np.array([target_x, target_y])
    
    return positioned


def create_eye_bag_contours(
    pts_rot: np.ndarray,
    eye_contour_path: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Create eye bag contours for both eyes."""
    png_files = glob.glob(os.path.join(eye_contour_path, "*.png"))
    
    if len(png_files) == 0:
        return None, None
    
    template_path = png_files[0]
    template = load_eye_bag_template(template_path)
    
    if template is None:
        return None, None
    
    if pts_rot.shape[0] < max(max(LEFT_EYE), max(RIGHT_EYE)) + 1:
        return None, None
    
    left_eye_props = get_eye_properties(pts_rot, LEFT_EYE)
    right_eye_props = get_eye_properties(pts_rot, RIGHT_EYE)
    
    left_bag = scale_and_position_eye_bag(
        template,
        left_eye_props,
        scale_multiplier=0.85,
        offset_down=2
    )
    
    # Mirror template for right eye
    template_mirrored = template.copy()
    template_cx = np.mean(template[:, 0])
    template_mirrored[:, 0] = 2 * template_cx - template[:, 0]
    
    right_bag = scale_and_position_eye_bag(
        template_mirrored,
        right_eye_props,
        scale_multiplier=0.85,
        offset_down=2
    )
    
    return left_bag, right_bag


# ==========================
# SVG GENERATION
# ==========================

def polygon_to_svg_path(polygon: np.ndarray) -> str:
    """Convert numpy polygon to SVG path string."""
    if polygon is None or len(polygon) == 0:
        return ""
    
    d_parts = []
    for i, (x, y) in enumerate(polygon):
        prefix = "M" if i == 0 else "L"
        d_parts.append(f"{prefix}{x:.2f} {y:.2f}")
    d_parts.append("Z")
    
    return " ".join(d_parts)


def regions_to_svg(
    rotated_image_b64: str,
    width: int,
    height: int,
    region_1_polygon: Optional[np.ndarray],
    nose_contour: Optional[np.ndarray],
    region_4_polygon: Optional[np.ndarray],
    ear_contours: List[np.ndarray],
    left_eye_bag: Optional[np.ndarray],
    right_eye_bag: Optional[np.ndarray],
    overlay_color: str = OVERLAY_COLOR,
    overlay_alpha: float = OVERLAY_ALPHA
) -> str:
    """
    Build SVG containing rotated face image with translucent region overlays.
    """
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="xMidYMid slice">'
    ]
    
    # Embedded rotated face image
    if rotated_image_b64:
        header, _, payload = rotated_image_b64.partition(",")
        if payload == "":
            payload = header
            data_uri = f"data:image/png;base64,{payload}"
        else:
            data_uri = rotated_image_b64
        
        data_uri_escaped = html.escape(data_uri, quote=True)
        svg_parts.append(
            f'<image x="0" y="0" width="{width}" height="{height}" '
            f'xlink:href="{data_uri_escaped}" />'
        )
    
    # Forehead region (Region 1)
    if region_1_polygon is not None:
        path_d = polygon_to_svg_path(region_1_polygon)
        if path_d:
            svg_parts.append(
                f'<path data-region="forehead" d="{path_d}" '
                f'fill="{overlay_color}" fill-opacity="{overlay_alpha}" '
                f'stroke="none" />'
            )
    
    # Nose region
    if nose_contour is not None:
        path_d = polygon_to_svg_path(nose_contour)
        if path_d:
            svg_parts.append(
                f'<path data-region="nose" d="{path_d}" '
                f'fill="{overlay_color}" fill-opacity="{overlay_alpha}" '
                f'stroke="none" />'
            )
    
    # Lower face region (Region 4)
    if region_4_polygon is not None:
        path_d = polygon_to_svg_path(region_4_polygon)
        if path_d:
            svg_parts.append(
                f'<path data-region="lower_face" d="{path_d}" '
                f'fill="{overlay_color}" fill-opacity="{overlay_alpha}" '
                f'stroke="none" />'
            )
    
    # Ear regions
    for i, ear_contour in enumerate(ear_contours):
        path_d = polygon_to_svg_path(ear_contour)
        if path_d:
            svg_parts.append(
                f'<path data-region="ear_{i}" d="{path_d}" '
                f'fill="{overlay_color}" fill-opacity="{overlay_alpha}" '
                f'stroke="none" />'
            )
    
    # Left eye bag
    if left_eye_bag is not None:
        path_d = polygon_to_svg_path(left_eye_bag)
        if path_d:
            svg_parts.append(
                f'<path data-region="left_eye_bag" d="{path_d}" '
                f'fill="{overlay_color}" fill-opacity="{overlay_alpha}" '
                f'stroke="none" />'
            )
    
    # Right eye bag
    if right_eye_bag is not None:
        path_d = polygon_to_svg_path(right_eye_bag)
        if path_d:
            svg_parts.append(
                f'<path data-region="right_eye_bag" d="{path_d}" '
                f'fill="{overlay_color}" fill-opacity="{overlay_alpha}" '
                f'stroke="none" />'
            )
    
    svg_parts.append("</svg>")
    return "".join(svg_parts)


def svg_to_base64(svg_str: str, data_uri: bool = True) -> str:
    """Return base64-encoded SVG, optionally as a data URI."""
    b = svg_str.encode("utf-8")
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}" if data_uri else b64


# ==========================
# MAIN PROCESSING PIPELINE
# ==========================

def process_face_image(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main monolithic processing function.
    
    Args:
        payload: Dict containing:
            - image: base64 encoded original image
            - segmentation_map: base64 encoded segmentation
            - landmarks: List of {x, y} dicts
            - original_width: Optional original image width
            - original_height: Optional original image height
    
    Returns:
        Dict containing:
            - svg: base64 encoded SVG with overlays
            - mask_contours: Dict of region contours (optional)
    """
    # ==========================
    # 1. DECODE INPUTS
    # ==========================
    seg_b64 = payload["segmentation_map"]
    orig_b64 = payload.get("image")
    landmarks = payload.get("landmarks", [])
    orig_w = payload.get("original_width")
    orig_h = payload.get("original_height")
    
    seg_pil = base64_to_pil(seg_b64)
    seg_np = pil_to_numpy(seg_pil)
    
    if orig_b64:
        orig_pil = base64_to_pil(orig_b64)
        orig_np = pil_to_numpy(orig_pil)
    else:
        orig_np = seg_np.copy()
    
    seg_h, seg_w = seg_np.shape[:2]
    
    # Ensure we're working with RGB (3 channels), not RGBA
    if orig_np.ndim == 3 and orig_np.shape[2] == 4:
        orig_np = orig_np[:, :, :3]
    
    # Convert segmentation to grayscale if needed
    if seg_np.ndim == 3:
        if seg_np.shape[2] == 4:
            seg_np = seg_np[:, :, :3]
        seg_gray = cv2.cvtColor(seg_np, cv2.COLOR_RGB2GRAY)
    else:
        seg_gray = seg_np
    
    # Resize original image to match segmentation
    if orig_np.shape[:2] != (seg_h, seg_w):
        orig_np = cv2.resize(orig_np, (seg_w, seg_h), interpolation=cv2.INTER_LINEAR)
    
    # ==========================
    # 2. SCALE LANDMARKS
    # ==========================
    pts = scale_landmarks_to_segmentation(landmarks, seg_w, seg_h, orig_w, orig_h)
    
    # ==========================
    # 3. COMPUTE PCA ROTATION
    # ==========================
    angle_deg, M = compute_pca_rotation(pts, seg_w, seg_h)
    
    # ==========================
    # 4. ROTATE EVERYTHING
    # ==========================
    rot_img, rot_seg, pts_rot = rotate_all(orig_np, seg_gray, pts, M, seg_w, seg_h)
    
    # ==========================
    # 5. FIND BOUNDARIES
    # ==========================
    left_bound, right_bound = find_hair_boundaries(rot_seg)
    
    # ==========================
    # 6. CONSTRUCT LINES
    # ==========================
    eyebrow_x, eyebrow_y = construct_eyebrow_line(pts_rot, seg_w, seg_h, left_bound, right_bound)
    nose_x, nose_y = construct_nose_line_straight(pts_rot, left_bound, right_bound)
    
    # ==========================
    # 7. EXTRACT CONTOURS
    # ==========================
    hairline_contour = get_hairline_contour(rot_seg)
    nose_contour = get_nose_contour_from_landmarks(rot_seg, pts_rot)
    head_outline = get_head_outline(rot_seg)
    
    # ==========================
    # 8. BUILD REGIONS
    # ==========================
    region_1 = create_region_1(hairline_contour, eyebrow_x, eyebrow_y, left_bound, right_bound)
    region_4 = create_region_4(head_outline, nose_x, nose_y, left_bound, right_bound)
    
    main_face_label = get_main_face_label(rot_seg, pts_rot)
    face_mesh_mask = create_face_mesh_mask(pts_rot, rot_seg.shape[:2])
    ear_contours = get_ear_region_contours(rot_seg, main_face_label, face_mesh_mask)
    
    # ==========================
    # 9. CREATE EYE BAG CONTOURS
    # ==========================
    left_eye_bag, right_eye_bag = create_eye_bag_contours(pts_rot, EYE_CONTOUR_PATH)
    
    # ==========================
    # 10. GENERATE SVG
    # ==========================
    rot_img_b64 = numpy_to_base64_png(rot_img)
    svg_str = regions_to_svg(
        rot_img_b64,
        seg_w,
        seg_h,
        region_1,
        nose_contour,
        region_4,
        ear_contours,
        left_eye_bag,
        right_eye_bag
    )
    svg_b64 = svg_to_base64(svg_str, data_uri=True)
    
    # ==========================
    # 11. PREPARE RESPONSE
    # ==========================
    result = {
        "svg": svg_b64,
        "mask_contours": {}
    }
    
    return result