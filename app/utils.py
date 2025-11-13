# app/utils.py
import base64
import io
import html
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np
import cv2

def base64_to_pil(b64: str) -> Image.Image:
    """
    Accepts either raw base64 or data URI (data:image/png;base64,...)
    Returns a PIL Image in RGBA mode.
    """
    header, _, payload = b64.partition(",")
    if payload == "":
        payload = header
    data = base64.b64decode(payload)
    return Image.open(io.BytesIO(data)).convert("RGBA")

def pil_to_numpy_rgba(img: Image.Image) -> np.ndarray:
    """Return H x W x 4 numpy array (RGBA)."""
    return np.array(img)

def extract_contours_from_segmap(np_img: np.ndarray) -> Dict[str, List[List[Tuple[int, int]]]]:
    """Extract contours for each unique RGB region in segmentation map."""
    h, w, c = np_img.shape
    if c < 3:
        raise ValueError("segmentation_map must have RGB(A) channels")

    rgb = np_img[..., :3]
    flat = rgb.reshape(-1, 3)
    colors, inverse = np.unique(flat, axis=0, return_inverse=True)

    contours_by_color = {}
    key_idx = 1
    for idx_color, color in enumerate(colors):
        if (color == [0, 0, 0]).all() or (color == [255, 255, 255]).all():
            continue
        mask = (inverse.reshape(h, w) == idx_color).astype("uint8") * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cont_list = []
        for c_ in cnts:
            pts = [(int(p[0][0]), int(p[0][1])) for p in c_]
            if len(pts) > 2:
                cont_list.append(pts)
        if cont_list:
            key = f"{key_idx}"
            contours_by_color[key] = cont_list
            key_idx += 1
    return contours_by_color

def contours_to_svg(
    contours_by_color: Dict[str, List[List[Tuple[int, int]]]],
    width: int,
    height: int,
    orig_image_b64: str = None,
    preserve_aspect: bool = True
) -> str:
    """
    Builds SVG containing:
    - Embedded original face image (optional)
    - Blue translucent overlay regions drawn as contours.
    """
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}"'
    ]
    if preserve_aspect:
        svg_parts[0] += ' preserveAspectRatio="xMidYMid slice"'
    svg_parts[0] += ">"

    # Embed original image
    if orig_image_b64:
        header, _, payload = orig_image_b64.partition(",")
        if payload == "":
            payload = header
            data_uri = f"data:image/png;base64,{payload}"
        else:
            data_uri = orig_image_b64
        data_uri_escaped = html.escape(data_uri, quote=True)
        svg_parts.append(
            f'<image x="0" y="0" width="{width}" height="{height}" '
            f'xlink:href="{data_uri_escaped}" />'
        )

    # Draw overlays
    for region_id, contours in contours_by_color.items():
        for contour in contours:
            if not contour:
                continue
            d_parts = [("M" if i == 0 else "L") + f"{x} {y}" for i, (x, y) in enumerate(contour)]
            d_parts.append("Z")
            path_d = " ".join(d_parts)
            svg_parts.append(
                f'<path data-region="{region_id}" d="{path_d}" '
                f'fill="#ADD8E6" fill-opacity="0.45" '
                f'stroke="#5DADE2" stroke-width="2" stroke-dasharray="6 3" />'
            )

    svg_parts.append("</svg>")
    return "".join(svg_parts)

def svg_to_base64(svg_str: str, data_uri: bool = True) -> str:
    """Return base64-encoded SVG, optionally as a data URI."""
    b = svg_str.encode("utf-8")
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}" if data_uri else b64
