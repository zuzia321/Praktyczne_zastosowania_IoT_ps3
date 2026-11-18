import math
from ultralytics import YOLO

MODEL_PATH = "best.pt"
SOURCE = "rtsp://10.99.233.37:8554/cam"
IMGSZ = 1280        # use the same size the model was trained with
CONF = 0.5         # ignore detections below 50% confidence
SIZE_MARGIN = 0.05 # boxes within 5% of each other's area count as equal
MIN = 0            # output range for scale()
MAX = 5


def get_pair(r):
    """Most confident box of each class, as (cx, cy, w, h). None if one is missing."""
    best = {}
    for box in r.boxes:
        c, conf = int(box.cls), float(box.conf)
        if c not in best or conf > best[c][0]:
            best[c] = (conf, box.xywh[0].tolist())
    if len(best) < 2:
        return None
    return best[0][1], best[1][1]


def split_by_size(pair, margin=SIZE_MARGIN):
    """Return (big_class, big_box, small_box), or None if they're about equal."""
    b0, b1 = pair
    a0, a1 = b0[2] * b0[3], b1[2] * b1[3]
    if abs(a0 - a1) <= margin * max(a0, a1):
        return None
    return (0, b0, b1) if a0 > a1 else (1, b1, b0)


def corners(box):
    """Named corners of a (cx, cy, w, h) box."""
    cx, cy, w, h = box
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return {
        "top_left":     (x1, y1),
        "top_right":    (x2, y1),
        "bottom_right": (x2, y2),
        "bottom_left":  (x1, y2),
    }


def angle_from_up(x, y):
    deg = math.degrees(math.atan2(x, -y))
    return deg % 360


def scale(angle, lo, hi):
    """Map an angle in [0, 360) linearly onto [lo, hi)."""
    return lo + (angle % 360) / 360 * (hi - lo)


OPPOSITE = {
    "top_left":     "bottom_right",
    "top_right":    "bottom_left",
    "bottom_right": "top_left",
    "bottom_left":  "top_right",
}


def main():
    model = YOLO(MODEL_PATH)
    for r in model(SOURCE, stream=True, imgsz=IMGSZ, conf=CONF, show=True):
        pair = get_pair(r)
        if not pair:
            print("No detect")
            continue                   # one of the items wasn't detected

        result = split_by_size(pair)
        if not result:
            continue                   # about equal size: do nothing

        big_cls, big, small = result
        center = (big[0], big[1])      # center of the larger box

        small_corners = corners(small)
        near_name, near = min(small_corners.items(), key=lambda kv: math.dist(kv[1], center))
        far_name = OPPOSITE[near_name]
        far = small_corners[far_name]

        # vector from the closest corner to the opposite corner
        vx, vy = far[0] - near[0], far[1] - near[1]
        length = math.hypot(vx, vy)
        angle = math.degrees(math.atan2(vy, vx))  # clockwise, since image y points down
        ux, uy = vx / length, vy / length         # unit vector (direction only)
        from_up = angle_from_up(vx, vy)
        scaled = scale(from_up, MIN, MAX)

        #print(
        #    f"larger: {model.names[big_cls]}, center ({center[0]:.0f}, {center[1]:.0f}) | "
        #    f"closest: {near_name} ({near[0]:.0f}, {near[1]:.0f}) | "
        #    f"opposite: {far_name} ({far[0]:.0f}, {far[1]:.0f}) | "
        #    f"vector ({vx:.0f}, {vy:.0f}), length {length:.0f}px, angle {angle:.0f}°, "
        #    f"unit ({ux:.2f}, {uy:.2f}), angle_from_up {from_up:.0f}°, scaled {scaled:.2f}"
        #)

        print(f"waga: {scaled:.2f}kg")


if __name__ == "__main__":
    main()
