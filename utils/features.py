import cv2

def calculate_blur(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_features(frame, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    results = model(frame, verbose=False)[0]
    boxes = []

    for r in results.boxes:
        cls = int(r.cls[0].item())
        if cls == 0:
            x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
            boxes.append((x1, y1, x2 - x1, y2 - y1))

    crowd_count = len(boxes)
    occlusion_ratio = 0.7 if crowd_count > 4 else 0.4
    ground_area = 120.0

    return [crowd_count, blur_score, occlusion_ratio, ground_area], boxes

