# BSChen
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models import YOLO


def get_yolo_model() -> YOLO:
    """ Get YOLOv8 model """
    model = YOLO("yolov8l.pt")
    return model


def yolo_forward(model: YOLO, x: torch.Tensor) -> tuple[torch.Tensor, tuple]:
    """ Forward function for YOLOv8 model """
    # x shape: (B, 3, H, W), range: (arbitrary)
    # y shape: (B, N, 6), range: [0, 1], last dim: [x1, y1, x2, y2, score, class]
    device = x.device
    model.to(device)

    # Preprocess value range and channel
    x = x - x.min()
    x = x / x.max()  # range: [0, 1]
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)  # gray to rgb

    # Prepocess size
    new_shape = (640, 640)
    x = F.interpolate(x, size=new_shape, mode='bilinear', align_corners=True)

    # Inference, no terminal output
    model.eval()
    with torch.no_grad():
        results = model.predict(x, verbose=False)

    # Collect boxes
    boxes = [r.boxes for r in results]
    return boxes, (640, 640)


def get_pedestrian_heatmap(
        boxes: list,
        img_size: tuple,
        map_size: tuple=None,
        threshold: float=0.5,
        device: torch.device=torch.device('cpu')
    ) -> torch.Tensor:
    """ Get pedestrian heatmap from YOLOv8 bounding boxes """
    B = len(boxes)
    H, W = img_size
    map_size = map_size if map_size is not None else (H, W)

    heatmap = torch.zeros((B, 1, map_size[0], map_size[1]), device=device)
    kernel = make_gaussian_kernel(size=9, sigma=3.0).to(device)

    for b, box in enumerate(boxes):
        # Skip if no box detected
        if len(box) == 0:
            continue

        xyxy = box.xyxy.cpu()   # shape: (N,4)
        conf = box.conf.cpu()   # shape: (N,)
        cls  = box.cls.cpu()    # shape: (N,)

        N = len(xyxy)
        for n in range(N):
            if cls[n] == 0 and conf[n] >= threshold:  # class 0 = pedestrian
                # x1, y1, x2, y2 = (xyxy[n] * torch.tensor([map_size[1]/W, map_size[0]/H, map_size[1]/W, map_size[0]/H])).int()
                # w, h = x2 - x1, y2 - y1
                # if w <= 0 or h <= 0:
                #     continue
                # # Fill rectangle
                # area = w * h
                # heatmap[b, 0, y1:y2, x1:x2] += 1 / area

                center = (
                    int((xyxy[n, 0] + xyxy[n, 2]) / 2 / W * map_size[1]),
                    int((xyxy[n, 1] + xyxy[n, 3]) / 2 / H * map_size[0])
                )
                x1 = max(0, center[0] - kernel.shape[1] // 2)
                y1 = max(0, center[1] - kernel.shape[0] // 2)
                x2 = min(map_size[1], center[0] + kernel.shape[1] // 2 + 1)
                y2 = min(map_size[0], center[1] + kernel.shape[0] // 2 + 1)
                kx1 = max(0, kernel.shape[1] // 2 - center[0])
                ky1 = max(0, kernel.shape[0] // 2 - center[1])
                kx2 = kx1 + (x2 - x1)
                ky2 = ky1 + (y2 - y1)

                heatmap[b, 0, y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]
                heatmap[b, 0, y1:y2, x1:x2] = torch.clamp(heatmap[b, 0, y1:y2, x1:x2], 0, 1)
    return heatmap


def make_gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """ Make a 2D Gaussian kernel """
    x = torch.arange(0, size, 1, dtype=torch.float32)
    y = x[:, None]
    x0 = y0 = size // 2

    kernel = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


if __name__ == "__main__":
    # Show class names
    model = get_yolo_model()
    print(model.names)  # should print ['0']
