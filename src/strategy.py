import cv2
import numpy as np
from abc import ABC, abstractmethod


class AnnotationStrategy(ABC):
    @abstractmethod
    def process(self, results, label_path, image_size):
        raise NotImplementedError


class SegmentationStrategy(AnnotationStrategy):
    def process(self, results, label_path, image_size):
        with open(label_path, 'w') as f:
            if results.masks is not None:
                for i, mask in enumerate(results.masks):
                    if i >= len(results.boxes):
                        continue

                    class_id = int(results.boxes.cls[i])
                    mask_np = mask.data[0].cpu().numpy()
                    mask_int = (mask_np > 0.5).astype(np.uint8) * 255

                    contours, _ = cv2.findContours(mask_int, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue

                    largest_contour = max(contours, key=cv2.contourArea)
                    epsilon = 0.001 * cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    approx = approx.reshape(-1, 2)

                    width, height = image_size
                    normalized = approx / [width, height]
                    normalized = normalized.flatten().round(6).tolist()

                    f.write(f"{class_id} " + " ".join(map(str, normalized)) + "\n")


class DetectionStrategy(AnnotationStrategy):
    def process(self, results, label_path, image_size):
        width, height = image_size
        with open(label_path, 'w') as f:
            for index in range(len(results.boxes.cls)):
                box = results.boxes.xyxy[index]
                class_id = int(results.boxes.cls[index])
                xmin, ymin, xmax, ymax = box

                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                w = xmax - xmin
                h = ymax - ymin

                x_center_norm = x_center / width
                y_center_norm = y_center / height
                w_norm = w / width
                h_norm = h / height

                line = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                f.write(line)


class OrientedDetectionStrategy(AnnotationStrategy):
    def process(self, results, label_path, image_size):
        pass


# Фабрика для создания стратегий
class AnnotationStrategyFactory:
    @staticmethod
    def create_strategy(model_type):
        strategies = {
            "Segmentation": SegmentationStrategy,
            "Object Detection": DetectionStrategy,
            "Oriented Detection": OrientedDetectionStrategy
        }
        return strategies.get(model_type, DetectionStrategy)()
