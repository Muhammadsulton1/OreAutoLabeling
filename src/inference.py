from ultralytics import YOLO
from typing import Any, Optional
import logging


class InferenceModel:
    def __init__(self, model_type: str, model_path: Optional[str] = None) -> None:
        self.model_type = model_type.lower()  # Приводим к нижнему регистру

        if model_path is None:
            if self.model_type == "yolov8":  # Проверяем в нижнем регистре
                model_path = '../weight/best_segment.pt'
            elif self.model_type == "sam":
                raise ValueError("SAM пока не реализован")

        if self.model_type == "yolov8":  # Проверяем в нижнем регистре
            self.model = self._load_yolo(model_path)
        elif self.model_type == "sam":
            self.model = self._load_sam(model_path)
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}. Доступные варианты: 'yolov8' или 'sam'.")

    def process(self, frame: Any, conf=0.5, iou=0.4) -> Any:
        """
        Обработка кадра
        """
        if not self.model:
            raise RuntimeError("Модель не была корректно инициализирована")

        try:
            return self.model.predict(frame, conf=conf, iou=iou, verbose=False)
        except Exception as e:
            logging.error(f"Error processing frame: {str(e)}")
            raise RuntimeError(f"Failed to process frame: {str(e)}") from e

    @staticmethod
    def _load_yolo(model_path: str) -> YOLO:
        """
        Загружает и настраивает модель YOLO.

        Args:
            model_path (str): Путь к файлу с весами модели

        Returns:
            YOLO: Экземпляр модели YOLO

        Raises:
            RuntimeError: Вызывается при ошибках загрузки модели
        """
        try:
            model = YOLO(model_path)
            model.fuse()
            logging.info(f"Loaded YOLO model from {model_path}")
            return model
        except Exception as e:
            logging.error(f"Ошибка обработки кадра: {str(e)}")
            raise RuntimeError(f"Ошибка обработки кадра: {str(e)}") from e

    @staticmethod
    def _load_sam(model_path: str) -> None:
        """
        Заглушка для загрузки модели SAM (Segment Anything Model).

        Args:
            model_path (str): Путь к файлу с весами модели

        Raises:
            NotImplementedError: Всегда вызывает исключение, так как реализация не завершена
        """
        raise NotImplementedError("SAM model support is not implemented yet")
