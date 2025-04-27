import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
import torch

from ultralytics import YOLO
from src.strategy import AnnotationStrategyFactory
from src.utils import download_model
from src.inference import InferenceModel


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Annotation Generator")
        self.geometry("1080x920")

        self.model_type_inference = tk.StringVar(value="YOLOV8")
        if torch.cuda.is_available():
            messagebox.showinfo("DEVICE", "Обработка моделей будет на ГПУ")
            weight = download_model('cuda')

            # self.model = YOLO(weight)
            # self.model.fuse()

            self.model = InferenceModel(self.model_type_inference.get())

        else:
            messagebox.showinfo("DEVICE", "Обработка моделей будет выполняться на процессоре")

        self.annotation_strategy = None
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self, text="Папка с материалом:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.media_folder = tk.StringVar()
        ttk.Entry(self, textvariable=self.media_folder, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self, text="Обзор", command=self.browse_media_folder).grid(row=1, column=2, padx=5, pady=5)

        ttk.Label(self, text="Куда сохранить:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.save_path = tk.StringVar()
        ttk.Entry(self, textvariable=self.save_path, width=50).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(self, text="Обзор", command=self.browse_save_folder).grid(row=2, column=2, padx=5, pady=5)

        ttk.Label(self, text="Имя ZIP-архива:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.zip_name = tk.StringVar(value="annotations.zip")
        ttk.Entry(self, textvariable=self.zip_name, width=50).grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(self, text="Порог уверенности (conf):").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.conf = tk.DoubleVar(value=0.2)
        ttk.Scale(self, variable=self.conf, from_=0.0, to=1.0, orient=tk.HORIZONTAL).grid(
            row=4, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)

        ttk.Label(self, text="Порог IoU:").grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        self.iou = tk.DoubleVar(value=0.2)
        ttk.Scale(self, variable=self.iou, from_=0.0, to=1.0, orient=tk.HORIZONTAL).grid(
            row=5, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)

        ttk.Label(self, text="Annotation Type:").grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
        self.model_type = tk.StringVar()
        model_menu = ttk.Combobox(
            self,
            textvariable=self.model_type,
            values=["Object Detection", "Segmentation", "Oriented Object Box"],
            state="readonly"
        )
        model_menu.grid(row=6, column=1, padx=5, pady=5, sticky=tk.W)
        model_menu.set("Object Detection")

        ttk.Label(self, text="Выберите с каким шагом нарезать кадр:").grid(row=7, column=0, padx=5, pady=5, sticky=tk.W)
        self.skip_frame = tk.IntVar(value=0)
        ttk.Entry(self, textvariable=self.skip_frame, width=50).grid(row=7, column=1, padx=5, pady=5)

        ttk.Label(self, text="Model Inference Type:").grid(row=8, column=0, padx=5, pady=5, sticky=tk.W)
        self.model_type_inference = tk.StringVar()
        model_menu = ttk.Combobox(
            self,
            textvariable=self.model_type_inference,
            values=["YOLOV8", "SAM"],
            state="readonly"
        )
        model_menu.grid(row=8, column=1, padx=5, pady=5, sticky=tk.W)
        model_menu.set("YOLOV8")

        ttk.Button(self, text="СТАРТ", command=self.start_processing).grid(row=9, column=1, pady=20)

    def browse_media_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.media_folder.set(folder)

    def browse_save_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.save_path.set(folder)

    def start_processing(self):
        try:
            if not self.media_folder.get() or not self.save_path.get():
                raise ValueError("Пожалуйста, укажите исходную папку и папку назначения")

            os.makedirs(self.save_path.get(), exist_ok=True)
            self.annotation_strategy = AnnotationStrategyFactory.create_strategy(self.model_type.get())
            self.process_files()

            messagebox.showinfo("Успех", "Обработка завершена успешно!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка:\n{str(e)}")

    def process_files(self):
        conf_threshold = self.conf.get()
        iou_threshold = self.iou.get()

        if os.path.isdir(self.media_folder.get()):
            for filename in os.listdir(self.media_folder.get()):
                file_path = os.path.join(self.media_folder.get(), filename)
                base_name = os.path.splitext(filename)[0]

                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.process_image(file_path, base_name, conf_threshold, iou_threshold)
                elif filename.lower().endswith((".mp4", ".avi")):
                    self.process_video(file_path, base_name, conf_threshold, iou_threshold)

        elif os.path.isfile(self.media_folder.get()):
            base_name = os.path.splitext(self.media_folder.get())[0]
            if self.model_type.get().endswith((".mp4", ".avi")):
                self.process_video(self.media_folder.get(), base_name, conf_threshold, iou_threshold)
            elif self.model_type.get().endswith((".jpg", ".jpeg", ".png")):
                self.process_image(self.media_folder.get(), base_name, conf_threshold, iou_threshold)
            else:
                raise ValueError("Не поддерживаемый формат медиа")

    def process_image(self, img_path, base_name, conf, iou):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))

        results = self.model.process(img, conf=conf, iou=iou)[0]

        save_img_path = os.path.join(self.save_path.get(), f"{base_name}.png")
        save_label_path = os.path.join(self.save_path.get(), f"{base_name}.txt")

        cv2.imwrite(save_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.annotation_strategy.process(results, save_label_path, (640, 640))

    def process_video(self, video_path, base_name, conf, iou):
        cap = cv2.VideoCapture(video_path)
        frame_num = 0
        skip_frame = self.skip_frame.get()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % skip_frame == 0:
                    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed_frame = cv2.resize(processed_frame, (640, 640))

                    results = self.model.process(processed_frame, conf=conf, iou=iou)[0]

                    save_img_path = os.path.join(self.save_path.get(), f"{base_name}_{frame_num:04d}.png")
                    save_label_path = os.path.join(self.save_path.get(), f"{base_name}_{frame_num:04d}.txt")

                    cv2.imwrite(save_img_path, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                    self.annotation_strategy.process(results, save_label_path, (640, 640))

                frame_num += 1
        finally:
            cap.release()


if __name__ == "__main__":
    app = Application()
    app.mainloop()
