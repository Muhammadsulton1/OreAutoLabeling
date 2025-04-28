import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
import torch

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

        ttk.Label(self, text="Вид разметки:").grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
        self.model_type = tk.StringVar()
        model_menu = ttk.Combobox(
            self,
            textvariable=self.model_type,
            values=["Object Detection", "Segmentation", "Oriented Object Box"],
            state="readonly"
        )
        model_menu.grid(row=6, column=1, padx=5, pady=5, sticky=tk.W)
        model_menu.set("Object Detection")

        ttk.Label(self, text="Формат аннотации:").grid(row=7, column=0, padx=5, pady=5, sticky=tk.W)
        self.annotation_format = tk.StringVar()
        format_menu = ttk.Combobox(
            self,
            textvariable=self.annotation_format,
            values=["YOLO", "COCO"],
            state="readonly"
        )
        format_menu.grid(row=7, column=1, padx=5, pady=5, sticky=tk.W)
        format_menu.set("YOLO")

        ttk.Label(self, text="Категории COCO (JSON):").grid(row=8, column=0, padx=5, pady=5, sticky=tk.W)
        self.coco_categories = tk.Text(self, height=5, width=50)
        self.coco_categories.grid(row=8, column=1, padx=5, pady=5)
        self.coco_categories.insert(tk.END, '[{"id": 0, "name": "person"},{"id": 1, "name": "car"}, {}]')  # Пример по умолчанию
        self.coco_categories.grid_remove()

        self.annotation_format.trace_add('write', self.toggle_coco_fields)

        ttk.Label(self, text="Выберите с каким шагом нарезать кадр:").grid(row=9, column=0, padx=5, pady=5, sticky=tk.W)
        self.skip_frame = tk.IntVar(value=0)
        ttk.Entry(self, textvariable=self.skip_frame, width=50).grid(row=9, column=1, padx=5, pady=5)

        ttk.Label(self, text="Model Inference Type:").grid(row=10, column=0, padx=5, pady=5, sticky=tk.W)
        self.model_type_inference = tk.StringVar()
        model_menu = ttk.Combobox(
            self,
            textvariable=self.model_type_inference,
            values=["YOLOV8", "SAM"],
            state="readonly"
        )
        model_menu.grid(row=10, column=1, padx=5, pady=5, sticky=tk.W)
        model_menu.set("YOLOV8")

        ttk.Button(self, text="СТАРТ", command=self.start_processing).grid(row=11, column=1, pady=20)

        # Текущий файл и прогресс
        self.current_file_label = ttk.Label(self, text="")
        self.current_file_label.grid(row=12, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)

        self.progress_bar = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress_bar.grid(row=13, column=0, columnspan=3, padx=5, pady=5)

    def toggle_coco_fields(self, *args):
        if self.annotation_format.get() == "COCO":
            self.coco_categories.grid()
        else:
            self.coco_categories.grid_remove()


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
        finally:
            self.current_file_label.config(text="")
            self.progress_bar["value"] = 0

    def process_files(self):
        conf_threshold = self.conf.get()
        iou_threshold = self.iou.get()

        media_path = self.media_folder.get()
        files_to_process = []
        if os.path.isdir(media_path):
            for filename in os.listdir(media_path):
                file_path = os.path.join(media_path, filename)
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".mp4", ".avi")):
                    files_to_process.append(file_path)
        elif os.path.isfile(media_path) and media_path.lower().endswith((".jpg", ".jpeg", ".png", ".mp4", ".avi")):
            files_to_process.append(media_path)

        total_files = len(files_to_process)
        if total_files == 0:
            raise ValueError("Нет файлов для обработки")

        for i, file_path in enumerate(files_to_process):
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            self.current_file_label.config(text=f"Обработка файла {i+1} из {total_files}: {os.path.basename(file_path)}")
            self.progress_bar["value"] = 0
            self.update_idletasks()

            if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
                self.process_image(file_path, base_name, conf_threshold, iou_threshold)
                self.progress_bar["value"] = 100
                self.update_idletasks()
            elif file_path.lower().endswith((".mp4", ".avi")):
                self.process_video(file_path, base_name, conf_threshold, iou_threshold)

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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1

        frame_num = 0
        skip_frame = self.skip_frame.get() or 1

        self.progress_bar["maximum"] = 100
        self.progress_bar["value"] = 0
        self.update_idletasks()

        try:
            while True:
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

                progress = (frame_num / total_frames) * 100
                self.progress_bar["value"] = progress
                self.update_idletasks()
                frame_num += 1

            self.progress_bar["value"] = 100
            self.update_idletasks()
        finally:
            cap.release()


if __name__ == "__main__":
    app = Application()
    app.mainloop()
