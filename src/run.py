import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
from zipfile import ZipFile
import numpy as np
from ultralytics import YOLO


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Annotation Generator")
        self.geometry("1080x920")

        self.model = YOLO('../weight/best_segment.pt')
        self.model.fuse()

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

        ttk.Button(self, text="СТАРТ", command=self.start_processing).grid(row=6, column=1, pady=20)

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

            # Create output directory if not exists
            os.makedirs(self.save_path.get(), exist_ok=True)

            # Process files
            self.process_files()

            # Create ZIP archive
            self.create_zip_archive()

            messagebox.showinfo("Успех", "Обработка завершена успешно!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка:\n{str(e)}")

    def process_files(self):
        conf_threshold = self.conf.get()
        iou_threshold = self.iou.get()

        for filename in os.listdir(self.media_folder.get()):
            file_path = os.path.join(self.media_folder.get(), filename)
            base_name = os.path.splitext(filename)[0]

            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                self.process_image(file_path, base_name, conf_threshold, iou_threshold)

            elif filename.lower().endswith((".mp4", ".avi")):
                self.process_video(file_path, base_name, conf_threshold, iou_threshold)

    def process_image(self, img_path, base_name, conf, iou):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))

        results = self.model.predict(img, conf=conf, iou=iou, verbose=False)[0]

        save_img_path = os.path.join(self.save_path.get(), f"{base_name}.png")
        save_label_path = os.path.join(self.save_path.get(), f"{base_name}.txt")

        cv2.imwrite(save_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.write_annotation(save_label_path, results)

    def process_video(self, video_path, base_name, conf, iou):
        cap = cv2.VideoCapture(video_path)
        frame_num = 1

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = cv2.resize(processed_frame, (640, 640))

                # Perform prediction
                results = self.model.predict(processed_frame, conf=conf, iou=iou, verbose=False)[0]

                # Save results
                save_img_path = os.path.join(self.save_path.get(), f"{base_name}_{frame_num:04d}.png")
                save_label_path = os.path.join(self.save_path.get(), f"{base_name}_{frame_num:04d}.txt")

                cv2.imwrite(save_img_path, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                self.write_annotation(save_label_path, results)

                frame_num += 1
        finally:
            cap.release()

    @staticmethod
    def write_annotation(label_path, results):
        with open(label_path, 'w') as f:
            if results.masks is not None:
                for i, mask in enumerate(results.masks):
                    # Check if corresponding box exists
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

                    height, width = mask_int.shape
                    normalized = approx / [width, height]
                    normalized = normalized.flatten().round(6).tolist()

                    f.write(f"{class_id} " + " ".join(map(str, normalized)) + "\n")

    def create_zip_archive(self):
        zip_path = os.path.join(os.path.dirname(self.save_path.get()), self.zip_name.get())

        with ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(self.save_path.get()):
                for file in files:
                    zipf.write(
                        os.path.join(root, file),
                        arcname=os.path.relpath(os.path.join(root, file), self.save_path.get())
                    )


if __name__ == "__main__":
    app = Application()
    app.mainloop()
