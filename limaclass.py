import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import cv2
import joblib
import numpy as np
import pywt
import matplotlib.pyplot as plt

from skimage.filters import roberts
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ================= LOAD MODEL =================
model, class_names = joblib.load("prediksi-bunga-svm.pkl")

# ================= FEATURE EXTRACTION =================
def preprocess(img):
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def color_histogram(img, bins=32):
    features = []
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [bins], [0,256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return np.array(features)


def extract_features(img):
    img, gray = preprocess(img)

    # Color histogram (96)
    hist = color_histogram(img)

    # Haar wavelet (8)
    cA, (cH, cV, cD) = pywt.dwt2(gray, 'haar')
    haar_feats = np.array([
        np.mean(cA), np.var(cA),
        np.mean(cH), np.var(cH),
        np.mean(cV), np.var(cV),
        np.mean(cD), np.var(cD)
    ])

    # Robert Cross (4)
    edge = roberts(gray)
    roberts_feats = np.array([
        np.mean(edge),
        np.var(edge),
        np.max(edge),
        np.sum(edge)
    ])

    return np.concatenate([hist, haar_feats, roberts_feats])


# ================= GUI FUNCTION =================
def open_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        return

    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("Error", "Gambar tidak dapat dibaca")
        return

    img_resized, gray = preprocess(img)

    # ===== ORIGINAL IMAGE =====
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((260, 260))
    img_tk = ImageTk.PhotoImage(img_pil)
    original_label.config(image=img_tk)
    original_label.image = img_tk

    # ===== ROBERT CROSS =====
    edge = roberts(gray)
    ax_roberts.clear()
    ax_roberts.imshow(edge, cmap='gray')
    ax_roberts.set_title("Robert Cross")
    ax_roberts.axis('off')
    canvas_roberts.draw()

    # ===== HAAR WAVELET =====
    cA, _ = pywt.dwt2(gray, 'haar')
    ax_haar.clear()
    ax_haar.imshow(cA, cmap='gray')
    ax_haar.set_title("Haar Wavelet (cA)")
    ax_haar.axis('off')
    canvas_haar.draw()

    # ===== COLOR HISTOGRAM =====
    ax_hist.clear()
    for i, col in enumerate(('b', 'g', 'r')):
        hist = cv2.calcHist([img_resized], [i], None, [32], [0,256])
        ax_hist.plot(hist, color=col)
    ax_hist.set_title("Color Histogram (RGB)")
    ax_hist.set_xlabel("Pixel Value")
    ax_hist.set_ylabel("Frequency")
    canvas_hist.draw()

    # ===== PREDICTION =====
    features = extract_features(img).reshape(1, -1)
    pred = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(features)[0][pred] * 100
        prediction_label.config(text=f"Prediksi : {class_names[pred]}")
        confidence_label.config(text=f"Confidence : {prob:.2f} %")
    else:
        prediction_label.config(text=f"Prediksi : {class_names[pred]}")
        confidence_label.config(text="Confidence : -")


# ================= GUI LAYOUT =================
root = tk.Tk()
root.title("Klasifikasi Bunga – Support Vector Machine (SVM)")
root.geometry("1100x850")
root.resizable(True, True)

# ===== TITLE =====
title = tk.Label(
    root,
    text="Klasifikasi Bunga Support Vector Machine (SVM)",
    font=("Arial", 16, "bold")
)
title.pack(pady=10)

btn = tk.Button(
    root,
    text="Pilih Gambar",
    command=open_image,
    font=("Arial", 12),
    width=20
)
btn.pack(pady=5)

# ===== MAIN FRAME =====
main_frame = tk.Frame(root)
main_frame.pack(pady=10)

# ===== ORIGINAL IMAGE =====
frame_original = tk.LabelFrame(main_frame, text="Original Image", padx=5, pady=5)
frame_original.grid(row=0, column=0, padx=10, pady=10)
original_label = tk.Label(frame_original)
original_label.pack()

# ===== ROBERT =====
frame_roberts = tk.LabelFrame(main_frame, text="Edge Detection (Robert Cross)", padx=5, pady=5)
frame_roberts.grid(row=0, column=1, padx=10, pady=10)
fig_roberts, ax_roberts = plt.subplots(figsize=(2.6,2.6))
canvas_roberts = FigureCanvasTkAgg(fig_roberts, master=frame_roberts)
canvas_roberts.get_tk_widget().pack()

# ===== HAAR =====
frame_haar = tk.LabelFrame(main_frame, text="Haar Wavelet", padx=5, pady=5)
frame_haar.grid(row=1, column=0, padx=10, pady=10)
fig_haar, ax_haar = plt.subplots(figsize=(2.6,2.6))
canvas_haar = FigureCanvasTkAgg(fig_haar, master=frame_haar)
canvas_haar.get_tk_widget().pack()

# ===== HISTOGRAM =====
frame_hist = tk.LabelFrame(main_frame, text="Color Histogram", padx=5, pady=5)
frame_hist.grid(row=1, column=1, padx=10, pady=10)
fig_hist, ax_hist = plt.subplots(figsize=(2.6,2.6))
canvas_hist = FigureCanvasTkAgg(fig_hist, master=frame_hist)
canvas_hist.get_tk_widget().pack()

# ===== RESULT =====
result_frame = tk.Frame(root)
result_frame.pack(pady=15)

prediction_label = tk.Label(
    result_frame,
    text="Prediksi : -",
    font=("Arial", 15, "bold")
)
prediction_label.pack()

confidence_label = tk.Label(
    result_frame,
    text="Confidence : -",
    font=("Arial", 13)
)
confidence_label.pack()

root.mainloop()