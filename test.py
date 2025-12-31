import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import cv2
import joblib
import numpy as np
import pywt
from skimage.filters import roberts
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



# ================= LOAD MODEL =================
model, class_names = joblib.load("svm_flower_gui.pkl")


# ================= FEATURE EXTRACTION =================
def preprocess(img):
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def color_histogram(img, bins=32):
    features = []
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return np.array(features)


def haar_wavelet(gray):
    cA, (cH, cV, cD) = pywt.dwt2(gray, 'haar')
    feats = []
    for m in [cA, cH, cV, cD]:
        feats.append(np.mean(m))
        feats.append(np.var(m))
    return np.array(feats)


def roberts_cross(gray):
    edge = roberts(gray)
    return np.array([
        np.mean(edge),
        np.var(edge),
        np.max(edge),
        np.sum(edge)
    ])


def extract_features(img):
    img, gray = preprocess(img)
    return np.concatenate([
        color_histogram(img),
        haar_wavelet(gray),
        roberts_cross(gray)
    ])

def show_roberts(gray):
    edge = roberts(gray)

    plt.figure("Robert Cross")
    plt.imshow(edge, cmap='gray')
    plt.title("Robert Cross Edge Detection")
    plt.axis('off')
    plt.show()

def show_haar(gray):
    cA, (cH, cV, cD) = pywt.dwt2(gray, 'haar')

    plt.figure("Haar Wavelet")
    plt.imshow(cA, cmap='gray')
    plt.title("Haar Wavelet (Approximation)")
    plt.axis('off')
    plt.show()

def show_color_histogram(img):
    colors = ('b', 'g', 'r')
    plt.figure("Color Histogram")

    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [32], [0,256])
        plt.plot(hist, color=col)

    plt.title("Color Histogram (RGB)")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.show()


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

    # ====== TAMPILKAN VISUAL ======
    show_roberts(gray)
    show_haar(gray)
    show_color_histogram(img_resized)

    # ====== PREDIKSI ======
    features = extract_features(img).reshape(1, -1)
    pred = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(features)[0][pred] * 100
        result_text.set(f"Prediksi: {class_names[pred]} ({prob:.2f}%)")
    else:
        result_text.set(f"Prediksi: {class_names[pred]}")

    # ====== TAMPILKAN GAMBAR ASLI ======
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img_pil)

    image_label.config(image=img_tk)
    image_label.image = img_tk

# ================= GUI LAYOUT =================
root = tk.Tk()
root.title("Flower Classification (SVM)")
root.geometry("400x500")

btn = tk.Button(root, text="Pilih Gambar", command=open_image, font=("Arial", 12))
btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Arial", 14))
result_label.pack(pady=20)

root.mainloop()