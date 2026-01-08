import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter
import cv2
import numpy as np
import pywt
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

# --- BAGIAN 1: FUNGSI EKSTRAKSI FITUR (TETAP) ---
def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_roberts_cross(image_gray):
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    img_float = image_gray.astype('float')
    roberts_x = cv2.filter2D(img_float, -1, kernel_x)
    roberts_y = cv2.filter2D(img_float, -1, kernel_y)
    magnitude = np.sqrt(np.square(roberts_x) + np.square(roberts_y))
    return np.mean(magnitude), np.std(magnitude), magnitude

def extract_haar_texture(image_gray):
    coeffs = pywt.dwt2(image_gray, 'haar')
    LL, (LH, HL, HH) = coeffs
    features = []
    for subband in [LH, HL, HH]:
        features.append(np.mean(np.abs(subband)))
        features.append(np.std(subband))
    return np.array(features), HH

# --- BAGIAN 2: LOGIKA GUI (LAYOUT DIPERBAIKI) ---

class FlowerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Klasifikasi Bunga: Rose dan Sunflower")
        self.root.geometry("1100x750") # Ukuran pas untuk 1 layar
        
        # Agar window bisa ditutup sempurna
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Load Model
        try:
            data = joblib.load('svm_bunga_model.pkl')
            self.model = data['model']
            self.classes = data['classes']
            print("Model berhasil dimuat.")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat model.\nError: {e}")
            self.on_closing()
            return

        self.setup_ui()

    def on_closing(self):
        self.root.destroy()
        sys.exit()

    def setup_ui(self):
        # 1. Container Utama (Frame)
        # expand=True dan fill="both" akan membuat frame ini mengisi seluruh window
        # dan isinya akan mudah diposisikan di tengah.
        self.main_container = tk.Frame(self.root)
        self.main_container.pack(expand=True, fill="both", padx=20, pady=20)

        # --- Bagian Atas: Tombol & Hasil ---
        top_frame = tk.Frame(self.main_container)
        top_frame.pack(pady=(0, 20)) # Jarak ke bawah

        btn_upload = tk.Button(top_frame, text="Upload Gambar Bunga", command=self.upload_image, 
                               font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", padx=20, pady=8)
        btn_upload.pack()

        self.lbl_result = tk.Label(top_frame, text="Hasil: Menunggu...", font=("Arial", 18, "bold"), fg="#333")
        self.lbl_result.pack(pady=10)

        # --- Bagian Visualisasi (Grid) ---
        self.viz_frame = tk.Frame(self.main_container)
        self.viz_frame.pack(expand=True) # expand=True PENTING agar frame ini diam di tengah layar vertical/horizontal

        # Konfigurasi Grid agar Kolom Rata Tengah
        # Memberikan bobot (weight) yang sama pada setiap kolom (0,1,2)
        # agar spasi terbagi rata.
        self.viz_frame.columnconfigure(0, weight=1)
        self.viz_frame.columnconfigure(1, weight=1)
        self.viz_frame.columnconfigure(2, weight=1)

        # Baris 1: Gambar Asli (Di tengah, gabung 3 kolom)
        tk.Label(self.viz_frame, text="Gambar Asli (Input)", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=3, pady=(0,5))
        
        self.panel_original = tk.Label(self.viz_frame, bg="#eee", width=30, height=15, relief="sunken")
        # columnspan=3 membuat dia mengambil lebar 3 kolom, jadi otomatis di tengah
        self.panel_original.grid(row=1, column=0, columnspan=3, padx=10, pady=(0, 20))
        
        # Baris 2: Judul Fitur
        font_style = ("Arial", 9)
        tk.Label(self.viz_frame, text="Histogram Warna (HSV)", font=font_style).grid(row=2, column=0, pady=5)
        tk.Label(self.viz_frame, text="Robert Cross (Tepi)", font=font_style).grid(row=2, column=1, pady=5)
        tk.Label(self.viz_frame, text="Haar Wavelet (Tekstur)", font=font_style).grid(row=2, column=2, pady=5)

        # Baris 3: Gambar Fitur
        # Histogram
        self.panel_hist = tk.Frame(self.viz_frame, bg="white", width=250, height=200, relief="sunken", borderwidth=1)
        self.panel_hist.grid(row=3, column=0, padx=15, pady=5)
        self.panel_hist.pack_propagate(False) # Agar ukuran tidak menyusut

        # Robert Cross
        self.panel_robert = tk.Label(self.viz_frame, bg="#eee", width=25, height=12, relief="sunken")
        self.panel_robert.grid(row=3, column=1, padx=15, pady=5)

        # Haar Wavelet
        self.panel_haar = tk.Label(self.viz_frame, bg="#eee", width=25, height=12, relief="sunken")
        self.panel_haar.grid(row=3, column=2, padx=15, pady=5)

    def display_image(self, img_array, widget, size, is_gray=False, apply_sharpen=False):
        img_pil = Image.fromarray(img_array)
        img_pil = img_pil.resize(size, Image.Resampling.LANCZOS)
        
        if apply_sharpen:
            img_pil = img_pil.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        img_tk = ImageTk.PhotoImage(img_pil)
        widget.configure(image=img_tk, text="", width=0, height=0) 
        widget.image = img_tk

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path: return
        self.process_image(file_path)

    def process_image(self, file_path):
        img = cv2.imread(file_path)
        img_processing = cv2.resize(img, (128, 128))
        img_rgb_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_processing = cv2.cvtColor(img_processing, cv2.COLOR_BGR2GRAY)

        # --- UPDATE UKURAN VISUALISASI ---
        
        # 1. Gambar Asli (300x300)
        self.display_image(img_rgb_display, self.panel_original, size=(300, 300), apply_sharpen=True)

        # 2. Histogram
        feat_hist = extract_color_histogram(img_processing)
        self.plot_histogram(img_rgb_display)

        # 3. Robert Cross (200x200)
        mean_rc, std_rc, viz_rc = extract_roberts_cross(gray_processing)
        viz_rc_show = cv2.normalize(viz_rc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.display_image(viz_rc_show, self.panel_robert, size=(200, 200), is_gray=True)

        # 4. Haar Wavelet (200x200)
        feat_haar, viz_haar = extract_haar_texture(gray_processing)
        viz_haar_show = cv2.normalize(np.abs(viz_haar), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.display_image(viz_haar_show, self.panel_haar, size=(200, 200), is_gray=True)

        # Prediksi
        global_feature = np.hstack([feat_hist, [mean_rc, std_rc], feat_haar])
        try:
            prediction_idx = self.model.predict([global_feature])[0]
            result_text = f"Prediksi: {prediction_idx}"
            
            # Warna Label
            color = "blue"
            txt = str(prediction_idx).lower()
            if "sunflower" in txt: color = "#FFC107" # Gold/Yellowish
            elif "rose" in txt: color = "#E91E63" # Pink
                
        except Exception as e:
            result_text = f"Error: {e}"
            color = "red"
        self.lbl_result.config(text=result_text, fg=color)

    def plot_histogram(self, image_rgb):
        for widget in self.panel_hist.winfo_children():
            widget.destroy()

        # Ukuran Plot Disesuaikan (3.5 x 2.5 inch)
        fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=100)
        colors = ('r', 'g', 'b')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
            ax.plot(hist, color=col, linewidth=1.5)
            ax.set_xlim([0, 256])
        
        ax.set_title("Color Histogram", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.panel_hist)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = FlowerApp(root)
    root.mainloop()