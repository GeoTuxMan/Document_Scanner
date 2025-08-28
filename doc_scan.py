import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageOps

# Optional: OCR and PDF generation
try:
    import pytesseract  # pip install pytesseract
except Exception:
    pytesseract = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib.utils import ImageReader
except Exception:
    pdf_canvas = None

#############################################
# Geometry helpers for perspective transform
#############################################

def order_points(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2) array
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute width of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    # compute height of the new image
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


#############################################
# Document detection & enhancement
#############################################

def find_document_contour(image_bgr: np.ndarray):
    # Resize for speed
    ratio = image_bgr.shape[0] / 800.0
    img = cv2.resize(image_bgr, (int(image_bgr.shape[1]/ratio), 800))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 75, 200)
    # close gaps
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.erode(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            # scale back to original size
            pts = approx.reshape(4, 2).astype(np.float32)
            pts *= ratio
            return pts
    return None


def enhance_document(warped_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold to mimic a scanned doc
    scanned = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return cv2.cvtColor(scanned, cv2.COLOR_GRAY2BGR)


#############################################
# Tkinter Application
#############################################

class DocumentScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Scanner")
        self.root.geometry("1000x700")

        self.original_bgr = None
        self.current_bgr = None
        self.display_img = None  # for Tk
        self.ocr_text = ""
        self.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # path to tesseract executable (optional Windows)

        self._build_ui()

    def _build_ui(self):
        # Top controls
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=10, pady=8)

        tk.Button(top, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Auto Detect + Scan", command=self.auto_scan).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Enhance (BW)", command=self.enhance_only).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="OCR Extract Text", command=self.run_ocr).pack(side=tk.LEFT, padx=5)

        tk.Button(top, text="Save Image", command=self.save_image).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Save PDF", command=self.save_pdf).pack(side=tk.LEFT, padx=5)

        # Optional: set tesseract path (Windows)
        tk.Button(top, text="Set Tesseract Path", command=self.set_tesseract_path).pack(side=tk.RIGHT, padx=5)

        # Canvas & OCR output
        mid = tk.Frame(self.root)
        mid.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(mid, width=680, height=520, bg="#333")
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        right = tk.Frame(mid)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=10, pady=10)

        tk.Label(right, text="OCR Output:").pack(anchor="w")
        self.text_box = tk.Text(right, width=40, height=28)
        self.text_box.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status = tk.StringVar(value="Load an image to begin…")
        tk.Label(self.root, textvariable=self.status, anchor="w").pack(fill=tk.X, padx=10, pady=(0,8))

    def set_status(self, msg: str):
        self.status.set(msg)
        self.root.update_idletasks()

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Failed to read image.")
            return
        self.original_bgr = img
        self.current_bgr = img.copy()
        self.ocr_text = ""
        self.text_box.delete("1.0", tk.END)
        self.render_on_canvas(self.current_bgr)
        self.set_status(f"Loaded: {os.path.basename(path)}  |  {img.shape[1]}x{img.shape[0]}")

    def render_on_canvas(self, bgr: np.ndarray):
        # Fit to canvas while preserving aspect
        canvas_w = self.canvas.winfo_width() or 680
        canvas_h = self.canvas.winfo_height() or 520
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img = ImageOps.contain(pil_img, (canvas_w, canvas_h))
        self.display_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w//2, canvas_h//2, image=self.display_img)

    def auto_scan(self):
        if self.original_bgr is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return
        self.set_status("Detecting document edges…")
        pts = find_document_contour(self.original_bgr)
        if pts is None:
            messagebox.showinfo("Not found", "No 4-corner document contour detected. Applying enhancement only.")
            scanned = enhance_document(self.original_bgr)
        else:
            warped = four_point_transform(self.original_bgr, pts)
            scanned = enhance_document(warped)
        self.current_bgr = scanned
        self.render_on_canvas(self.current_bgr)
        self.set_status("Scan complete.")

    def enhance_only(self):
        if self.original_bgr is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return
        self.current_bgr = enhance_document(self.original_bgr)
        self.render_on_canvas(self.current_bgr)
        self.set_status("Enhanced (B/W).")

    def run_ocr(self):
        if self.current_bgr is None:
            messagebox.showwarning("No image", "Please load and process an image first.")
            return
        if pytesseract is None:
            messagebox.showerror("Missing dependency", "Install OCR support: pip install pytesseract\nAlso install Tesseract engine: https://tesseract-ocr.github.io/")
            return
        # Configure tesseract cmd if provided (Windows)
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        gray = cv2.cvtColor(self.current_bgr, cv2.COLOR_BGR2GRAY)
        # OCR using English by default (you can add languages if installed)
        self.set_status("Running OCR…")
        try:
            text = pytesseract.image_to_string(gray)
        except Exception as e:
            messagebox.showerror("OCR error", str(e))
            return
        self.ocr_text = text
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert(tk.END, text)
        self.set_status("OCR complete.")

    def save_image(self):
        if self.current_bgr is None:
            messagebox.showwarning("No image", "Nothing to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf")])
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        rgb = cv2.cvtColor(self.current_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        try:
            if ext == ".pdf":
                pil.save(path, "PDF")
            else:
                pil.save(path)
        except Exception as e:
            messagebox.showerror("Save error", str(e))
            return
        messagebox.showinfo("Saved", f"Saved to {path}")

    def save_pdf(self):
        if self.current_bgr is None:
            messagebox.showwarning("No image", "Nothing to save.")
            return
        if pdf_canvas is None:
            messagebox.showerror("Missing dependency", "Install ReportLab to export PDF with text: pip install reportlab")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if not path:
            return
        # Create PDF with image on page 1, OCR text (if any) on page 2
        try:
            c = pdf_canvas.Canvas(path, pagesize=A4)
            page_w, page_h = A4
            # Add image (fit to page with margins)
            rgb = cv2.cvtColor(self.current_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            img_reader = ImageReader(pil)
            margin = 36  # 0.5 inch
            img_w, img_h = pil.size
            scale = min((page_w - 2*margin) / img_w, (page_h - 2*margin) / img_h)
            draw_w, draw_h = img_w * scale, img_h * scale
            x = (page_w - draw_w) / 2
            y = (page_h - draw_h) / 2
            c.drawImage(img_reader, x, y, width=draw_w, height=draw_h)
            c.showPage()
            # Text page
            text_obj = c.beginText(36, page_h - 48)
            text_obj.setFont("Helvetica", 11)
            text = self.ocr_text.strip() or "(No OCR text. Run 'OCR Extract Text' first.)"
            for line in text.splitlines() or [text]:
                # simple wrap
                while len(line) > 100:
                    text_obj.textLine(line[:100])
                    line = line[100:]
                text_obj.textLine(line)
            c.drawText(text_obj)
            c.save()
        except Exception as e:
            messagebox.showerror("PDF error", str(e))
            return
        messagebox.showinfo("Saved", f"PDF saved to {path}")

    def set_tesseract_path(self):
        path = filedialog.askopenfilename(title="Select tesseract executable",
                                          filetypes=[("Executable", "*" )])
        if path:
            self.tesseract_cmd = path
            messagebox.showinfo("Tesseract", f"Using: {path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentScannerApp(root)
    root.mainloop()
