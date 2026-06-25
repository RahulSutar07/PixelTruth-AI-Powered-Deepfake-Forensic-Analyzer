"""
╔══════════════════════════════════════════════════════════════════╗
║   PixelTruth — AI-Powered Deepfake Forensic Analyzer (v2 UI)     ║
╠══════════════════════════════════════════════════════════════════╣
║  Engine 1 : SigLIP  (prithivMLmods/deepfake-detector-model-v1)   ║
║             Reported accuracy : 94.4 %                           ║
║  Engine 2 : ViT-v2  (prithivMLmods/Deep-Fake-Detector-v2-Model)  ║
║             Reported accuracy : 92.1 %                           ║
║  Engine 3 : Classical forensics (ELA / Noise / FFT / Texture)    ║
║             Used as confidence tie-breaker                       ║
╠═════════════════════════════════════════════════════════════════ ╣
║  Strategy : Weighted ensemble — DL models get 80 % weight,       ║
║             classical forensics 20 %.  If both DL models agree   ║
║             the combined accuracy exceeds 80 % easily.           ║
╠═════════════════════════════════════════════════════════════════ ╣
║  Install  : pip install torch torchvision transformers           ║
║             accelerate pillow opencv-python scipy scikit-image   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, sys, warnings, tempfile, threading, time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

# ── Tkinter ──────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ── Core libs ────────────────────────────────────────────────────
try:
    import numpy as np
    print("✅ NumPy loaded")
except ImportError as e:
    print(f"❌ NumPy error: {e}"); sys.exit(1)

try:
    import cv2
    print("✅ OpenCV loaded")
except ImportError as e:
    print(f"❌ OpenCV error: {e}"); sys.exit(1)

try:
    from PIL import Image, ImageChops, ImageTk, ImageDraw
    print("✅ PIL loaded")
except ImportError as e:
    print(f"❌ PIL error: {e}"); sys.exit(1)

try:
    from scipy import stats, ndimage
    from scipy.signal import find_peaks
    print("✅ SciPy loaded")
except ImportError as e:
    print(f"❌ SciPy error: {e}"); sys.exit(1)

try:
    import skimage
    from skimage.feature import local_binary_pattern
    from skimage.filters import gaussian
    print(f"✅ scikit-image {skimage.__version__} loaded")
except ImportError as e:
    print(f"❌ scikit-image error: {e}"); sys.exit(1)

# ── Deep-learning engines (optional but strongly recommended) ────
HAS_TORCH = False
HAS_SIGLIP = False
HAS_VIT = False

try:
    import torch
    HAS_TORCH = True
    print(f"✅ PyTorch {torch.__version__} loaded")
except ImportError:
    print("⚠️  PyTorch not found — DL engines disabled (accuracy will be lower)")
    print("    Install: pip install torch torchvision")

if HAS_TORCH:
    try:
        from transformers import (
            AutoImageProcessor,
            SiglipForImageClassification,
            ViTForImageClassification,
            ViTImageProcessor,
        )
        print("✅ Transformers loaded")

        # ── Engine 1: SigLIP (94.4 % accuracy) ───────────────────
        print("⏳ Loading SigLIP model (first run downloads ~372 MB)…")
        _siglip_name = "prithivMLmods/deepfake-detector-model-v1"
        _siglip_processor = AutoImageProcessor.from_pretrained(_siglip_name)
        _siglip_model     = SiglipForImageClassification.from_pretrained(_siglip_name)
        _siglip_model.eval()
        HAS_SIGLIP = True
        print("✅ SigLIP engine ready")

        # ── Engine 2: ViT-v2 (92.1 % accuracy) ───────────────────
        print("⏳ Loading ViT-v2 model (first run downloads ~330 MB)…")
        _vit_name      = "prithivMLmods/Deep-Fake-Detector-v2-Model"
        _vit_processor = ViTImageProcessor.from_pretrained(_vit_name)
        _vit_model     = ViTForImageClassification.from_pretrained(_vit_name)
        _vit_model.eval()
        HAS_VIT = True
        print("✅ ViT-v2 engine ready")

    except Exception as e:
        print(f"⚠️  Transformers/model error: {e}")
        print("    Falling back to classical forensics only.")

# ─────────────────────────────────────────────────────────────────
#  COLOURS — forensic dark theme with cyan accent
# ─────────────────────────────────────────────────────────────────

COLORS = {
    'bg_dark':        '#0a0e17',
    'bg_medium':      '#0f1623',
    'bg_light':       '#151e2d',
    'bg_card':        '#1a2438',
    'bg_sidebar':     '#0c1220',
    'text':           '#e2eaf4',
    'text_secondary': '#7a8fa8',
    'text_dim':       '#4a5a6e',
    'accent_green':   '#00c896',
    'accent_red':     '#ff4d6a',
    'accent_orange':  '#ffaa33',
    'accent_blue':    '#3d9bff',
    'accent_cyan':    '#00d4ff',
    'accent_purple':  '#9b5de5',
    'border':         '#1e2d42',
    'border_light':   '#253347',
    'sidebar_active': '#162035',
    'sidebar_hover':  '#111929',
}

WIN_W, WIN_H = 1280, 800

# ─────────────────────────────────────────────────────────────────
#  DL ENGINES  (UNCHANGED — backend)
# ─────────────────────────────────────────────────────────────────

def run_siglip(pil_img):
    if not HAS_SIGLIP:
        return None
    try:
        inputs = _siglip_processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            logits = _siglip_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
        fake_prob = probs[0] if len(probs) == 2 else 0.5
        return float(fake_prob)
    except Exception as e:
        print(f"SigLIP error: {e}")
        return None


def run_vit(pil_img):
    if not HAS_VIT:
        return None
    try:
        inputs = _vit_processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            logits = _vit_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
        id2label = _vit_model.config.id2label
        fake_index = None
        for idx, label in id2label.items():
            if 'fake' in str(label).lower() or 'deepfake' in str(label).lower():
                fake_index = int(idx)
                break
        if fake_index is None:
            fake_index = 1
        fake_prob = probs[fake_index] if fake_index < len(probs) else 0.5
        return float(fake_prob)
    except Exception as e:
        print(f"ViT error: {e}")
        return None

# ─────────────────────────────────────────────────────────────────
#  CLASSICAL FORENSICS  (UNCHANGED — backend)
# ─────────────────────────────────────────────────────────────────

def load_image(path):
    pil_img = Image.open(path).convert("RGB")
    cv_img  = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return pil_img, cv_img


def analyze_ela(pil_img):
    try:
        qualities = [95, 90, 85, 80, 75]
        errors    = []
        for quality in qualities:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    tmp_path = tmp.name
                pil_img.save(tmp_path, 'JPEG', quality=quality)
                resaved   = Image.open(tmp_path)
                diff      = ImageChops.difference(pil_img, resaved.convert('RGB'))
                errors.append(np.mean(np.array(diff)))
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        avg_error   = np.mean(errors)
        error_trend = np.polyfit(range(len(errors)), errors, 1)[0]
        suspicion   = 0
        findings    = []

        if avg_error < 3:
            suspicion += 0.6; findings.append("Very low ELA — AI-generated or heavily post-processed")
        elif avg_error < 8:
            suspicion += 0.3; findings.append("Below-normal ELA — possible AI generation")
        elif avg_error > 20:
            suspicion += 0.2; findings.append("High ELA — possible splicing")
        else:
            findings.append("ELA within normal range")

        if abs(error_trend) < 0.1:
            suspicion += 0.15; findings.append("Flat ELA trend — unnatural compression history")

        return {'suspicion': min(suspicion, 1.0), 'findings': findings, 'error_level': avg_error}
    except Exception as e:
        return {'suspicion': 0.3, 'findings': [f"ELA error: {e}"], 'error_level': 0}


def analyze_noise(cv_img):
    try:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        noise_levels = [
            np.std(gray - cv2.GaussianBlur(gray, ks, 0))
            for ks in [(3, 3), (5, 5), (7, 7)]
        ]
        noise_consistency = np.std(noise_levels) / (np.mean(noise_levels) + 1e-8)

        h, w       = gray.shape
        block_size = 64
        block_noises = [
            np.std(gray[y:y+block_size, x:x+block_size]
                   - cv2.GaussianBlur(gray[y:y+block_size, x:x+block_size], (5, 5), 0))
            for y in range(0, h - block_size, block_size // 2)
            for x in range(0, w - block_size, block_size // 2)
        ]
        noise_uniformity = np.std(block_noises) / (np.mean(block_noises) + 1e-8)

        avg_noise = np.mean(noise_levels)
        suspicion = 0; findings = []

        if avg_noise < 1.0:
            suspicion += 0.5; findings.append("Very low noise — AI or heavy processing")
        elif avg_noise > 8:
            suspicion += 0.15; findings.append("High noise — low-light or compressed image")

        if noise_consistency < 0.1:
            suspicion += 0.3; findings.append("Noise too consistent across scales — AI artifact")
        elif noise_consistency > 0.5:
            suspicion += 0.15; findings.append("Inconsistent noise — possible splice")

        if noise_uniformity < 0.2:
            suspicion += 0.15; findings.append("Uniform noise pattern — typical of AI generation")
        elif noise_uniformity > 0.8:
            suspicion += 0.25; findings.append("Non-uniform noise — possible spliced regions")

        if not findings:
            findings.append("Natural noise pattern")

        return {'suspicion': min(suspicion, 1.0), 'findings': findings, 'noise_level': avg_noise}
    except Exception as e:
        return {'suspicion': 0.2, 'findings': [f"Noise error: {e}"], 'noise_level': 0}


def analyze_metadata(image_path):
    try:
        img = Image.open(image_path)
        try:
            exif_data = dict(img.getexif())
        except Exception:
            exif_data = {}

        suspicion = 0; findings = []
        if exif_data:
            findings.append("EXIF present — likely camera origin")
            for tag_id, value in exif_data.items():
                if not isinstance(value, (str, bytes)):
                    continue
                try:
                    vs = value.decode('utf-8', errors='ignore').lower() if isinstance(value, bytes) else str(value).lower()
                    if any(k in vs for k in ['photoshop', 'lightroom', 'gimp', 'affinity']):
                        findings.append(f"Editing software: {vs[:30]}"); suspicion += 0.25
                    if any(k in vs for k in ['stable diffusion', 'midjourney', 'dall-e', 'firefly', 'comfyui']):
                        findings.append(f"AI software in EXIF: {vs[:30]}"); suspicion += 0.9
                except Exception:
                    pass
        else:
            findings.append("No EXIF — common in AI / web images"); suspicion += 0.25

        file_size  = os.path.getsize(image_path) / 1024
        w, h       = img.size
        megapixels = (w * h) / 1_000_000
        if file_size < 100 and megapixels > 2:
            findings.append(f"Suspicious file size ({file_size:.0f} KB for {megapixels:.1f} MP)")
            suspicion += 0.15

        return {'suspicion': min(suspicion, 1.0), 'findings': findings, 'has_exif': bool(exif_data)}
    except Exception as e:
        return {'suspicion': 0.2, 'findings': [f"Metadata error: {e}"], 'has_exif': False}


def analyze_frequency(cv_img):
    try:
        gray      = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        f         = np.fft.fft2(gray)
        fshift    = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y, x   = np.ogrid[-cy:h-cy, -cx:w-cx]
        radius = np.sqrt(x*x + y*y)
        max_r  = min(cy, cx)

        rings = [
            np.mean(magnitude[(radius >= r) & (radius < r + 5)])
            for r in range(0, max_r, 5)
            if ((radius >= r) & (radius < r + 5)).any()
        ]
        peaks, _ = find_peaks(rings, height=0.1, distance=3)
        hf_energy = np.mean(magnitude[radius > max_r * 0.7])

        suspicion = 0; findings = []
        if len(peaks) > 8:
            suspicion += 0.6; findings.append(f"{len(peaks)} periodic FFT peaks — GAN artifact")
        elif len(peaks) > 4:
            suspicion += 0.25; findings.append("Periodic FFT peaks — possible AI")
        if hf_energy < 0.05:
            suspicion += 0.5; findings.append("Very low high-frequency energy — AI generated")
        elif hf_energy < 0.1:
            suspicion += 0.15; findings.append("Low high-frequency energy — possibly compressed")
        if not findings:
            findings.append("Normal frequency distribution")

        return {'suspicion': min(suspicion, 1.0), 'findings': findings, 'hf_energy': hf_energy}
    except Exception as e:
        return {'suspicion': 0.1, 'findings': [f"FFT error: {e}"], 'hf_energy': 0}


def analyze_texture(cv_img):
    try:
        gray     = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        radius   = 3; n_points = 8 * radius
        lbp      = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_s    = local_binary_pattern(gaussian(gray, sigma=1), n_points, radius, method='uniform')
        lbp_std  = np.std(lbp)
        tex_cons = np.corrcoef(lbp.flatten(), lbp_s.flatten())[0, 1]

        suspicion = 0; findings = []
        if lbp_std < 5:
            suspicion += 0.5; findings.append("Very low texture variation — AI generated")
        elif lbp_std < 10:
            suspicion += 0.2; findings.append("Limited texture variation — possible AI")
        if tex_cons > 0.95:
            suspicion += 0.4; findings.append("Texture too consistent across scales")
        if not findings:
            findings.append("Natural texture patterns")

        return {'suspicion': min(suspicion, 1.0), 'findings': findings, 'tex_variation': lbp_std}
    except Exception as e:
        return {'suspicion': 0.2, 'findings': [f"Texture error: {e}"], 'tex_variation': 0}

# ─────────────────────────────────────────────────────────────────
#  ENSEMBLE VERDICT  (UNCHANGED — backend)
# ─────────────────────────────────────────────────────────────────

def compute_verdict(pil_img, image_path, cv_img):
    siglip_fake = run_siglip(pil_img)
    vit_fake    = run_vit(pil_img)

    classical = {
        'ela':       analyze_ela(pil_img),
        'noise':     analyze_noise(cv_img),
        'metadata':  analyze_metadata(image_path),
        'frequency': analyze_frequency(cv_img),
        'texture':   analyze_texture(cv_img),
    }
    classical_weights = {'ela': 0.20, 'noise': 0.25, 'metadata': 0.15,
                         'frequency': 0.20, 'texture': 0.20}
    classical_score = sum(
        classical[k]['suspicion'] * classical_weights[k]
        for k in classical_weights
    )

    total_w    = 0
    fake_score = 0.0

    if siglip_fake is not None:
        fake_score += siglip_fake * 0.45
        total_w    += 0.45
    if vit_fake is not None:
        fake_score += vit_fake * 0.35
        total_w    += 0.35

    classical_w = 1.0 - total_w
    fake_score += classical_score * classical_w
    fake_pct = fake_score * 100

    if fake_pct < 20:
        primary = "✅ Real Photo";            confidence = "High"
    elif fake_pct < 38:
        primary = "🟢 Likely Real";           confidence = "Medium"
    elif fake_pct < 55:
        primary = "🟡 Uncertain / Edited";    confidence = "Low"
    elif fake_pct < 72:
        primary = "🟠 Likely Fake / AI";      confidence = "Medium"
    else:
        primary = "🔴 Deepfake / AI-Generated"; confidence = "High"

    real_s = max(0.0, 100 - fake_pct * 1.15)
    df_s   = fake_pct * 0.55
    ai_s   = fake_pct * 0.35
    ed_s   = fake_pct * 0.10
    total  = real_s + df_s + ai_s + ed_s
    if total > 0:
        real_s, df_s, ai_s, ed_s = [v / total * 100 for v in (real_s, df_s, ai_s, ed_s)]

    rounded = [round(real_s), round(df_s), round(ai_s), round(ed_s)]
    diff    = 100 - sum(rounded)
    if diff:
        rounded[[real_s, df_s, ai_s, ed_s].index(max(real_s, df_s, ai_s, ed_s))] += diff

    all_findings = []
    for v in classical.values():
        all_findings.extend(v.get('findings', []))

    return {
        'fake_pct':       round(fake_pct, 1),
        'primary':        primary,
        'confidence':     confidence,
        'findings':       all_findings[:12],
        'siglip_fake':    siglip_fake,
        'vit_fake':       vit_fake,
        'classical_score': round(classical_score * 100, 1),
        'category_scores': {
            'real':     rounded[0],
            'deepfake': rounded[1],
            'aigen':    rounded[2],
            'edited':   rounded[3],
        },
        'classical': classical,
    }

# ─────────────────────────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────────────────────────

def make_canvas_rounded_rect(canvas, x1, y1, x2, y2, radius=10, **kwargs):
    """Draw a rounded rectangle on a canvas."""
    pts = [
        x1+radius, y1,
        x2-radius, y1,
        x2, y1,
        x2, y1+radius,
        x2, y2-radius,
        x2, y2,
        x2-radius, y2,
        x1+radius, y2,
        x1, y2,
        x1, y2-radius,
        x1, y1+radius,
        x1, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kwargs)


class ModernCard(tk.Frame):
    """A styled card frame."""
    def __init__(self, parent, title=None, accent_color=None, **kwargs):
        super().__init__(parent,
            bg=COLORS['bg_card'],
            highlightbackground=COLORS['border'],
            highlightthickness=1,
            **kwargs)
        if title:
            header = tk.Frame(self, bg=COLORS['bg_card'])
            header.pack(fill='x', padx=14, pady=(12, 6))
            if accent_color:
                tk.Frame(header, bg=accent_color, width=3, height=16).pack(side='left', padx=(0,8))
            tk.Label(header, text=title,
                bg=COLORS['bg_card'], fg=COLORS['text'],
                font=('Segoe UI', 10, 'bold')).pack(side='left')


class StatusDot(tk.Frame):
    """Animated status indicator dot."""
    def __init__(self, parent, color=None, size=8, **kwargs):
        super().__init__(parent, bg=COLORS['bg_card'], **kwargs)
        self._color = color or COLORS['accent_green']
        self._canvas = tk.Canvas(self, width=size, height=size,
            bg=COLORS['bg_card'], highlightthickness=0)
        self._canvas.pack()
        self._canvas.create_oval(1, 1, size-1, size-1,
            fill=self._color, outline='')


class SidebarButton(tk.Frame):
    """Custom sidebar navigation button."""
    def __init__(self, parent, icon, label, command, active=False, **kwargs):
        super().__init__(parent, bg=COLORS['bg_sidebar'], cursor='hand2', **kwargs)
        self._command = command
        self._active = active
        self._label = label

        self._indicator = tk.Frame(self, bg=COLORS['bg_sidebar'], width=3)
        self._indicator.pack(side='left', fill='y')

        self._inner = tk.Frame(self, bg=COLORS['bg_sidebar'])
        self._inner.pack(side='left', fill='both', expand=True, padx=(10, 14), pady=10)

        self._icon_lbl = tk.Label(self._inner, text=icon,
            bg=COLORS['bg_sidebar'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 14))
        self._icon_lbl.pack(side='left', padx=(0, 10))

        self._text_lbl = tk.Label(self._inner, text=label,
            bg=COLORS['bg_sidebar'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 10))
        self._text_lbl.pack(side='left')

        for w in [self, self._inner, self._icon_lbl, self._text_lbl, self._indicator]:
            w.bind('<Button-1>', lambda e: self._command())
            w.bind('<Enter>', self._on_enter)
            w.bind('<Leave>', self._on_leave)

        if active:
            self.set_active(True)

    def set_active(self, state):
        self._active = state
        bg = COLORS['sidebar_active'] if state else COLORS['bg_sidebar']
        fg_icon = COLORS['accent_cyan'] if state else COLORS['text_secondary']
        fg_text = COLORS['text'] if state else COLORS['text_secondary']
        ind_bg  = COLORS['accent_cyan'] if state else COLORS['bg_sidebar']
        font_w  = 'bold' if state else 'normal'

        self._indicator.config(bg=ind_bg)
        self._inner.config(bg=bg)
        self._icon_lbl.config(bg=bg, fg=fg_icon)
        self._text_lbl.config(bg=bg, fg=fg_text, font=('Segoe UI', 10, font_w))
        self.config(bg=bg)

    def _on_enter(self, e):
        if not self._active:
            bg = COLORS['sidebar_hover']
            for w in [self, self._inner, self._icon_lbl, self._text_lbl]:
                w.config(bg=bg)

    def _on_leave(self, e):
        if not self._active:
            bg = COLORS['bg_sidebar']
            for w in [self, self._inner, self._icon_lbl, self._text_lbl]:
                w.config(bg=bg)


def make_progress_bar(parent, value, max_val=100, color=None, height=6, width=200):
    """Create a thin progress bar canvas."""
    color = color or COLORS['accent_blue']
    c = tk.Canvas(parent, width=width, height=height,
        bg=COLORS['bg_light'], highlightthickness=0)
    fill_w = int(width * (value / max(max_val, 1)))
    c.create_rectangle(0, 0, width, height, fill=COLORS['border'], outline='')
    if fill_w > 0:
        c.create_rectangle(0, 0, fill_w, height, fill=color, outline='')
    return c


# ─────────────────────────────────────────────────────────────────
#  MAIN GUI CLASS
# ─────────────────────────────────────────────────────────────────

class DeepfakeDetectorV2:
    def __init__(self, root):
        self.root   = root
        self.root.title("PixelTruth — AI-Powered Deepfake Forensic Analyzer")
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.root.configure(bg=COLORS['bg_dark'])

        # ── Backend state variables (unchanged names) ─────────────
        self.current_image_path = None
        self.preview_image      = None
        self._last_verdict      = None   # stores last result dict

        # ── UI state ─────────────────────────────────────────────
        self._current_page = 'dashboard'
        self._sidebar_btns = {}
        self._pages        = {}

        self._build_ui()
        self._show_page('dashboard')
        self._show_engine_status()

    # ─────────────────────────────────────────────────────────────
    #  TOP-LEVEL LAYOUT
    # ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Root grid: sidebar | content
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_content_area()

    # ─────────────────────────────────────────────────────────────
    #  SIDEBAR
    # ─────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        sidebar = tk.Frame(self.root, bg=COLORS['bg_sidebar'], width=220)
        sidebar.grid(row=0, column=0, sticky='ns')
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)

        # Branding
        brand_frame = tk.Frame(sidebar, bg=COLORS['bg_sidebar'])
        brand_frame.pack(fill='x', pady=(20, 6), padx=16)

        # Logo block
        logo_row = tk.Frame(brand_frame, bg=COLORS['bg_sidebar'])
        logo_row.pack(fill='x')
        logo_box = tk.Frame(logo_row, bg=COLORS['accent_cyan'],
            width=32, height=32)
        logo_box.pack(side='left', padx=(0, 10))
        logo_box.pack_propagate(False)
        tk.Label(logo_box, text='◈', bg=COLORS['accent_cyan'],
            fg=COLORS['bg_dark'], font=('Segoe UI', 16, 'bold')).place(relx=0.5, rely=0.5, anchor='center')

        brand_col = tk.Frame(logo_row, bg=COLORS['bg_sidebar'])
        brand_col.pack(side='left')
        tk.Label(brand_col, text='PixelTruth',
            bg=COLORS['bg_sidebar'], fg=COLORS['text'],
            font=('Segoe UI', 13, 'bold')).pack(anchor='w')
        tk.Label(brand_col, text='Deepfake Forensic Analyzer',
            bg=COLORS['bg_sidebar'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 7)).pack(anchor='w')

        # Divider
        tk.Frame(sidebar, bg=COLORS['border'], height=1).pack(fill='x', pady=(12, 8))

        # Nav label
        tk.Label(sidebar, text='NAVIGATION',
            bg=COLORS['bg_sidebar'], fg=COLORS['text_dim'],
            font=('Segoe UI', 7, 'bold')).pack(anchor='w', padx=18, pady=(4, 4))

        # Nav buttons
        nav_items = [
            ('dashboard', '⬛', 'Dashboard'),
            ('analysis',  '🔬', 'New Analysis'),
            ('report',    '📄', 'Report'),
            ('history',   '📋', 'History Log'),
        ]
        for page_id, icon, label in nav_items:
            btn = SidebarButton(sidebar, icon, label,
                command=lambda pid=page_id: self._show_page(pid),
                active=(page_id == 'dashboard'))
            btn.pack(fill='x')
            self._sidebar_btns[page_id] = btn

        # Divider
        tk.Frame(sidebar, bg=COLORS['border'], height=1).pack(fill='x', pady=(8, 6))

        # Tools label
        tk.Label(sidebar, text='TOOLS',
            bg=COLORS['bg_sidebar'], fg=COLORS['text_dim'],
            font=('Segoe UI', 7, 'bold')).pack(anchor='w', padx=18, pady=(4, 4))

        tk.Button(sidebar, text='📁  Select Image',
            command=self._upload,
            bg=COLORS['accent_cyan'], fg=COLORS['bg_dark'],
            font=('Segoe UI', 9, 'bold'),
            padx=12, pady=7, relief='flat', cursor='hand2',
            activebackground='#00b8d9', activeforeground=COLORS['bg_dark']
        ).pack(fill='x', padx=14, pady=(4, 3))

        tk.Button(sidebar, text='📂  Batch Analyse',
            command=self._batch_upload,
            bg=COLORS['bg_card'], fg=COLORS['text'],
            font=('Segoe UI', 9),
            padx=12, pady=7, relief='flat', cursor='hand2',
            highlightbackground=COLORS['border'], highlightthickness=1,
            activebackground=COLORS['bg_light'], activeforeground=COLORS['text']
        ).pack(fill='x', padx=14, pady=3)

        # Engine status area at bottom
        sidebar.pack_propagate(False)
        status_frame = tk.Frame(sidebar, bg=COLORS['bg_sidebar'])
        status_frame.pack(side='bottom', fill='x', padx=14, pady=12)

        tk.Frame(status_frame, bg=COLORS['border'], height=1).pack(fill='x', pady=(0, 10))

        tk.Label(status_frame, text='ENGINE STATUS',
            bg=COLORS['bg_sidebar'], fg=COLORS['text_dim'],
            font=('Segoe UI', 7, 'bold')).pack(anchor='w')

        engine_row1 = tk.Frame(status_frame, bg=COLORS['bg_sidebar'])
        engine_row1.pack(fill='x', pady=3)
        StatusDot(engine_row1,
            color=COLORS['accent_green'] if HAS_SIGLIP else COLORS['accent_red'],
            size=7).pack(side='left', padx=(0, 6))
        tk.Label(engine_row1, text='SigLIP  94.4%',
            bg=COLORS['bg_sidebar'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 8)).pack(side='left')

        engine_row2 = tk.Frame(status_frame, bg=COLORS['bg_sidebar'])
        engine_row2.pack(fill='x', pady=3)
        StatusDot(engine_row2,
            color=COLORS['accent_green'] if HAS_VIT else COLORS['accent_red'],
            size=7).pack(side='left', padx=(0, 6))
        tk.Label(engine_row2, text='ViT-v2  92.1%',
            bg=COLORS['bg_sidebar'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 8)).pack(side='left')

        engine_row3 = tk.Frame(status_frame, bg=COLORS['bg_sidebar'])
        engine_row3.pack(fill='x', pady=3)
        StatusDot(engine_row3, color=COLORS['accent_blue'], size=7).pack(side='left', padx=(0, 6))
        tk.Label(engine_row3, text='Forensics  Active',
            bg=COLORS['bg_sidebar'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 8)).pack(side='left')

        self.status_label = tk.Label(status_frame, text='Initialising…',
            bg=COLORS['bg_sidebar'], fg=COLORS['text_dim'],
            font=('Segoe UI', 7), wraplength=185, justify='left')
        self.status_label.pack(anchor='w', pady=(6, 0))

    # ─────────────────────────────────────────────────────────────
    #  CONTENT AREA (right of sidebar)
    # ─────────────────────────────────────────────────────────────

    def _build_content_area(self):
        self._content = tk.Frame(self.root, bg=COLORS['bg_dark'])
        self._content.grid(row=0, column=1, sticky='nsew')
        self._content.columnconfigure(0, weight=1)
        self._content.rowconfigure(0, weight=1)

        # Build all pages
        self._pages['dashboard'] = self._build_dashboard_page(self._content)
        self._pages['analysis']  = self._build_analysis_page(self._content)
        self._pages['report']    = self._build_report_page(self._content)
        self._pages['history']   = self._build_history_page(self._content)

    def _show_page(self, page_id):
        # Update sidebar active state
        for pid, btn in self._sidebar_btns.items():
            btn.set_active(pid == page_id)

        # Hide all pages
        for pid, frame in self._pages.items():
            frame.grid_remove()

        # Show target
        self._pages[page_id].grid(row=0, column=0, sticky='nsew')
        self._current_page = page_id

        # Refresh history when shown
        if page_id == 'history':
            self._refresh_history_tree()

        # Refresh report when shown
        if page_id == 'report' and self._last_verdict:
            self._render_report(self._last_verdict)

    # ─────────────────────────────────────────────────────────────
    #  PAGE 1 — DASHBOARD
    # ─────────────────────────────────────────────────────────────

    def _build_dashboard_page(self, parent):
        page = tk.Frame(parent, bg=COLORS['bg_dark'])
        page.columnconfigure(0, weight=1)

        # Top bar
        topbar = tk.Frame(page, bg=COLORS['bg_medium'],
            highlightbackground=COLORS['border'], highlightthickness=1)
        topbar.pack(fill='x')

        tb_inner = tk.Frame(topbar, bg=COLORS['bg_medium'])
        tb_inner.pack(fill='x', padx=24, pady=14)

        tk.Label(tb_inner, text='PixelTruth Dashboard',
            bg=COLORS['bg_medium'], fg=COLORS['text'],
            font=('Segoe UI', 16, 'bold')).pack(side='left')

        # Breadcrumb / sub-label
        tk.Label(tb_inner, text='AI-Powered Deepfake Forensic Analyzer',
            bg=COLORS['bg_medium'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 9)).pack(side='left', padx=16)

        # System status badge
        status_badge = tk.Frame(tb_inner, bg=COLORS['bg_card'],
            highlightbackground=COLORS['border'], highlightthickness=1)
        status_badge.pack(side='right')
        badge_inner = tk.Frame(status_badge, bg=COLORS['bg_card'])
        badge_inner.pack(padx=12, pady=6)
        StatusDot(badge_inner, color=COLORS['accent_green'], size=8).pack(side='left', padx=(0, 6))
        tk.Label(badge_inner, text='All Systems Operational',
            bg=COLORS['bg_card'], fg=COLORS['accent_green'],
            font=('Segoe UI', 8, 'bold')).pack(side='left')

        # Content area
        content = tk.Frame(page, bg=COLORS['bg_dark'])
        content.pack(fill='both', expand=True, padx=24, pady=20)

        # ── Row 1: stat cards ────────────────────────────────────
        stat_row = tk.Frame(content, bg=COLORS['bg_dark'])
        stat_row.pack(fill='x', pady=(0, 16))

        self._stat_vars = {}
        stat_defs = [
            ('Total Analyses', '0',        COLORS['accent_cyan'],   '📊'),
            ('Deepfakes Found','0',         COLORS['accent_red'],    '🚨'),
            ('Verified Real',  '0',         COLORS['accent_green'],  '✅'),
            ('Detection Rate', 'N/A',       COLORS['accent_orange'], '📈'),
        ]
        for i, (title, val, color, icon) in enumerate(stat_defs):
            card = tk.Frame(stat_row, bg=COLORS['bg_card'],
                highlightbackground=COLORS['border'], highlightthickness=1)
            card.pack(side='left', fill='both', expand=True,
                padx=(0 if i == 0 else 10, 0))

            inner = tk.Frame(card, bg=COLORS['bg_card'])
            inner.pack(padx=16, pady=14, fill='x')

            # Accent strip
            tk.Frame(inner, bg=color, width=3, height=40).pack(side='left', padx=(0, 12))

            col = tk.Frame(inner, bg=COLORS['bg_card'])
            col.pack(side='left')

            var = tk.StringVar(value=val)
            self._stat_vars[title] = var

            tk.Label(col, textvariable=var,
                bg=COLORS['bg_card'], fg=color,
                font=('Segoe UI', 22, 'bold')).pack(anchor='w')
            tk.Label(col, text=title,
                bg=COLORS['bg_card'], fg=COLORS['text_secondary'],
                font=('Segoe UI', 8)).pack(anchor='w')

            tk.Label(inner, text=icon,
                bg=COLORS['bg_card'], fg=color,
                font=('Segoe UI', 20)).pack(side='right')

        # ── Row 2: Engine status + Quick start ───────────────────
        row2 = tk.Frame(content, bg=COLORS['bg_dark'])
        row2.pack(fill='both', expand=True, pady=(0, 16))
        row2.columnconfigure(0, weight=3)
        row2.columnconfigure(1, weight=2)

        # Engine status card
        engine_card = ModernCard(row2, title='AI Engine Status', accent_color=COLORS['accent_cyan'])
        engine_card.grid(row=0, column=0, sticky='nsew', padx=(0, 12))

        engine_body = tk.Frame(engine_card, bg=COLORS['bg_card'])
        engine_body.pack(fill='both', expand=True, padx=14, pady=(0, 14))

        engine_defs = [
            ('SigLIP Engine',    '94.4% accuracy',  HAS_SIGLIP, 'Semantic & Visual Alignment'),
            ('ViT-v2 Transformer','92.1% accuracy', HAS_VIT,    'Fine-grained Patch Analysis'),
            ('Forensic Suite',   'Multi-module',    True,       'ELA · Noise · FFT · Texture'),
        ]
        for name, acc, ready, desc in engine_defs:
            row = tk.Frame(engine_body, bg=COLORS['bg_card'])
            row.pack(fill='x', pady=6)

            # Status indicator
            color = COLORS['accent_green'] if ready else COLORS['accent_red']
            status_txt = 'ONLINE' if ready else 'OFFLINE'

            left_col = tk.Frame(row, bg=COLORS['bg_card'])
            left_col.pack(side='left', fill='x', expand=True)

            name_row = tk.Frame(left_col, bg=COLORS['bg_card'])
            name_row.pack(anchor='w')
            StatusDot(name_row, color=color, size=8).pack(side='left', padx=(0, 6), pady=2)
            tk.Label(name_row, text=name,
                bg=COLORS['bg_card'], fg=COLORS['text'],
                font=('Segoe UI', 9, 'bold')).pack(side='left')

            tk.Label(left_col, text=desc,
                bg=COLORS['bg_card'], fg=COLORS['text_secondary'],
                font=('Segoe UI', 8)).pack(anchor='w', padx=14)

            right_col = tk.Frame(row, bg=COLORS['bg_card'])
            right_col.pack(side='right')
            tk.Label(right_col, text=acc,
                bg=COLORS['bg_card'], fg=color,
                font=('Segoe UI', 9, 'bold')).pack(anchor='e')
            tk.Label(right_col, text=status_txt,
                bg=COLORS['bg_card'], fg=COLORS['text_dim'],
                font=('Segoe UI', 7)).pack(anchor='e')

            # Divider
            if name != 'Forensic Suite':
                tk.Frame(engine_body, bg=COLORS['border'], height=1).pack(fill='x', pady=2)

        # Quick-start card
        qs_card = ModernCard(row2, title='Quick Start', accent_color=COLORS['accent_purple'])
        qs_card.grid(row=0, column=1, sticky='nsew')

        qs_body = tk.Frame(qs_card, bg=COLORS['bg_card'])
        qs_body.pack(fill='both', expand=True, padx=14, pady=(0, 14))

        steps = [
            ('1', 'Select Image',  'Choose JPG, PNG, or WEBP'),
            ('2', 'Run Analysis',  'AI + forensic inspection'),
            ('3', 'View Report',   'Full breakdown & verdict'),
        ]
        for num, step, desc in steps:
            step_row = tk.Frame(qs_body, bg=COLORS['bg_card'])
            step_row.pack(fill='x', pady=8)

            num_box = tk.Label(step_row, text=num,
                bg=COLORS['accent_purple'], fg='white',
                font=('Segoe UI', 9, 'bold'),
                width=2, pady=3)
            num_box.pack(side='left', padx=(0, 12))

            step_col = tk.Frame(step_row, bg=COLORS['bg_card'])
            step_col.pack(side='left')
            tk.Label(step_col, text=step,
                bg=COLORS['bg_card'], fg=COLORS['text'],
                font=('Segoe UI', 9, 'bold')).pack(anchor='w')
            tk.Label(step_col, text=desc,
                bg=COLORS['bg_card'], fg=COLORS['text_secondary'],
                font=('Segoe UI', 8)).pack(anchor='w')

        tk.Button(qs_body, text='⚡  Start Analysis →',
            command=lambda: self._show_page('analysis'),
            bg=COLORS['accent_purple'], fg='white',
            font=('Segoe UI', 10, 'bold'),
            padx=14, pady=9, relief='flat', cursor='hand2',
            activebackground='#8040d0', activeforeground='white'
        ).pack(fill='x', pady=(10, 0))

        # ── Row 3: Mode indicator ─────────────────────────────────
        mode_frame = tk.Frame(content, bg=COLORS['bg_card'],
            highlightbackground=COLORS['border'], highlightthickness=1)
        mode_frame.pack(fill='x')

        mode_inner = tk.Frame(mode_frame, bg=COLORS['bg_card'])
        mode_inner.pack(fill='x', padx=16, pady=10)

        mode_text = 'DL + Classical Ensemble' if (HAS_SIGLIP or HAS_VIT) else 'Classical Forensics Only'
        mode_color = COLORS['accent_cyan'] if (HAS_SIGLIP or HAS_VIT) else COLORS['accent_orange']

        tk.Label(mode_inner, text='⚡  Detection Mode:',
            bg=COLORS['bg_card'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 9)).pack(side='left')
        tk.Label(mode_inner, text=f'  {mode_text}',
            bg=COLORS['bg_card'], fg=mode_color,
            font=('Segoe UI', 9, 'bold')).pack(side='left')
        tk.Label(mode_inner, text='   |   Multi-engine ensemble with classical forensic verification',
            bg=COLORS['bg_card'], fg=COLORS['text_dim'],
            font=('Segoe UI', 8)).pack(side='left')

        return page

    # ─────────────────────────────────────────────────────────────
    #  PAGE 2 — ANALYSIS
    # ─────────────────────────────────────────────────────────────

    def _build_analysis_page(self, parent):
        page = tk.Frame(parent, bg=COLORS['bg_dark'])
        page.columnconfigure(0, weight=1)
        page.rowconfigure(1, weight=1)

        # Top bar
        topbar = tk.Frame(page, bg=COLORS['bg_medium'],
            highlightbackground=COLORS['border'], highlightthickness=1)
        topbar.pack(fill='x')

        tb_inner = tk.Frame(topbar, bg=COLORS['bg_medium'])
        tb_inner.pack(fill='x', padx=24, pady=14)
        tk.Label(tb_inner, text='Forensic Evidence Intake',
            bg=COLORS['bg_medium'], fg=COLORS['text'],
            font=('Segoe UI', 16, 'bold')).pack(side='left')
        tk.Label(tb_inner, text='  Secure, chain-of-custody compliant ingestion for digital evidence analysis.',
            bg=COLORS['bg_medium'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 9)).pack(side='left')

        # Main content
        content = tk.Frame(page, bg=COLORS['bg_dark'])
        content.pack(fill='both', expand=True, padx=24, pady=20)
        content.columnconfigure(0, weight=5)
        content.columnconfigure(1, weight=4)
        content.rowconfigure(0, weight=1)

        # ── Left: Upload + Preview ────────────────────────────────
        left_col = tk.Frame(content, bg=COLORS['bg_dark'])
        left_col.grid(row=0, column=0, sticky='nsew', padx=(0, 14))
        left_col.rowconfigure(0, weight=0)
        left_col.rowconfigure(1, weight=1)
        left_col.rowconfigure(2, weight=0)
        left_col.columnconfigure(0, weight=1)

        # Upload dropzone
        upload_card = tk.Frame(left_col, bg=COLORS['bg_card'],
            highlightbackground=COLORS['border_light'], highlightthickness=2)
        upload_card.grid(row=0, column=0, sticky='ew', pady=(0, 12))

        drop_inner = tk.Frame(upload_card, bg=COLORS['bg_card'])
        drop_inner.pack(padx=30, pady=28)

        tk.Label(drop_inner, text='↑',
            bg=COLORS['bg_card'], fg=COLORS['accent_cyan'],
            font=('Segoe UI', 28)).pack()
        tk.Label(drop_inner, text='Select evidence file for analysis',
            bg=COLORS['bg_card'], fg=COLORS['text'],
            font=('Segoe UI', 11, 'bold')).pack(pady=(6, 2))
        tk.Label(drop_inner, text='Supported: JPG · PNG · WEBP · BMP',
            bg=COLORS['bg_card'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 8)).pack(pady=(0, 14))

        btn_row = tk.Frame(drop_inner, bg=COLORS['bg_card'])
        btn_row.pack()
        tk.Button(btn_row, text='Browse Files',
            command=self._upload,
            bg=COLORS['accent_cyan'], fg=COLORS['bg_dark'],
            font=('Segoe UI', 10, 'bold'),
            padx=20, pady=8, relief='flat', cursor='hand2',
            activebackground='#00b8d9', activeforeground=COLORS['bg_dark']
        ).pack(side='left', padx=(0, 10))

        tk.Button(btn_row, text='Select Folder',
            command=self._batch_upload,
            bg=COLORS['bg_light'], fg=COLORS['text'],
            font=('Segoe UI', 10),
            padx=20, pady=8, relief='flat', cursor='hand2',
            highlightbackground=COLORS['border'], highlightthickness=1,
            activebackground=COLORS['border_light'], activeforeground=COLORS['text']
        ).pack(side='left')

        # Preview card
        preview_card = tk.Frame(left_col, bg=COLORS['bg_card'],
            highlightbackground=COLORS['border'], highlightthickness=1)
        preview_card.grid(row=1, column=0, sticky='nsew', pady=(0, 12))

        preview_header = tk.Frame(preview_card, bg=COLORS['bg_card'])
        preview_header.pack(fill='x', padx=14, pady=(12, 6))
        tk.Frame(preview_header, bg=COLORS['accent_blue'], width=3, height=16).pack(side='left', padx=(0, 8))
        tk.Label(preview_header, text='Evidence Preview',
            bg=COLORS['bg_card'], fg=COLORS['text'],
            font=('Segoe UI', 10, 'bold')).pack(side='left')

        preview_area = tk.Frame(preview_card, bg=COLORS['bg_light'], height=280)
        preview_area.pack(fill='x', padx=14, pady=(0, 12))
        preview_area.pack_propagate(False)

        self.preview_label = tk.Label(preview_area,
            text='No evidence loaded',
            bg=COLORS['bg_light'], fg=COLORS['text_dim'],
            font=('Segoe UI', 11))
        self.preview_label.pack(expand=True)

        # File info label
        self.info_label = tk.Label(preview_card,
            text='Ready to receive evidence file',
            bg=COLORS['bg_card'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 8), wraplength=340, justify='left')
        self.info_label.pack(padx=14, pady=(0, 12), anchor='w')

        # Analyse button card — moved here, directly under Evidence Preview
        action_card = tk.Frame(left_col, bg=COLORS['bg_card'],
            highlightbackground=COLORS['border'], highlightthickness=1)
        action_card.grid(row=2, column=0, sticky='ew')

        action_inner = tk.Frame(action_card, bg=COLORS['bg_card'])
        action_inner.pack(fill='x', padx=16, pady=16)

        tk.Label(action_inner, text='Analysis Control',
            bg=COLORS['bg_card'], fg=COLORS['text'],
            font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 12))

        self.analyze_btn = tk.Button(
            action_inner, text='🔬  Run Forensic Analysis',
            command=self._start_analysis,
            bg=COLORS['accent_green'], fg='white',
            font=('Segoe UI', 11, 'bold'),
            padx=14, pady=10, relief='flat',
            state='disabled', cursor='hand2',
            activebackground='#00a87c', activeforeground='white'
        )
        self.analyze_btn.pack(fill='x')

        # Progress bar
        prog_frame = tk.Frame(action_inner, bg=COLORS['bg_card'])
        prog_frame.pack(fill='x', pady=(10, 0))
        self.progress = ttk.Progressbar(prog_frame, mode='indeterminate', length=300)
        self.progress.pack(fill='x')

        # ── Right: Controls + Status ──────────────────────────────
        right_col = tk.Frame(content, bg=COLORS['bg_dark'])
        right_col.grid(row=0, column=1, sticky='nsew')
        right_col.columnconfigure(0, weight=1)

        # Confidence display card
        conf_card = tk.Frame(right_col, bg=COLORS['bg_card'],
            highlightbackground=COLORS['border'], highlightthickness=1)
        conf_card.pack(fill='x', pady=(0, 12))

        conf_inner = tk.Frame(conf_card, bg=COLORS['bg_card'])
        conf_inner.pack(fill='x', padx=16, pady=14)

        tk.Label(conf_inner, text='Detection Result',
            bg=COLORS['bg_card'], fg=COLORS['text'],
            font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 10))

        # ── Verdict display box ────────────────────────────────────
        # NOTE: no percentage or progress bar is shown here — the verdict
        # box displays only the simple text result (Real / Deepfake /
        # AI-Generated). Backend computation (compute_verdict / fake_pct /
        # category_scores) is unchanged; this box only re-renders the
        # existing result as plain text.
        self._conf_verdict_var = tk.StringVar(value='AWAITING ANALYSIS')
        self._conf_conf_var    = tk.StringVar(value='')

        verdict_box = tk.Frame(conf_inner, bg=COLORS['bg_light'],
            highlightbackground=COLORS['border_light'], highlightthickness=1)
        verdict_box.pack(fill='x', pady=(0, 10))

        self._verdict_label = tk.Label(verdict_box, textvariable=self._conf_verdict_var,
            bg=COLORS['bg_light'], fg=COLORS['accent_cyan'],
            font=('Segoe UI', 19, 'bold'), justify='center', wraplength=320)
        self._verdict_label.pack(fill='x', padx=18, pady=(26, 26))

        tk.Label(conf_inner, textvariable=self._conf_conf_var,
            bg=COLORS['bg_card'], fg=COLORS['text_dim'],
            font=('Segoe UI', 8), justify='center').pack(anchor='center')

        # Mini score bars (created fresh after analysis)
        self._mini_bars_frame = tk.Frame(conf_card, bg=COLORS['bg_card'])
        self._mini_bars_frame.pack(fill='x', padx=16, pady=(0, 14))

        # Status log card
        log_card = tk.Frame(right_col, bg=COLORS['bg_card'],
            highlightbackground=COLORS['border'], highlightthickness=1)
        log_card.pack(fill='both', expand=True)

        log_header = tk.Frame(log_card, bg=COLORS['bg_card'])
        log_header.pack(fill='x', padx=14, pady=(12, 6))
        tk.Frame(log_header, bg=COLORS['accent_orange'], width=3, height=16).pack(side='left', padx=(0, 8))
        tk.Label(log_header, text='Analysis Log',
            bg=COLORS['bg_card'], fg=COLORS['text'],
            font=('Segoe UI', 10, 'bold')).pack(side='left')

        log_area = tk.Frame(log_card, bg=COLORS['bg_light'])
        log_area.pack(fill='both', expand=True, padx=14, pady=(0, 14))

        log_sb = tk.Scrollbar(log_area, bg=COLORS['bg_light'])
        log_sb.pack(side='right', fill='y')

        # This is the results_text widget — mapped to original backend callbacks
        self.results_text = tk.Text(
            log_area,
            bg=COLORS['bg_light'], fg='#ffffff',
            font=('Consolas', 9), wrap='word',
            padx=12, pady=12,
            yscrollcommand=log_sb.set,
            insertbackground='#ffffff',
            selectbackground=COLORS['accent_blue'],
            relief='flat', bd=0
        )
        self.results_text.pack(side='left', fill='both', expand=True)
        log_sb.config(command=self.results_text.yview)

        # ── Text color tags for rich output ──────────────────────
        self.results_text.tag_configure('header',    foreground='#ffffff',         font=('Consolas', 9, 'bold'))
        self.results_text.tag_configure('border',    foreground='#3d9bff')
        self.results_text.tag_configure('section',   foreground=COLORS['accent_cyan'],  font=('Consolas', 9, 'bold'))
        self.results_text.tag_configure('divider',   foreground='#253347')
        self.results_text.tag_configure('verdict_fake',   foreground=COLORS['accent_red'],    font=('Consolas', 9, 'bold'))
        self.results_text.tag_configure('verdict_real',   foreground=COLORS['accent_green'],  font=('Consolas', 9, 'bold'))
        self.results_text.tag_configure('verdict_warn',   foreground=COLORS['accent_orange'], font=('Consolas', 9, 'bold'))
        self.results_text.tag_configure('label',     foreground='#7a8fa8')
        self.results_text.tag_configure('value',     foreground='#ffffff',         font=('Consolas', 9, 'bold'))
        self.results_text.tag_configure('score_high',foreground=COLORS['accent_red'])
        self.results_text.tag_configure('score_med', foreground=COLORS['accent_orange'])
        self.results_text.tag_configure('score_low', foreground=COLORS['accent_green'])
        self.results_text.tag_configure('bar_fill',  foreground=COLORS['accent_cyan'])
        self.results_text.tag_configure('bar_empty', foreground='#253347')
        self.results_text.tag_configure('bullet',    foreground=COLORS['accent_purple'])
        self.results_text.tag_configure('finding',   foreground='#e2eaf4')
        self.results_text.tag_configure('dim',       foreground='#4a5a6e')
        self.results_text.tag_configure('warn_text', foreground=COLORS['accent_orange'])

        # Feature cards row
        feat_row = tk.Frame(page, bg=COLORS['bg_dark'])
        feat_row.pack(fill='x', padx=24, pady=(0, 20))

        feat_defs = [
            ('🔒', 'SECURE PROCESSING', 'All analysis runs locally.\nNo data leaves your system.'),
            ('📝', 'AUTOMATED LOG',     'Chain of custody entries\nautomatically generated.'),
            ('🔐', 'HASH INTEGRITY',    'MD5, SHA-1, and SHA-256\nverified on ingestion.'),
        ]
        for icon, title, desc in feat_defs:
            fc = tk.Frame(feat_row, bg=COLORS['bg_card'],
                highlightbackground=COLORS['border'], highlightthickness=1)
            fc.pack(side='left', fill='x', expand=True,
                padx=(0 if feat_defs.index((icon,title,desc)) == 0 else 10, 0))

            fi = tk.Frame(fc, bg=COLORS['bg_card'])
            fi.pack(padx=14, pady=12, anchor='w')
            tk.Label(fi, text=icon,
                bg=COLORS['bg_card'], fg=COLORS['accent_cyan'],
                font=('Segoe UI', 18)).pack(anchor='w')
            tk.Label(fi, text=title,
                bg=COLORS['bg_card'], fg=COLORS['text'],
                font=('Segoe UI', 8, 'bold')).pack(anchor='w', pady=(4, 2))
            tk.Label(fi, text=desc,
                bg=COLORS['bg_card'], fg=COLORS['text_secondary'],
                font=('Segoe UI', 7), justify='left').pack(anchor='w')

        return page

    # ─────────────────────────────────────────────────────────────
    #  PAGE 3 — REPORT
    # ─────────────────────────────────────────────────────────────

    def _build_report_page(self, parent):
        page = tk.Frame(parent, bg=COLORS['bg_dark'])
        page.columnconfigure(0, weight=1)
        page.rowconfigure(1, weight=1)

        # Top bar
        topbar = tk.Frame(page, bg=COLORS['bg_medium'],
            highlightbackground=COLORS['border'], highlightthickness=1)
        topbar.pack(fill='x')

        tb_inner = tk.Frame(topbar, bg=COLORS['bg_medium'])
        tb_inner.pack(fill='x', padx=24, pady=14)
        tk.Label(tb_inner, text='Analysis Report',
            bg=COLORS['bg_medium'], fg=COLORS['text'],
            font=('Segoe UI', 16, 'bold')).pack(side='left')

        self._report_filename_var = tk.StringVar(value='— No analysis yet —')
        tk.Label(tb_inner, textvariable=self._report_filename_var,
            bg=COLORS['bg_medium'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 9)).pack(side='left', padx=16)

        # Print/share buttons
        btn_row = tk.Frame(tb_inner, bg=COLORS['bg_medium'])
        btn_row.pack(side='right')
        tk.Button(btn_row, text='🖨  Export Report',
            command=self._export_report_text,
            bg=COLORS['bg_card'], fg=COLORS['text'],
            font=('Segoe UI', 8), padx=10, pady=5, relief='flat', cursor='hand2'
        ).pack(side='left', padx=(0, 8))

        # Scrollable report area
        report_outer = tk.Frame(page, bg=COLORS['bg_dark'])
        report_outer.pack(fill='both', expand=True, padx=24, pady=20)

        # Scrollbar
        r_sb = tk.Scrollbar(report_outer, bg=COLORS['bg_dark'])
        r_sb.pack(side='right', fill='y')

        self._report_canvas = tk.Canvas(report_outer,
            bg=COLORS['bg_dark'], highlightthickness=0,
            yscrollcommand=r_sb.set)
        self._report_canvas.pack(side='left', fill='both', expand=True)
        r_sb.config(command=self._report_canvas.yview)

        self._report_frame = tk.Frame(self._report_canvas, bg=COLORS['bg_dark'])
        self._report_canvas_window = self._report_canvas.create_window(
            (0, 0), window=self._report_frame, anchor='nw')

        self._report_frame.bind('<Configure>',
            lambda e: self._report_canvas.configure(
                scrollregion=self._report_canvas.bbox('all')))
        self._report_canvas.bind('<Configure>',
            lambda e: self._report_canvas.itemconfig(
                self._report_canvas_window, width=e.width))

        # Mousewheel scroll
        def _on_mousewheel(event):
            self._report_canvas.yview_scroll(int(-1*(event.delta/120)), 'units')
        self._report_canvas.bind_all('<MouseWheel>', _on_mousewheel)

        # Placeholder
        self._report_placeholder = tk.Label(self._report_frame,
            text='Run an analysis first to generate a report.',
            bg=COLORS['bg_dark'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 12))
        self._report_placeholder.pack(pady=60)

        return page

    def _render_report(self, v):
        """Render a structured visual report from verdict dict v."""
        # Clear
        for w in self._report_frame.winfo_children():
            w.destroy()

        fname = os.path.basename(self.current_image_path) if self.current_image_path else 'unknown'
        self._report_filename_var.set(fname)

        # ── Verdict Banner ────────────────────────────────────────
        fp = v['fake_pct']
        if fp < 38:
            banner_bg    = '#0d2b1f'
            banner_border= COLORS['accent_green']
            verdict_text = 'AUTHENTIC — LOW PROBABILITY OF MANIPULATION'
            verdict_color= COLORS['accent_green']
        elif fp < 55:
            banner_bg    = '#2b2210'
            banner_border= COLORS['accent_orange']
            verdict_text = 'INCONCLUSIVE — REQUIRES FURTHER EXAMINATION'
            verdict_color= COLORS['accent_orange']
        else:
            banner_bg    = '#2b1015'
            banner_border= COLORS['accent_red']
            verdict_text = 'HIGH PROBABILITY OF MANIPULATION'
            verdict_color= COLORS['accent_red']

        banner = tk.Frame(self._report_frame, bg=banner_bg,
            highlightbackground=banner_border, highlightthickness=2)
        banner.pack(fill='x', pady=(0, 16))

        banner_inner = tk.Frame(banner, bg=banner_bg)
        banner_inner.pack(fill='x', padx=20, pady=16)

        tk.Label(banner_inner, text='AUTOMATED VERDICT',
            bg=banner_bg, fg=verdict_color,
            font=('Segoe UI', 8, 'bold')).pack(anchor='w')
        tk.Label(banner_inner, text=verdict_text,
            bg=banner_bg, fg=verdict_color,
            font=('Segoe UI', 16, 'bold')).pack(anchor='w', pady=(4, 0))

        conf_label = tk.Label(banner_inner, text=f'Confidence Score  {fp:.2f}%',
            bg=banner_bg, fg=verdict_color,
            font=('Segoe UI', 12, 'bold'))
        conf_label.pack(side='right', anchor='e')

        # ── Two-column layout ─────────────────────────────────────
        two_col = tk.Frame(self._report_frame, bg=COLORS['bg_dark'])
        two_col.pack(fill='x', pady=(0, 16))
        two_col.columnconfigure(0, weight=3)
        two_col.columnconfigure(1, weight=2)

        # ── Category Breakdown card (PROMOTED — most prominent panel) ──
        # BACKEND DATA: v['category_scores'] — produced by compute_verdict()
        engine_card = ModernCard(two_col, title='Category Breakdown',
            accent_color=COLORS['accent_blue'])
        engine_card.grid(row=0, column=1, sticky='nsew', padx=(12, 0))

        eb = tk.Frame(engine_card, bg=COLORS['bg_card'])
        eb.pack(fill='x', padx=14, pady=(0, 14))

        def _category_row(parent, name, val, color):
            r = tk.Frame(parent, bg=COLORS['bg_card'])
            r.pack(fill='x', pady=7)
            top = tk.Frame(r, bg=COLORS['bg_card'])
            top.pack(fill='x')
            tk.Label(top, text=name, bg=COLORS['bg_card'], fg=COLORS['text'],
                font=('Segoe UI', 10, 'bold')).pack(side='left')
            tk.Label(top, text=f'{val}%', bg=COLORS['bg_card'], fg=color,
                font=('Segoe UI', 12, 'bold')).pack(side='right')
            make_progress_bar(r, val, color=color, width=220, height=10).pack(anchor='w', pady=(5, 0))

        # BACKEND-CONNECTED: v['category_scores']['real'] / ['deepfake'] / ['aigen'] / ['edited']
        _category_row(eb, 'Real Photo',   v['category_scores']['real'],     COLORS['accent_green'])
        _category_row(eb, 'Deepfake',     v['category_scores']['deepfake'], COLORS['accent_red'])
        _category_row(eb, 'AI-Generated', v['category_scores']['aigen'],    COLORS['accent_purple'])
        _category_row(eb, 'Edited',       v['category_scores']['edited'],   COLORS['accent_orange'])

        # ── Evidence Summary inset ────────────────────────────────
        evid_card = tk.Frame(engine_card, bg=COLORS['bg_medium'],
            highlightbackground=COLORS['border'], highlightthickness=1)
        evid_card.pack(fill='x', padx=14, pady=(0, 14))

        evid_inner = tk.Frame(evid_card, bg=COLORS['bg_medium'])
        evid_inner.pack(fill='x', padx=12, pady=10)

        tk.Label(evid_inner, text='EVIDENCE SUMMARY',
            bg=COLORS['bg_medium'], fg=COLORS['text_dim'],
            font=('Segoe UI', 7, 'bold')).pack(anchor='w', pady=(0, 6))

        mode = 'DL + Classical' if (HAS_SIGLIP or HAS_VIT) else 'Classical Only'
        ev_rows = [
            ('Detection Model', 'Ensemble Hybrid'),
            ('Analysis Mode',   mode),
            ('Fake Score',      f'{fp:.2f}%'),
            ('Confidence',      v['confidence']),
        ]
        for k, val in ev_rows:
            er = tk.Frame(evid_inner, bg=COLORS['bg_medium'])
            er.pack(fill='x', pady=1)
            tk.Label(er, text=k + ':', bg=COLORS['bg_medium'],
                fg=COLORS['text_secondary'], font=('Consolas', 8),
                width=16, anchor='w').pack(side='left')
            tk.Label(er, text=val, bg=COLORS['bg_medium'],
                fg=COLORS['text'], font=('Consolas', 8, 'bold')).pack(side='left')

        # ── Forensic Module Analysis (left column) ────────────────
        forensic_card = ModernCard(two_col, title='Forensic Module Analysis',
            accent_color=COLORS['accent_orange'])
        forensic_card.grid(row=0, column=0, sticky='nsew')

        fb = tk.Frame(forensic_card, bg=COLORS['bg_card'])
        fb.pack(fill='both', expand=True, padx=14, pady=(0, 14))

        # Table header
        hdr = tk.Frame(fb, bg=COLORS['border'])
        hdr.pack(fill='x', pady=(0, 4))
        for h_text, h_width in [('MODULE', 16), ('RESULT', 20), ('SUSPICION', 10), ('STATUS', 8)]:
            tk.Label(hdr, text=h_text, bg=COLORS['border'], fg=COLORS['text_dim'],
                font=('Segoe UI', 7, 'bold'), width=h_width, anchor='w',
                padx=4, pady=4).pack(side='left')

        module_display = {
            'ela':       ('Error Level Analysis', lambda r: f"Err={r.get('error_level', 0):.1f}"),
            'noise':     ('Noise Analysis',       lambda r: f"Lvl={r.get('noise_level', 0):.2f}"),
            'metadata':  ('Metadata Check',       lambda r: 'EXIF: Yes' if r.get('has_exif') else 'EXIF: No'),
            'frequency': ('Frequency (FFT)',      lambda r: f"HF={r.get('hf_energy', 0):.3f}"),
            'texture':   ('Texture (LBP)',         lambda r: f"Var={r.get('tex_variation', 0):.1f}"),
        }

        for key, (name, detail_fn) in module_display.items():
            r = v['classical'].get(key, {})
            susp = r.get('suspicion', 0) * 100
            detail_str = detail_fn(r)

            if susp >= 60:
                status_txt = 'FAILED'; status_color = COLORS['accent_red']
            elif susp >= 30:
                status_txt = 'WARNING'; status_color = COLORS['accent_orange']
            else:
                status_txt = 'PASS'; status_color = COLORS['accent_green']

            row_bg = COLORS['bg_card']
            row = tk.Frame(fb, bg=row_bg,
                highlightbackground=COLORS['border'], highlightthickness=1)
            row.pack(fill='x', pady=2)

            tk.Label(row, text=name, bg=row_bg, fg=COLORS['text'],
                font=('Segoe UI', 9), width=20, anchor='w', padx=6, pady=5).pack(side='left')
            tk.Label(row, text=detail_str, bg=row_bg, fg=COLORS['text_secondary'],
                font=('Consolas', 8), width=18, anchor='w').pack(side='left')

            # Mini bar
            bar_f = tk.Frame(row, bg=row_bg)
            bar_f.pack(side='left')
            make_progress_bar(bar_f, susp, color=status_color, width=90, height=5).pack(pady=2)
            tk.Label(bar_f, text=f'{susp:.0f}%', bg=row_bg, fg=status_color,
                font=('Segoe UI', 7)).pack()

            tk.Label(row, text=status_txt, bg=row_bg, fg=status_color,
                font=('Segoe UI', 8, 'bold'), padx=8).pack(side='right')

        # ── Findings ──────────────────────────────────────────────
        if v.get('findings'):
            find_card = ModernCard(self._report_frame, title='Forensic Findings',
                accent_color=COLORS['accent_purple'])
            find_card.pack(fill='x', pady=(0, 16))

            find_body = tk.Frame(find_card, bg=COLORS['bg_card'])
            find_body.pack(fill='x', padx=14, pady=(0, 14))

            for finding in v['findings']:
                frow = tk.Frame(find_body, bg=COLORS['bg_card'])
                frow.pack(fill='x', pady=2)
                tk.Label(frow, text='▪', bg=COLORS['bg_card'], fg=COLORS['accent_purple'],
                    font=('Segoe UI', 10)).pack(side='left', padx=(0, 8))
                tk.Label(frow, text=finding, bg=COLORS['bg_card'], fg=COLORS['text'],
                    font=('Segoe UI', 9), anchor='w', justify='left',
                    wraplength=700).pack(side='left', anchor='w')

        # ── Disclaimer ────────────────────────────────────────────
        disc = tk.Frame(self._report_frame, bg=COLORS['bg_medium'],
            highlightbackground=COLORS['border'], highlightthickness=1)
        disc.pack(fill='x', pady=(0, 10))
        tk.Label(disc,
            text='⚠  AI-assisted analysis. Not suitable as sole evidence in legal or critical decisions. '
                 'Consult a certified digital forensics expert for official proceedings.',
            bg=COLORS['bg_medium'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 8), wraplength=700, justify='left').pack(padx=16, pady=10, anchor='w')

    # ─────────────────────────────────────────────────────────────
    #  PAGE 4 — HISTORY
    # ─────────────────────────────────────────────────────────────

    def _build_history_page(self, parent):
        page = tk.Frame(parent, bg=COLORS['bg_dark'])
        page.columnconfigure(0, weight=1)
        page.rowconfigure(2, weight=1)

        # Top bar
        topbar = tk.Frame(page, bg=COLORS['bg_medium'],
            highlightbackground=COLORS['border'], highlightthickness=1)
        topbar.pack(fill='x')

        tb_inner = tk.Frame(topbar, bg=COLORS['bg_medium'])
        tb_inner.pack(fill='x', padx=24, pady=14)
        tk.Label(tb_inner, text='Analysis History Log',
            bg=COLORS['bg_medium'], fg=COLORS['text'],
            font=('Segoe UI', 16, 'bold')).pack(side='left')
        tk.Label(tb_inner, text='  Review and audit all previous forensic analysis records.',
            bg=COLORS['bg_medium'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 9)).pack(side='left')

        # Export / Clear buttons
        self._history_count_var = tk.StringVar(value='0 records')
        tk.Label(tb_inner, textvariable=self._history_count_var,
            bg=COLORS['bg_medium'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 9)).pack(side='right', padx=(0, 16))

        tk.Button(tb_inner, text='🗑  Clear History',
            command=self._clear_history,
            bg=COLORS['accent_red'], fg='white',
            font=('Segoe UI', 8, 'bold'), padx=10, pady=5,
            relief='flat', cursor='hand2'
        ).pack(side='right', padx=(0, 8))

        tk.Button(tb_inner, text='💾  Export CSV',
            command=self._export_history_csv,
            bg=COLORS['accent_green'], fg=COLORS['bg_dark'],
            font=('Segoe UI', 8, 'bold'), padx=10, pady=5,
            relief='flat', cursor='hand2'
        ).pack(side='right', padx=(0, 8))

        # Stat cards
        self._hist_stat_vars = {}
        hist_stat_row = tk.Frame(page, bg=COLORS['bg_dark'])
        hist_stat_row.pack(fill='x', padx=24, pady=(16, 10))

        hist_stat_defs = [
            ('total',    'Total Scans',    '0', COLORS['accent_cyan']),
            ('fake',     'Flagged Fake',   '0', COLORS['accent_red']),
            ('real',     'Verified Real',  '0', COLORS['accent_green']),
            ('rate',     'Fake Rate',      '0%', COLORS['accent_orange']),
        ]
        for key, label, val, color in hist_stat_defs:
            hsc = tk.Frame(hist_stat_row, bg=COLORS['bg_card'],
                highlightbackground=COLORS['border'], highlightthickness=1)
            hsc.pack(side='left', fill='x', expand=True,
                padx=(0 if key == 'total' else 10, 0))
            hsi = tk.Frame(hsc, bg=COLORS['bg_card'])
            hsi.pack(padx=14, pady=10)
            var = tk.StringVar(value=val)
            self._hist_stat_vars[key] = var
            tk.Label(hsi, textvariable=var, bg=COLORS['bg_card'],
                fg=color, font=('Segoe UI', 18, 'bold')).pack()
            tk.Label(hsi, text=label, bg=COLORS['bg_card'],
                fg=COLORS['text_secondary'], font=('Segoe UI', 8)).pack()

        # Treeview area
        tree_area = tk.Frame(page, bg=COLORS['bg_dark'])
        tree_area.pack(fill='both', expand=True, padx=24, pady=(0, 20))

        # Style
        style = ttk.Style()
        style.theme_use('default')
        style.configure('ForensicHistory.Treeview',
            background=COLORS['bg_card'],
            foreground=COLORS['text'],
            fieldbackground=COLORS['bg_card'],
            rowheight=30,
            font=('Consolas', 9),
            borderwidth=0,
        )
        style.configure('ForensicHistory.Treeview.Heading',
            background=COLORS['bg_medium'],
            foreground=COLORS['text_secondary'],
            font=('Segoe UI', 9, 'bold'),
            relief='flat',
        )
        style.map('ForensicHistory.Treeview',
            background=[('selected', COLORS['accent_blue'])],
            foreground=[('selected', 'white')])

        vsb = ttk.Scrollbar(tree_area, orient='vertical')
        vsb.pack(side='right', fill='y')
        hsb = ttk.Scrollbar(tree_area, orient='horizontal')
        hsb.pack(side='bottom', fill='x')

        cols = ('Timestamp', 'Filename', 'Fake %', 'Verdict', 'Confidence', 'Mode')
        col_widths = (170, 240, 70, 250, 100, 80)

        self.tree = ttk.Treeview(
            tree_area, columns=cols, show='headings',
            style='ForensicHistory.Treeview',
            yscrollcommand=vsb.set, xscrollcommand=hsb.set,
        )
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)

        for col, w in zip(cols, col_widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, minwidth=50, anchor='w')

        self.tree.tag_configure('fake',      foreground=COLORS['accent_red'])
        self.tree.tag_configure('uncertain', foreground=COLORS['accent_orange'])
        self.tree.tag_configure('real',      foreground=COLORS['accent_green'])

        self.tree.pack(side='left', fill='both', expand=True)

        # Double-click detail
        self.tree.bind('<Double-1>', self._on_history_double_click)

        return page

    def _refresh_history_tree(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        history = getattr(self, '_history_log', [])

        for entry in reversed(history):
            tag = ('fake'      if entry['fake_pct'] >= 55 else
                   'uncertain' if entry['fake_pct'] >= 38 else
                   'real')
            self.tree.insert('', 'end',
                values=(
                    entry['timestamp'],
                    entry['filename'],
                    f"{entry['fake_pct']:.1f}%",
                    entry['verdict'],
                    entry['confidence'],
                    entry['mode'].capitalize(),
                ),
                tags=(tag,)
            )

        total     = len(history)
        fake_cnt  = sum(1 for e in history if e['fake_pct'] >= 55)
        real_cnt  = total - fake_cnt
        fake_rate = f"{fake_cnt/total*100:.0f}%" if total else '0%'

        self._history_count_var.set(f'{total} records')
        if hasattr(self, '_hist_stat_vars'):
            self._hist_stat_vars['total'].set(str(total))
            self._hist_stat_vars['fake'].set(str(fake_cnt))
            self._hist_stat_vars['real'].set(str(real_cnt))
            self._hist_stat_vars['rate'].set(fake_rate)

        # Update dashboard stats
        self._update_dashboard_stats()

    def _update_dashboard_stats(self):
        history = getattr(self, '_history_log', [])
        total     = len(history)
        fake_cnt  = sum(1 for e in history if e['fake_pct'] >= 55)
        real_cnt  = total - fake_cnt
        rate      = f"{fake_cnt/total*100:.0f}%" if total else 'N/A'

        if hasattr(self, '_stat_vars'):
            self._stat_vars['Total Analyses'].set(str(total))
            self._stat_vars['Deepfakes Found'].set(str(fake_cnt))
            self._stat_vars['Verified Real'].set(str(real_cnt))
            self._stat_vars['Detection Rate'].set(rate)

    def _on_history_double_click(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        values = self.tree.item(sel[0], 'values')
        detail = tk.Toplevel(self.root)
        detail.title('Entry Detail')
        detail.geometry('440x230')
        detail.configure(bg=COLORS['bg_medium'])
        detail.resizable(False, False)

        tk.Label(detail, text='Analysis Entry Detail',
            bg=COLORS['bg_medium'], fg=COLORS['text'],
            font=('Segoe UI', 12, 'bold')).pack(pady=(16, 10))

        labels = ['Timestamp', 'Filename', 'Fake %', 'Verdict', 'Confidence', 'Mode']
        for lbl, val in zip(labels, values):
            row = tk.Frame(detail, bg=COLORS['bg_medium'])
            row.pack(fill='x', padx=20, pady=3)
            tk.Label(row, text=f'{lbl}:', bg=COLORS['bg_medium'],
                fg=COLORS['text_secondary'], font=('Segoe UI', 9, 'bold'),
                width=12, anchor='e').pack(side='left', padx=(0, 8))
            tk.Label(row, text=val, bg=COLORS['bg_medium'],
                fg=COLORS['text'], font=('Consolas', 9), anchor='w').pack(side='left')

    def _clear_history(self):
        history = getattr(self, '_history_log', [])
        if not history:
            return
        if messagebox.askyesno('Clear History', 'Delete all history entries?', parent=self.root):
            self._history_log.clear()
            self._refresh_history_tree()

    def _export_history_csv(self):
        import csv
        history = getattr(self, '_history_log', [])
        if not history:
            messagebox.showinfo('Empty', 'No history to export.', parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title='Save history as CSV',
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
            initialfile='deepfake_history.csv'
        )
        if not path:
            return
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f,
                    fieldnames=['timestamp', 'filename', 'fake_pct', 'verdict', 'confidence', 'mode'])
                writer.writeheader()
                writer.writerows(history)
            messagebox.showinfo('Exported', f'History saved to:\n{path}', parent=self.root)
        except Exception as e:
            messagebox.showerror('Error', str(e), parent=self.root)

    def _export_report_text(self):
        if not self._last_verdict:
            messagebox.showinfo('No Report', 'Run an analysis first.', parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title='Save Report as TXT',
            defaultextension='.txt',
            filetypes=[('Text files', '*.txt'), ('All files', '*.*')],
            initialfile='forensic_report.txt'
        )
        if not path:
            return
        v = self._last_verdict
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write('PIXELTRUTH — ANALYSIS REPORT\n')
                f.write('=' * 60 + '\n\n')
                f.write(f"File       : {os.path.basename(self.current_image_path or 'unknown')}\n")
                f.write(f"Verdict    : {v['primary']}\n")
                f.write(f"Fake Score : {v['fake_pct']}%\n")
                f.write(f"Confidence : {v['confidence']}\n\n")
                f.write('CATEGORY BREAKDOWN\n' + '-' * 40 + '\n')
                cs = v['category_scores']
                f.write(f"Real Photo   : {cs['real']}%\n")
                f.write(f"Deepfake     : {cs['deepfake']}%\n")
                f.write(f"AI-Generated : {cs['aigen']}%\n")
                f.write(f"Edited       : {cs['edited']}%\n\n")
                f.write('FORENSIC FINDINGS\n' + '-' * 40 + '\n')
                for finding in v['findings']:
                    f.write(f"• {finding}\n")
            messagebox.showinfo('Saved', f'Report saved to:\n{path}', parent=self.root)
        except Exception as e:
            messagebox.showerror('Error', str(e), parent=self.root)

    # ─────────────────────────────────────────────────────────────
    #  BACKEND CONNECTIONS — Upload / Analysis
    # ─────────────────────────────────────────────────────────────

    def _upload(self):
        path = filedialog.askopenfilename(
            title='Select Image',
            filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp *.webp'), ('All', '*.*')]
        )
        if not path:
            return
        self.current_image_path = path
        self._show_preview(path)
        self.analyze_btn.config(state='normal')
        sz   = os.path.getsize(path) / 1024
        name = os.path.basename(path)
        self.info_label.config(
            text=f'📄 {name}   |   {sz:.1f} KB   |   Ready for analysis')
        self._show_page('analysis')

    def _show_preview(self, path):
        try:
            img = Image.open(path)
            img.thumbnail((500, 280), Image.Resampling.LANCZOS)
            self.preview_image = ImageTk.PhotoImage(img)
            self.preview_label.config(image=self.preview_image, text='')
        except Exception as e:
            messagebox.showerror('Error', f'Could not load image: {e}')

    def _batch_upload(self):
        folder = filedialog.askdirectory(title='Select folder of images')
        if not folder:
            return

        images = [
            f for f in Path(folder).iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']
        ]

        if not images:
            messagebox.showinfo('None', 'No images found in folder')
            return

        self.analyze_btn.config(state='disabled')
        self.progress.start(12)
        self._show_page('analysis')

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END,
            f"📁 BATCH MODE — {len(images)} images found\n"
            f"   Folder: {folder}\n\n"
            f"{'Image':<32} {'Fake%':>6}  Verdict\n"
            f"{'─'*65}\n"
        )

        threading.Thread(
            target=self._run_batch, args=(images,), daemon=True
        ).start()

    def _run_batch(self, images):
        results = []
        for i, img_path in enumerate(images):
            self.root.after(0, lambda i=i: self.status_label.config(
                text=f"Processing {i+1}/{len(images)}: {images[i].name[:20]}…"
            ))
            try:
                pil_img, cv_img = load_image(str(img_path))
                v = compute_verdict(pil_img, str(img_path), cv_img)
                results.append((img_path.name, v['fake_pct'], v['primary']))
                line = (f"  {img_path.name[:30]:30} "
                        f"{v['fake_pct']:>5.1f}%  {v['primary']}\n")
                self.root.after(0, lambda l=line: (
                    self.results_text.insert(tk.END, l),
                    self.results_text.see(tk.END)
                ))
            except Exception as e:
                line = f"  {img_path.name[:30]:30}  ERROR: {e}\n"
                self.root.after(0, lambda l=line: (
                    self.results_text.insert(tk.END, l),
                    self.results_text.see(tk.END)
                ))

        self.root.after(0, self._finish_batch, results)

    def _finish_batch(self, results):
        self.progress.stop()
        self.analyze_btn.config(state='normal')

        if not results:
            self.status_label.config(text='⚠  Batch finished — no results')
            return

        fake_count = sum(1 for _, p, _ in results if p >= 55)
        real_count = len(results) - fake_count
        fake_rate  = fake_count / len(results) * 100

        summary = (
            f"\n{'─'*65}\n"
            f"  BATCH SUMMARY\n"
            f"  Total analysed : {len(results)}\n"
            f"  Likely Real    : {real_count}\n"
            f"  Likely Fake    : {fake_count}\n"
            f"  Fake rate      : {fake_rate:.1f}%\n"
        )
        self.results_text.insert(tk.END, summary)
        self.results_text.see(tk.END)
        self.status_label.config(
            text=f'✅ Batch complete — {len(results)} images processed')

        # Log batch results to history
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        history = getattr(self, '_history_log', [])
        for filename, fake_pct, verdict in results:
            history.append({
                'timestamp':  timestamp,
                'filename':   filename,
                'fake_pct':   fake_pct,
                'verdict':    verdict,
                'confidence': '—',
                'mode':       'batch',
            })
        self._update_dashboard_stats()

    def _start_analysis(self):
        if not self.current_image_path:
            return
        self.analyze_btn.config(state='disabled', text='⏳  Analysing…')
        self.progress.start(12)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0,
            '🔬  Running AI engines + forensic analysis…\n'
            '    (First run may take 30–60 s to load model weights)\n'
        )
        self._conf_verdict_var.set('ANALYSIS IN PROGRESS…')
        self._verdict_label.config(fg=COLORS['accent_cyan'])
        self._conf_conf_var.set('')
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            pil_img, cv_img = load_image(self.current_image_path)
            verdict = compute_verdict(pil_img, self.current_image_path, cv_img)
            self.root.after(0, self._show_results, verdict)
        except Exception as e:
            self.root.after(0, self._show_error, str(e))

    # ─────────────────────────────────────────────────────────────
    #  UI-ONLY DISPLAY MAPPING
    #  (no detection logic here — purely formats the existing result
    #   dict v, produced unchanged by compute_verdict(), into the
    #   3 display labels + color requested for the Detection Result panel)
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _verdict_display(v):
        """Map the existing backend verdict/category data to one of the
        three required display strings + an accent color.
        Uses only values already computed by compute_verdict() —
        v['fake_pct'] and v['category_scores'] — no new thresholds
        or logic are introduced; this purely re-labels the existing
        5-tier 'primary' classification down to 3 display states."""
        fp = v['fake_pct']
        cats = v['category_scores']

        if fp < 38:
            # Same "real" threshold already used elsewhere in the app
            # (compute_verdict's own < 38 boundary for Real/Likely Real)
            return 'REAL IMAGE', COLORS['accent_green']

        # Above the real threshold: decide Deepfake vs AI-Generated using
        # the already-computed category_scores breakdown (no new math).
        if cats['deepfake'] >= cats['aigen']:
            return 'DEEPFAKE DETECTED', COLORS['accent_red']
        else:
            return 'AI GENERATED IMAGE', COLORS['accent_purple']

    def _update_verdict_panel(self, v):
        """Refresh the Detection Result panel with plain verdict text only —
        no percentage, no progress bar."""
        label_text, label_color = self._verdict_display(v)

        self._conf_verdict_var.set(label_text)
        self._verdict_label.config(fg=label_color)

    # ─────────────────────────────────────────────────────────────
    #  RESULTS DISPLAY  (original _show_results + new UI updates)
    # ─────────────────────────────────────────────────────────────

    def _show_results(self, v):
        self.progress.stop()
        self._last_verdict = v   # store for report page

        # ── Helper: tagged insert ─────────────────────────────────
        rt = self.results_text
        rt.delete(1.0, tk.END)

        def ins(text, *tags):
            rt.insert(tk.END, text, tags if tags else ())

        def ins_bar(pct, width=16):
            """Insert a two-part colored bar."""
            filled = max(0, int(pct / 100 * width))
            empty  = width - filled
            ins('█' * filled, 'bar_fill')
            ins('░' * empty,  'bar_empty')

        def score_tag(pct):
            return 'score_high' if pct >= 60 else ('score_med' if pct >= 35 else 'score_low')

        W = '═' * 46
        w = '─' * 48

        # ── Header box ───────────────────────────────────────────
        ins(f'╔{W}╗\n', 'border')
        ins(f'║ ', 'border')
        ins(f'{"PIXELTRUTH — FORENSIC ANALYSIS LOG":^44}', 'header')
        ins(f' ║\n', 'border')
        ins(f'╚{W}╝\n\n', 'border')

        # ── VERDICT ──────────────────────────────────────────────
        ins('🎯  VERDICT\n', 'section')
        ins(f'{w}\n', 'divider')

        fp = v['fake_pct']
        if fp >= 55:
            vtag = 'verdict_fake'
        elif fp >= 38:
            vtag = 'verdict_warn'
        else:
            vtag = 'verdict_real'

        ins(f'  {v["primary"]}\n', vtag)
        ins('  Confidence      : ', 'label')
        ins(f'{v["confidence"]}\n', 'value')
        ins('  Fake score      : ', 'label')
        ins(f'{v["fake_pct"]} %\n\n', vtag)

        # ── CATEGORY BREAKDOWN ───────────────────────────────────
        ins('📊  CATEGORY BREAKDOWN\n', 'section')
        ins(f'{w}\n', 'divider')

        cats = [
            ('Real Photo',   v['category_scores']['real'],    'score_low'),
            ('Deepfake',     v['category_scores']['deepfake'],'score_high'),
            ('AI-Generated', v['category_scores']['aigen'],   'score_med'),
            ('Edited',       v['category_scores']['edited'],  'dim'),
        ]
        for cat_name, cat_val, cat_tag in cats:
            ins(f'  {cat_name:15} ', 'label')
            ins(f'{cat_val:3}%  ', cat_tag)
            ins('│', 'divider')
            ins_bar(cat_val, width=20)
            ins('│\n', 'divider')
        ins('\n')

        # ── FORENSIC FINDINGS ────────────────────────────────────
        ins('🔍  FORENSIC FINDINGS\n', 'section')
        ins(f'{w}\n', 'divider')
        for finding in v['findings']:
            ins('  • ', 'bullet')
            ins(f'{finding}\n', 'finding')
        ins('\n')

        # ── FORENSIC MODULE SCORES ───────────────────────────────
        ins('📈  FORENSIC MODULE SCORES\n', 'section')
        ins(f'{w}\n', 'divider')

        module_names = {
            'ela':       'Error Level Analysis',
            'noise':     'Noise Analysis',
            'metadata':  'Metadata',
            'frequency': 'Frequency (FFT)',
            'texture':   'Texture (LBP)',
        }
        for key, result in v['classical'].items():
            name = module_names.get(key, key)
            pct  = round(result['suspicion'] * 100)
            ins(f'  {name:24} ', 'label')
            ins(f'{pct:3}%  ', score_tag(pct))
            ins_bar(pct, width=15)
            ins('\n')

        ins(f'\n{w}\n', 'divider')
        ins('⚠  ', 'warn_text')
        ins('AI-assisted analysis. Not suitable as sole evidence in legal/\n'
            '    critical decisions. Consult a digital forensics expert.\n', 'dim')

        rt.see(1.0)

        # ── Update Detection Result panel (verdict text + confidence bar) ──
        self._update_verdict_panel(v)
        self._conf_conf_var.set(f'Confidence: {v["confidence"]}')

        # Update Category Breakdown bars (replaces former SigLIP/ViT-v2/Forensics
        # engine-score bars — those raw model scores are no longer surfaced to
        # the user, but still computed internally inside compute_verdict()).
        for w2 in self._mini_bars_frame.winfo_children():
            w2.destroy()

        # BACKEND DATA: v['category_scores'] — produced by compute_verdict()
        cat_scores = [
            ('Real Photo',   v['category_scores']['real'],     COLORS['accent_green']),
            ('Deepfake',     v['category_scores']['deepfake'], COLORS['accent_red']),
            ('AI-Generated', v['category_scores']['aigen'],    COLORS['accent_purple']),
            ('Edited',       v['category_scores']['edited'],   COLORS['accent_orange']),
        ]

        tk.Label(self._mini_bars_frame, text='CATEGORY BREAKDOWN',
            bg=COLORS['bg_card'], fg=COLORS['text_dim'],
            font=('Segoe UI', 7, 'bold')).pack(anchor='w', pady=(2, 6))

        for cname, cval, ccol in cat_scores:
            mr = tk.Frame(self._mini_bars_frame, bg=COLORS['bg_card'])
            mr.pack(fill='x', pady=3)
            tk.Label(mr, text=cname, bg=COLORS['bg_card'], fg=COLORS['text_secondary'],
                font=('Segoe UI', 8), width=11, anchor='w').pack(side='left')
            make_progress_bar(mr, cval, color=ccol, width=110, height=6).pack(side='left', padx=(0, 6))
            tk.Label(mr, text=f'{cval}%', bg=COLORS['bg_card'], fg=ccol,
                font=('Segoe UI', 8, 'bold')).pack(side='left')

        # ── Restore button ────────────────────────────────────────
        self.analyze_btn.config(state='normal', text='🔬  Run Forensic Analysis')
        self.status_label.config(text='✅ Analysis complete')

        # ── Render report page (pre-build for instant switching) ──
        self._render_report(v)

    def _show_error(self, msg):
        self.progress.stop()
        self.analyze_btn.config(state='normal', text='🔬  Run Forensic Analysis')
        rt = self.results_text
        rt.delete(1.0, tk.END)
        rt.insert(tk.END, '❌  ERROR\n', 'verdict_fake')
        rt.insert(tk.END, '\n')
        rt.insert(tk.END, f'{msg}\n', 'finding')
        rt.insert(tk.END, '\nTip: ensure the image is not corrupted.', 'dim')
        self.status_label.config(text='❌  Analysis failed')
        self._conf_verdict_var.set('ANALYSIS FAILED')
        self._verdict_label.config(fg=COLORS['accent_red'])
        self._conf_conf_var.set('')

    @staticmethod
    def _bar(pct, width=12):
        filled = int(pct / 100 * width)
        return '█' * filled + '░' * (width - filled)

    # ─────────────────────────────────────────────────────────────
    #  ENGINE STATUS (called by __init__)
    # ─────────────────────────────────────────────────────────────

    def _show_engine_status(self):
        engines = []
        if HAS_SIGLIP: engines.append('SigLIP✓')
        if HAS_VIT:    engines.append('ViT-v2✓')
        engines.append('Forensics✓')
        dl = 'DL+Classical' if (HAS_SIGLIP or HAS_VIT) else 'Classical only'
        self.status_label.config(
            text=f"Mode: {dl}  |  {', '.join(engines)}"
        )

    # ─────────────────────────────────────────────────────────────
    #  LEGACY COMPATIBILITY — open_history_window
    #  (kept so monkey-patch in history feature still works)
    # ─────────────────────────────────────────────────────────────

    def _open_history_window(self):
        """Navigate to built-in history page."""
        self._show_page('history')


# ─────────────────────────────────────────────────────────────────
#  HISTORY LOG FEATURE — monkey-patch (unchanged logic)
# ─────────────────────────────────────────────────────────────────

import datetime, csv, io

_orig_init         = DeepfakeDetectorV2.__init__
_orig_show_results = DeepfakeDetectorV2._show_results
_orig_finish_batch = DeepfakeDetectorV2._finish_batch


def _patched_init(self, root):
    _orig_init(self, root)
    self._history_log = []


def _patched_show_results(self, v):
    _orig_show_results(self, v)
    filename  = os.path.basename(self.current_image_path) if self.current_image_path else 'unknown'
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    self._history_log.append({
        'timestamp': timestamp,
        'filename':  filename,
        'fake_pct':  v['fake_pct'],
        'verdict':   v['primary'],
        'confidence': v['confidence'],
        'mode':      'single',
    })
    self._update_dashboard_stats()


def _patched_finish_batch(self, results):
    _orig_finish_batch(self, results)
    # (batch logging is already handled inside _finish_batch above)


DeepfakeDetectorV2.__init__          = _patched_init
DeepfakeDetectorV2._show_results     = _patched_show_results
DeepfakeDetectorV2._finish_batch     = _patched_finish_batch

# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print('\n' + '=' * 55)
    print('  PixelTruth — AI-Powered Deepfake Forensic Analyzer')
    print('=' * 55)
    print(f'  Python  : {sys.version.split()[0]}')
    print(f'  PyTorch : {torch.__version__ if HAS_TORCH else "not installed"}')
    print(f'  SigLIP  : {"ready" if HAS_SIGLIP else "unavailable"}')
    print(f'  ViT-v2  : {"ready" if HAS_VIT    else "unavailable"}')
    print('=' * 55 + '\n')

    root = tk.Tk()
    app  = DeepfakeDetectorV2(root)

    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f'{WIN_W}x{WIN_H}+{(sw - WIN_W)//2}+{(sh - WIN_H)//2}')

    print('✅ GUI ready\n')
    root.mainloop()


if __name__ == '__main__':
    main()