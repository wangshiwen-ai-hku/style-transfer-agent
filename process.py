from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def step1_fft(image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read image, compute 2D FFT and return shifted spectrum and magnitude.

    Returns (fshift, magnitude_spectrum, original_array).
    """
    img = Image.open(image_path).convert("L")
    img.save("gray.png")
    arr = np.asarray(img, dtype=float)
    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1.0)
    return fshift, magnitude_spectrum, arr


def step2_visual_fft(magnitude_spectrum: np.ndarray, out_path: str) -> str:
    """Save a visualization of the FFT magnitude spectrum to disk.

    Normalizes the magnitude to 0-255 and writes a grayscale PNG.
    Returns the output path.
    """
    mag = magnitude_spectrum
    mag_min = float(np.min(mag))
    mag_max = float(np.max(mag))
    scaled = 255.0 * (mag - mag_min) / (mag_max - mag_min + 1e-12)
    img = Image.fromarray(np.clip(scaled, 0, 255).astype(np.uint8))
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_file)
    return str(out_file)


def step3_filter_fft(fshift: np.ndarray, filter_type: str = "lowpass", radius: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a simple circular frequency-domain filter to the shifted FFT.

    Supported filter_type: 'lowpass', 'highpass'.
    Returns (fshift_filtered, mask).
    """
    rows, cols = fshift.shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    # distance = np.sqrt((Y - crow) ** 2 + (X - ccol) ** 2)
    # mask = np.zeros((rows, cols), dtype=float)
    # if filter_type == "lowpass":
    #     mask[distance <= radius] = 1.0
    # elif filter_type == "highpass":
    #     mask[distance > radius] = 1.0
    # else:
    #     raise ValueError(f"Unsupported filter_type: {filter_type}")
    print("shape of fshift", fshift.shape)
    mask = np.ones((rows, cols), dtype=float)
    mask[crow-radius:crow+radius, :] = 0.0
    mask[:, ccol-radius:ccol+radius] = 0.0
    fshift_filtered = fshift * mask
    scaled = 20 * np.log(np.abs(fshift_filtered) + 1.0)
    sceled = 255.0 * (scaled - np.min(scaled)) / (np.max(scaled) - np.min(scaled) + 1e-12)
    Image.fromarray(np.clip(sceled, 0, 255).astype(np.uint8)).save(out_directory / "fshift_filtered.png")
    return fshift_filtered, mask


def step4_inverse_filtered_fft(fshift_filtered: np.ndarray) -> np.ndarray:
    """Compute the inverse FFT from a shifted, filtered frequency-domain array.

    Returns a 2D uint8 image array (0-255).
    """
    f_ishift = np.fft.ifftshift(fshift_filtered)
    
    img_back_complex = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back_complex)
    img_back -= img_back.min()
    denom = img_back.max() if img_back.max() != 0 else 1.0
    img_back = img_back / denom * 255.0
    return np.clip(img_back, 0, 255).astype(np.uint8)


def step5_visual_inverse_filtered_fft(img_back_array: np.ndarray, out_path: str) -> str:
    """Save the inverse-transformed image array to disk as PNG and return the path."""
    img = Image.fromarray(img_back_array)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_file)
    return str(out_file)


def run_pipeline(input_image_path: str, out_dir: str = "./out", filter_radius: int = 50) -> dict:
    """Convenience function: run full pipeline and write outputs.

    Outputs written:
      - {out_dir}/fft_magnitude.png
      - {out_dir}/ifft_filtered.png

    Returns a dict with output paths.
    """
    global out_directory
    out_directory = Path(out_dir) / Path(input_image_path).stem
    out_directory.mkdir(parents=True, exist_ok=True)
    fshift, magnitude, _ = step1_fft(input_image_path)
    mag_path = str(out_directory / "fft_magnitude.png")
    step2_visual_fft(magnitude, mag_path)
    fshift_filtered, _ = step3_filter_fft(fshift, filter_type="lowpass", radius=filter_radius)
    img_back = step4_inverse_filtered_fft(fshift_filtered)
    ifft_path = str(out_directory / "ifft_filtered.png")
    step5_visual_inverse_filtered_fft(img_back, ifft_path)
    return {"magnitude_path": mag_path, "ifft_path": ifft_path}

def rgb_to_ycbcr_array(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 array (H,W,3) to YCbCr array (float, 0-255).

    Uses ITU-R BT.601 conversion.
    """
    if rgb.dtype != np.float64 and rgb.dtype != np.float32:
        rgb = rgb.astype(np.float64)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    ycbcr = np.stack([y, cb, cr], axis=-1)
    return ycbcr


def ycbcr_to_rgb_array(ycbcr: np.ndarray) -> np.ndarray:
    """Convert YCbCr array (float, 0-255) back to RGB uint8 array.

    Uses inverse of ITU-R BT.601 conversion and clips to 0-255.
    """
    if ycbcr.dtype != np.float64 and ycbcr.dtype != np.float32:
        ycbcr = ycbcr.astype(np.float64)
    y = ycbcr[..., 0]
    cb = ycbcr[..., 1] - 128.0
    cr = ycbcr[..., 2] - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def run_pipeline_uv(input_image_path: str, out_dir: str = "./out_uv", filter_radius: int = 50) -> dict:
    """Run pipeline converting RGB->YCbCr, processing Cb/Cr in frequency domain, and converting back.

    Saves magnitude visualization and final reconstructed RGB image under
    {out_dir}/{stem}/ with names `fft_uv_magnitude.png` and `ifft_uv_rgb.png`.
    """
    global out_directory
    out_directory = Path(out_dir) / Path(input_image_path).stem
    out_directory.mkdir(parents=True, exist_ok=True)

    img = Image.open(input_image_path).convert("RGB")
    arr_rgb = np.asarray(img)
    ycbcr = rgb_to_ycbcr_array(arr_rgb)
    y = ycbcr[..., 0]
    cb = ycbcr[..., 1]
    cr = ycbcr[..., 2]

    # FFT on Cb and Cr channels (operate on float arrays)
    f_cb = np.fft.fftshift(np.fft.fft2(cb))
    f_cr = np.fft.fftshift(np.fft.fft2(cr))
    mag_cb = 20 * np.log(np.abs(f_cb) + 1.0)
    mag_cr = 20 * np.log(np.abs(f_cr) + 1.0)
    combined_mag = np.sqrt(mag_cb ** 2 + mag_cr ** 2)
    mag_path = str(out_directory / "fft_uv_magnitude.png")
    step2_visual_fft(combined_mag, mag_path)

    # Filter both channels
    f_cb_filt, _ = step3_filter_fft(f_cb, filter_type="lowpass", radius=filter_radius)
    f_cr_filt, _ = step3_filter_fft(f_cr, filter_type="lowpass", radius=filter_radius)

    # Inverse transform
    cb_back = step4_inverse_filtered_fft(f_cb_filt)
    cr_back = step4_inverse_filtered_fft(f_cr_filt)

    # cb_back and cr_back are uint8 (0-255), y may be float - convert y to uint8
    y_uint8 = np.clip(y, 0, 255).astype(np.uint8)
    ycbcr_back = np.stack([y_uint8, cb_back, cr_back], axis=-1)
    rgb_back = ycbcr_to_rgb_array(ycbcr_back)

    out_rgb_path = str(out_directory / "ifft_uv_rgb.png")
    Image.fromarray(rgb_back).save(out_rgb_path)
    return {"magnitude_path": mag_path, "ifft_rgb_path": out_rgb_path}

# run_pipeline_uv("tree2.png")
run_pipeline("contents/gt.jpg")
run_pipeline("styles/style.png")
run_pipeline("tests/tree2.png")
run_pipeline("tests/tree3.png")