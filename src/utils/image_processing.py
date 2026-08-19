import os

# cv2 is imported lazily inside canny_edge_detection. It is only needed when a run
# sets need_edges=True (default False), but a module-level import made the whole
# agent graph unimportable without opencv, since graph.py imports this module for
# convert_to_grayscale as well.

def canny_edge_detection(input_image_path: str, output_dir: str, filename: str, low_threshold: int = 50, high_threshold: int = 150):
    """
    Applies Canny edge detection to an image and saves the result.

    Args:
        input_image_path (str): Path to the input image.
        output_dir (str): Directory to save the output image.
        filename (str): The name for the output file.
        low_threshold (int): Lower threshold for the Canny algorithm.
        high_threshold (int): Higher threshold for the Canny algorithm.

    Returns:
        str: The path to the saved edge-detected image.
    """
    import cv2

    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {input_image_path}")
    
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    
    edges = cv2.Canny(blurred_img, low_threshold, high_threshold)
    
    # Invert colors to get black edges on a white background, which is more typical for sketches.
    inverted_edges = cv2.bitwise_not(edges)
    
    os.makedirs(output_dir, exist_ok=True)
    output_image_path = os.path.join(output_dir, filename)
    
    cv2.imwrite(output_image_path, inverted_edges)
    print(f"Canny edge detection successful. Image saved to {output_image_path}")
    
    return output_image_path

def convert_to_grayscale(image_path: str):
    """Return a grayscale copy of an image as a PIL Image.

    Moved here from the standalone `image_tools` module, which was imported by
    the agent graphs but never shipped in this repository -- the graphs failed at
    import time on any fresh checkout. Copying the original module wholesale was
    not an option: it imports google.genai and src.utils.input_processor at module
    level, which would reintroduce the vendor SDK dependency that the APIYI
    refactor removed.

    Note the return type: a PIL Image, not a path, matching what the call sites
    in preprocess_style_image_node actually use (they call .save() on it).
    The original returned "" on failure, which then raised an opaque
    AttributeError at the call site; failing loudly here is more useful.
    """
    from PIL import Image

    try:
        return Image.open(image_path).convert("L")
    except Exception as e:
        raise RuntimeError(f"Failed to convert {image_path} to grayscale: {e}") from e
