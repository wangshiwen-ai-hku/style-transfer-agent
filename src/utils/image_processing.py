import cv2
import numpy as np
import os

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
