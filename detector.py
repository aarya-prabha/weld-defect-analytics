import cv2
import numpy as np
from skimage import exposure

def preprocess_image(image_path):
    """
    Reads an image from a file, converts it to grayscale, applies a Gaussian blur, 
    and enhances contrast using CLAHE.
    
    Args:
        image_path (str): The file path to the image.
        
    Returns:
        numpy.ndarray: The preprocessed grayscale image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur kernel 5x5 for denoising
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    return enhanced

def detect_porosity(image):
    """
    Detects circular voids (porosity) using HoughCircles.
    
    Args:
        image (numpy.ndarray): Preprocessed grayscale image.
        
    Returns:
        dict: Contains 'count' (int), 'area_percentage' (float), 
              'pore_sizes' (list of radii in pixels), and 'annotated_image' (numpy array, red circles).
    """
    annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # HoughCircles parameters (tuned for generic circular void detection)
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=2, maxRadius=30)
    
    count = 0
    pore_sizes = []
    total_pore_area = 0.0
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        count = len(circles)
        for (x, y, r) in circles:
            pore_sizes.append(float(r))
            total_pore_area += np.pi * (r ** 2)
            # Red circles (BGR format)
            cv2.circle(annotated_image, (x, y), r, (0, 0, 255), 2)
            
    # Calculate area percentage relative to the whole image
    image_area = image.shape[0] * image.shape[1]
    area_percentage = (total_pore_area / image_area) * 100 if image_area > 0 else 0.0
    
    return {
        "count": count,
        "area_percentage": area_percentage,
        "pore_sizes": pore_sizes,
        "annotated_image": annotated_image
    }

def detect_undercut(image):
    """
    Detects edge grooves (undercut) using Canny edge detection + contour analysis.
    
    Args:
        image (numpy.ndarray): Preprocessed grayscale image.
        
    Returns:
        dict: Contains 'detected' (bool), 'depth_estimate' (float 0.0-1.0), 
              and 'annotated_image' (numpy array, orange highlight).
    """
    annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Canny edge detection
    edges = cv2.Canny(image, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected = False
    depth_estimate = 0.0
    
    # Analyze contours for edge grooves
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / max(h, 1)
            
            # Simple heuristic: undercuts appear as relatively long and shallow grooves at edges
            if aspect_ratio > 2.0:
                detected = True
                # Normalized depth estimate based on image height
                current_depth = h / float(image.shape[0])
                depth_estimate = max(depth_estimate, current_depth)
                # Orange highlight (BGR format)
                cv2.drawContours(annotated_image, [cnt], -1, (0, 165, 255), 2)
                
    return {
        "detected": detected,
        "depth_estimate": depth_estimate,
        "annotated_image": annotated_image
    }

def detect_cracks(image):
    """
    Detects linear discontinuities (cracks) using morphological operations + edge detection.
    
    Args:
        image (numpy.ndarray): Preprocessed grayscale image.
        
    Returns:
        dict: Contains 'detected' (bool), 'crack_count' (int), 
              and 'annotated_image' (numpy array, yellow highlight).
    """
    annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Morphological Black Hat to highlight dark structures on bright backgrounds
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    
    # Thresholding + Edge detection on the result
    _, thresh = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crack_count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = max(w, h) / float(min(w, h) + 1e-5)
            # Cracks are highly linear (high aspect ratio)
            if aspect_ratio > 3.0:
                crack_count += 1
                # Yellow highlight (BGR format)
                cv2.drawContours(annotated_image, [cnt], -1, (0, 255, 255), 2)
                
    return {
        "detected": crack_count > 0,
        "crack_count": crack_count,
        "annotated_image": annotated_image
    }

def detect_spatter(image):
    """
    Detects bright scattered spots (spatter) outside the weld bead using threshold + blob detection.
    
    Args:
        image (numpy.ndarray): Preprocessed grayscale image.
        
    Returns:
        dict: Contains 'count' (int) and 'annotated_image' (numpy array, blue markers).
    """
    annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Thresholding for bright spots
    _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5 < area < 100:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            # Spatter is relatively small and scattered
            if radius < 15:
                count += 1
                # Blue highlight (BGR format)
                cv2.circle(annotated_image, (int(x), int(y)), int(radius), (255, 0, 0), 2)
                
    return {
        "count": count,
        "annotated_image": annotated_image
    }

def compute_image_quality_metrics(image_path):
    """
    Computes Grey Scale Value (GV), Signal to Noise Ratio (SNR),
    and Image Quality Indicator (IQI) for a weld image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gv = float(np.mean(gray))
    
    std_dev = float(np.std(gray))
    if std_dev == 0:
        snr = 0.0
    else:
        snr = gv / std_dev
        
    if snr + 1.0 == 0:
        iqi = 0.0
    else:
        iqi = (gv / 255.0) * (snr / (snr + 1.0)) * 100.0
        
    return {
        "gv": gv,
        "snr": snr,
        "iqi": iqi
    }
