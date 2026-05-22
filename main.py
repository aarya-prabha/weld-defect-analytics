import sys
import os
import cv2
import numpy as np

from analyzer import compare_images, classify_severity, generate_summary_text
from report_generator import generate_pdf_report
import config

# ANSI escape codes for coloring terminal output
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"

def list_available_images(directory):
    if not os.path.exists(directory):
        return f"Directory '{directory}' does not exist."
    try:
        files = os.listdir(directory)
        if not files:
            return f"No images found in '{directory}'."
        return "Available images in folder:\n  - " + "\n  - ".join(files)
    except Exception as e:
        return f"Could not read directory '{directory}': {e}"

def create_composite_image(base_img_path, results):
    """
    Overlays all detector annotations on top of the original defective image.
    """
    base_img = cv2.imread(base_img_path)
    if base_img is None:
        return None
        
    # Ensure color image format
    if len(base_img.shape) == 2 or base_img.shape[2] == 1:
        composite = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        composite = base_img.copy()
        
    for key in ['porosity', 'undercut', 'crack', 'spatter']:
        ann_img = results[key].get('annotated_image')
        if ann_img is not None:
            # We identify drawn pixels by checking where color is not grayscale (R=G=B)
            # Since annotated_images are drawn on grayscale background.
            mask = (ann_img[:, :, 0] != ann_img[:, :, 1]) | (ann_img[:, :, 1] != ann_img[:, :, 2])
            composite[mask] = ann_img[mask]
            
    return composite

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <standard_image_path> <defective_image_path>")
        sys.exit(1)
        
    std_path = sys.argv[1]
    def_path = sys.argv[2]
    
    # Validation
    error = False
    if not os.path.isfile(std_path):
        print(f"Error: Standard image not found at '{std_path}'")
        print(list_available_images(os.path.dirname(std_path) or '.'))
        print("")
        error = True
        
    if not os.path.isfile(def_path):
        print(f"Error: Defective image not found at '{def_path}'")
        print(list_available_images(os.path.dirname(def_path) or '.'))
        print("")
        error = True
        
    if error:
        sys.exit(1)
        
    print("Running defect analysis...")
    
    # Process
    results = compare_images(std_path, def_path)
    severity = classify_severity(results, config)
    
    # 1. Print Summary
    summary = generate_summary_text(results, severity)
    print("\n" + summary + "\n")
    
    # 2. Composite Image
    os.makedirs("output", exist_ok=True)
    composite_img = create_composite_image(def_path, results)
    composite_path = "output/annotated_defective.jpg"
    if composite_img is not None:
        cv2.imwrite(composite_path, composite_img)
        print(f"Saved annotated composite image to: {composite_path}")
        
    # 3. PDF Report
    report_path = "output/report.pdf"
    # Pass the composite image to the PDF report to be shown as the "Annotated" side
    generate_pdf_report(results, severity, std_path, composite_path, report_path)
    print(f"Saved PDF report to: {report_path}")
    
    # 4. Colorful Final Verdict
    overall = severity['overall']
    color = COLOR_GREEN
    if overall == config.SEVERITY_MARGINAL:
        color = COLOR_YELLOW
    elif overall == config.SEVERITY_REJECT:
        color = COLOR_RED
        
    print(f"\nFinal Verdict: {color}{overall}{COLOR_RESET}\n")

if __name__ == "__main__":
    main()
