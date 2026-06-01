import cv2
from skimage.metrics import structural_similarity as ssim
import config
from detector import (
    preprocess_image,
    detect_porosity,
    detect_undercut,
    detect_cracks,
    detect_spatter,
    compute_image_quality_metrics
)

def compare_images(standard_path, defective_path):
    """
    Loads standard and defective images, computes SSIM, and runs detectors on the defective image.
    
    Args:
        standard_path (str): File path to standard weld image.
        defective_path (str): File path to defective weld image.
        
    Returns:
        dict: Complete results including detector outputs and SSIM score.
    """
    std_img = preprocess_image(standard_path)
    def_img = preprocess_image(defective_path)
    
    # Resize standard image to match defective image dimensions if needed for SSIM
    if std_img.shape != def_img.shape:
        std_img = cv2.resize(std_img, (def_img.shape[1], def_img.shape[0]))
        
    # Calculate SSIM using skimage.metrics
    ssim_score, _ = ssim(std_img, def_img, full=True)
    
    # Run all defect detectors on the defective image
    porosity_results = detect_porosity(def_img)
    undercut_results = detect_undercut(def_img)
    crack_results = detect_cracks(def_img)
    spatter_results = detect_spatter(def_img)
    
    quality_metrics = compute_image_quality_metrics(defective_path)
    
    return {
        "ssim_score": ssim_score,
        "porosity": porosity_results,
        "undercut": undercut_results,
        "crack": crack_results,
        "spatter": spatter_results,
        "quality_metrics": quality_metrics
    }

def classify_severity(results, cfg):
    """
    Assigns ACCEPT / MARGINAL / REJECT to each defect type and computes overall verdict.
    
    Args:
        results (dict): Dictionary of results from compare_images.
        cfg: Configuration module (e.g., config.py) with severity thresholds.
        
    Returns:
        dict: A dictionary containing severity classifications and the overall verdict.
    """
    # 1. Porosity Severity
    area_percent = results["porosity"]["area_percentage"]
    porosity_severity = cfg.get_defect_severity(
        area_percent, 
        cfg.ISO5817LevelC.POROSITY_MAX_AREA_PERCENTAGE
    )
    
    # Override based on single max pore size
    max_pore = max(results["porosity"]["pore_sizes"]) if results["porosity"]["pore_sizes"] else 0
    if max_pore > cfg.ISO5817LevelC.POROSITY_MAX_SINGLE_PORE_SIZE_MM:
        porosity_severity = cfg.SEVERITY_REJECT
        
    # 2. Undercut Severity
    undercut_depth = results["undercut"]["depth_estimate"]
    undercut_severity = cfg.get_defect_severity(
        undercut_depth,
        cfg.ISO5817LevelC.UNDERCUT_MAX_DEPTH_RATIO
    )
    
    # 3. Spatter Severity
    spatter_count = results["spatter"]["count"]
    spatter_severity = cfg.get_defect_severity(
        spatter_count,
        cfg.ISO5817LevelC.SPATTER_MAX_COUNT_PER_100MM
    )
    
    # 4. Crack Severity
    crack_count = results["crack"]["crack_count"]
    crack_severity = cfg.get_defect_severity(
        crack_count,
        cfg.ISO5817LevelC.CRACK_TOLERANCE
    )
    
    # Image Quality Severity
    gv = results["quality_metrics"]["gv"]
    gv_severity = cfg.get_quality_severity_range(gv, cfg.ImageQuality.GV_MIN, cfg.ImageQuality.GV_MAX)
    
    snr = results["quality_metrics"]["snr"]
    snr_severity = cfg.get_quality_severity_min(snr, cfg.ImageQuality.SNR_MIN)
    
    iqi = results["quality_metrics"]["iqi"]
    iqi_severity = cfg.get_quality_severity_min(iqi, cfg.ImageQuality.IQI_MIN)
    
    severities = {
        "porosity": porosity_severity,
        "undercut": undercut_severity,
        "spatter": spatter_severity,
        "crack": crack_severity,
        "gv": gv_severity,
        "snr": snr_severity,
        "iqi": iqi_severity
    }
    
    # 5. Overall Verdict
    if cfg.SEVERITY_REJECT in severities.values():
        overall_verdict = cfg.SEVERITY_REJECT
    elif cfg.SEVERITY_MARGINAL in severities.values():
        overall_verdict = cfg.SEVERITY_MARGINAL
    else:
        overall_verdict = cfg.SEVERITY_ACCEPT
        
    return {
        "individual": severities,
        "overall": overall_verdict
    }

def generate_summary_text(results, severity):
    """
    Generates a formatted multiline string summarizing findings and severity verdicts.
    
    Args:
        results (dict): The detection results from compare_images.
        severity (dict): The severity classification results from classify_severity.
        
    Returns:
        str: Formatted summary report.
    """
    report = []
    report.append("=========================================")
    report.append("       WELDING DEFECT ANALYSIS REPORT    ")
    report.append("=========================================")
    report.append(f"Structural Similarity Index (SSIM): {results['ssim_score']:.4f}\n")
    
    report.append("--- IMAGE QUALITY METRICS ---")
    
    qm = results["quality_metrics"]
    gv_sev = severity["individual"]["gv"]
    snr_sev = severity["individual"]["snr"]
    iqi_sev = severity["individual"]["iqi"]
    
    report.append(f"Grey Scale Value (GV): {qm['gv']:.2f} -> [{gv_sev}]")
    report.append(f"Signal to Noise Ratio (SNR): {qm['snr']:.2f} -> [{snr_sev}]")
    report.append(f"Image Quality Indicator (IQI): {qm['iqi']:.2f} -> [{iqi_sev}]\n")
    
    report.append("--- DEFECT FINDINGS ---")
    
    # Porosity
    p_res = results["porosity"]
    p_sev = severity["individual"]["porosity"]
    report.append(f"Porosity: {p_res['count']} pores detected, Area: {p_res['area_percentage']:.2f}% -> [{p_sev}]")
    
    # Undercut
    u_res = results["undercut"]
    u_sev = severity["individual"]["undercut"]
    report.append(f"Undercut: Detected={u_res['detected']}, Depth Ratio={u_res['depth_estimate']:.3f} -> [{u_sev}]")
    
    # Cracks
    c_res = results["crack"]
    c_sev = severity["individual"]["crack"]
    report.append(f"Cracks:   {c_res['crack_count']} detected -> [{c_sev}]")
    
    # Spatter
    s_res = results["spatter"]
    s_sev = severity["individual"]["spatter"]
    report.append(f"Spatter:  {s_res['count']} spots detected -> [{s_sev}]\n")
    
    report.append("--- FINAL VERDICT ---")
    report.append(f"Overall Recommendation: {severity['overall']}")
    report.append("=========================================")
    
    return "\n".join(report)
