"""
Configuration parameters and acceptance criteria for welding defect image analytics.
Based on ISO 5817 Level C acceptance criteria.
"""

# Defect severity levels
SEVERITY_ACCEPT = "ACCEPT"
SEVERITY_MARGINAL = "MARGINAL"
SEVERITY_REJECT = "REJECT"

# ISO 5817 Level C Acceptance Criteria
class ISO5817LevelC:
    # Porosity
    POROSITY_MAX_AREA_PERCENTAGE = 1.5
    POROSITY_MAX_SINGLE_PORE_SIZE_MM = 4.0
    
    # Undercut
    UNDERCUT_MAX_DEPTH_RATIO = 0.1  # fraction of plate thickness
    
    # Spatter
    SPATTER_MAX_COUNT_PER_100MM = 5
    
    # Crack
    CRACK_TOLERANCE = 0.0  # zero tolerance - not permitted

def get_defect_severity(defect_value, limit):
    """
    Classifies defect severity against acceptance criteria.
    
    Severity thresholds:
    - if defect value == 0 -> ACCEPT
    - if within 80% of limit -> MARGINAL (interpreted as between 80% and 100% of limit)
    - if exceeds limit -> REJECT
    """
    if limit == 0:
        return SEVERITY_ACCEPT if defect_value == 0 else SEVERITY_REJECT
        
    if defect_value == 0:
        return SEVERITY_ACCEPT
    elif defect_value > limit:
        return SEVERITY_REJECT
    elif defect_value >= (0.8 * limit):
        return SEVERITY_MARGINAL
    else:
        # If defect is > 0 but less than 80% of the limit, it's acceptable
        return SEVERITY_ACCEPT
