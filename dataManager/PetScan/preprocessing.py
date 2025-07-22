import loader
import logger

def identity(scan): return 0

PREPOCESSINGS = {
    'identity': identity,
}


def preprocess_a_scan(labelized_scan, preprocessing_method=None):
    """
    Args:
        labelized_scan: dict {'scan': np.ndarray, 'label': int}
        preprocessing_method: str (key from PREPROCESSING)

    Returns:
        Nothing, the scan as been preprocessed and modified with Bohr effect
    """

    scan = labelized_scan['data']
    
    if preprocessing_method is not None:
        preprocess = PREPOCESSINGS[preprocessing_method]
        preprocess(scan)


def preprocess_all_scans(labelized_scans, preprocessing_method=None):
    """
    Args:
        labelized_scans: list of dict {'scan': np.ndarray, 'label': int}
        preprocessing_method: str (key from PREPROCESSING)

    Returns:
        Nothing, the scan as been preprocessed and modified with Bohr effect
    """

    scan_amount = len(labelized_scans)
    scan_index = 0

    ProgressBar = logger.ProgressReporter('Preprocessing')

    for scan in labelized_scans :
        scan_index += 1
        ProgressBar.update((scan_index/scan_amount)*100)
        preprocess_a_scan(scan, preprocessing_method)
        
    print("\nTerminé.")