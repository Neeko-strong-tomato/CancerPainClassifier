import loader

def identity(scan): return scan

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