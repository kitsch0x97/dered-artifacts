# hashlib: export LD_LIBRARY_PATH=/usr/local/ssl/lib:$LD_LIBRARY_PATH

WORK_PATH = "."
DATA_PATH = "dataset"
GT_PATH = "groundtruth"

FEATURE_EXTRACT_PATH = WORK_PATH + "/code/0_binary_preprocess/ida_process_binary.py"

PACKAGE_PATH = "<python2.7_site_packages_path>"  # e.g., /path/to/python2.7/site-packages/

IDA_PATH = "<ida_pro_path>/idal64"  # e.g., /path/to/IDA_Pro/idal64