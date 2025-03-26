def read_config(file_path="../common.txt"):
    config = {}
    with open(file_path, 'r') as f:
        for line in f:
            # Skip empty lines or comments
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                # Remove quotes and whitespace from value
                value = value.strip().strip("'").strip('"')
                config[key] = value
    return config

# Read configuration from the file (change the filename if needed)
# filepath = "
config = read_config()

COMPANY = config.get("COMPANY")
TRAIN_START = config.get("TRAIN_START")
TRAIN_END = config.get("TRAIN_END")
FEATURE_COLUMNS = config.get("FEATURE_COLUMNS")
PREDICTION_DAYS = config.get("PREDICTIONS_DAYS")  # Number of days to look back for each prediction input
STEPS_TO_PREDICT = config.get("STEPS_TO_PREDICT")  # How many future steps to predict; using 'future' as target
TARGET_COLUMN = 'future'
print("COMPANY:", COMPANY)
print("TRAIN_START:", TRAIN_START)
print("TRAIN_END:", TRAIN_END)
