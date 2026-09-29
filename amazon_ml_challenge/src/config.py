from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Data directories
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

# Output
OUTPUT_DIR = PROJECT_ROOT / "output"

# Training files
TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

# Test files
TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"


# Processing
TSV_SEPARATOR = "\t"

# We will keep this configurable because SageMaker memory is limited.
DEFAULT_CHUNK_SIZE = 10_000