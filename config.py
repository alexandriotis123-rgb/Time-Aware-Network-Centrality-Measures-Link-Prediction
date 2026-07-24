"""
Project configuration file.

All global parameters used by the project are stored here.
"""

# =====================================================
# DATA PATHS
# =====================================================

DATA_FOLDER = "data"

DATASET_FILE = "data/sx-stackoverflow.txt"

OUTPUT_FOLDER = "outputs"

FIGURES_FOLDER = "outputs/figures"

RESULTS_FOLDER = "outputs/results"

MODELS_FOLDER = "outputs/models"

# =====================================================
# TEMPORAL NETWORK PARAMETERS
# =====================================================

NUMBER_OF_PERIODS = 10

GRAPH_IS_DIRECTED = False

# =====================================================
# RANDOMNESS
# =====================================================

RANDOM_SEED = 42

# =====================================================
# VISUALIZATION
# =====================================================

SAVE_FIGURES = True

FIGURE_DPI = 300

# =====================================================
# TRAINING
# =====================================================

DEFAULT_THRESHOLD = 0.5

# =====================================================
# NUMERICAL SETTINGS
# =====================================================

EPSILON = 1e-10

# =====================================================
# DEVELOPMENT
# =====================================================

DEBUG_MODE = True

DEBUG_MAX_ROWS = 500000
DEBUG_MAX_CANDIDATE_EDGES = 200000

#DEBUG_MODE = False
#DEBUG_MAX_ROWS = None
#DEBUG_MAX_CANDIDATE_EDGES = None

# ==========================================================
# Execution Controller
# ==========================================================

RUN = {
    # Part 1
    "network_evolution": False,
    # Centrality Measures
    "degree": False,
    "closeness": False,
    "betweenness": False,
    "eigenvector": False,
    "katz": False,
}

RUN_CANDIDATE_EDGES = False
RUN_FEATURE_VECTORS = True
RUN_DATASET = True

RUN_TRAINING_EXPERIMENT = True

# Training
MAX_UNIQUE_SCORES = 50
# Holdout test ratio when creating train/validation/test splits
TEST_HOLDOUT_RATIO = 0.2
# Max number of non-overlapping intervals to accept per similarity measure
MAX_INTERVALS = 3