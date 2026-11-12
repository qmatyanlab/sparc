import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("PROJECT_ROOT", ROOT)
warnings.filterwarnings("ignore", category=UserWarning)
