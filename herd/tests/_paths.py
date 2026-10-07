import os
import sys

HERD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERD not in sys.path:
    sys.path.insert(0, HERD)
