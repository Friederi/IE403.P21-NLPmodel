import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
inference_path = os.path.join(project_root, 'inference')

if inference_path not in sys.path:
    sys.path.insert(0, inference_path)