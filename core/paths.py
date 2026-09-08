# core/paths.py
import os
from pathlib import Path


class Paths:
    PROJECT_ROOT = Path(__file__).parent.parent
    # SMPL body model used to derive joint labels for PoseNet training.
    # Download basicmodel_m.pkl from https://smpl.is.tue.mpg.de (license required)
    # and either place it under assets/smpl/ or set SMPL_MODEL_PATH.
    SMPL_FILE = Path(os.environ.get("SMPL_MODEL_PATH", PROJECT_ROOT / "assets" / "smpl" / "basicmodel_m.pkl"))
