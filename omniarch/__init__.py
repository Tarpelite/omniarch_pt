"""OmniArch (智算天璇) mini."""
from .inference import OmniArchPredictor
from .model import MAX_CHANNELS, PRESETS, OmniArch, OmniArchConfig

__all__ = ["OmniArch", "OmniArchConfig", "OmniArchPredictor", "MAX_CHANNELS", "PRESETS"]
__version__ = "0.1.0"
