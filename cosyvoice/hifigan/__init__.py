"""
HifiGAN module for CosyVoice
"""

from .generator import HiFTGenerator
from .hifigan import HiFiGan
from .discriminator import MultipleDiscriminator, MultiResSpecDiscriminator
from .f0_predictor import ConvRNNF0Predictor

__all__ = ["HiFTGenerator", "HiFiGan", "MultipleDiscriminator", "MultiResSpecDiscriminator", "ConvRNNF0Predictor"]