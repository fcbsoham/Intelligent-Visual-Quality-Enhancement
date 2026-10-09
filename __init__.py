"""Model zoo for the Visual Restoration Suite."""

from .autoencoder import DenoisingAutoencoder
from .gan import Generator, PatchDiscriminator
from .unet import UNet

__all__ = ["DenoisingAutoencoder", "UNet", "Generator", "PatchDiscriminator"]
