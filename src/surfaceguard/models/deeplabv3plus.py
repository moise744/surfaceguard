"""
Compatibility shim: surfaceguard.models.deeplabv3plus → surfaceguard.models.deeplabv3

Allows notebooks to use:
    from surfaceguard.models.deeplabv3plus import DeepLabV3Plus
"""
from surfaceguard.models.deeplabv3 import DeepLabV3Plus  # noqa: F401

__all__ = ["DeepLabV3Plus"]
