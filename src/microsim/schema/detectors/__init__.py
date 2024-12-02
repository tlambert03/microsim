from ._camera import ICX285, Camera, CameraCCD, CameraCMOS, CameraEMCCD
from ._simulate import simulate_camera

Detector = CameraEMCCD | CameraCMOS | CameraCCD

__all__ = [
    "ICX285",
    "Camera",
    "CameraCCD",
    "CameraCMOS",
    "CameraEMCCD",
    "Detector",
    "simulate_camera",
]
