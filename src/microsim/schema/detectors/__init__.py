from ._camera import Camera, CameraCCD, CameraCMOS, CameraEMCCD
from ._simulate import simulate_camera

Detector = CameraEMCCD | CameraCMOS | CameraCCD

__all__ = [
    "Detector",
    "Camera",
    "CameraCCD",
    "CameraCMOS",
    "CameraEMCCD",
    "simulate_camera",
]
