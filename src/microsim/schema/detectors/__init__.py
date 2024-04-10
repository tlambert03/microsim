from ._camera import Camera, CameraCCD, CameraCMOS, CameraEMCCD

Detector = CameraEMCCD | CameraCMOS | CameraCCD

__all__ = ["Detector", "Camera", "CameraCCD", "CameraCMOS", "CameraEMCCD"]
