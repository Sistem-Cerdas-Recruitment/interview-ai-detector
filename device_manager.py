import torch


class DeviceManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
            cls._instance.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        return cls._instance.device
