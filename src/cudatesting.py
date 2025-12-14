import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("cudnn enabled:", torch.backends.cudnn.enabled)

if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("current device index:", torch.cuda.current_device())
    print("current device name:", torch.cuda.get_device_name())