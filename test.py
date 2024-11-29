import torch
print(torch.version.cuda)  # CUDA バージョンを表示
print(torch.cuda.is_available())  # True になるべき
print(torch.cuda.get_device_name(0))  # 使用可能な GPU の名前を表示
