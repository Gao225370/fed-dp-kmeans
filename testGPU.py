import torch
print("CUDA可用:", torch.cuda.is_available())  # 应输出True
print("GPU数量:", torch.cuda.device_count())  # 应≥1
print("当前GPU:", torch.cuda.current_device())  # 输出GPU索引（通常为0）
print("GPU名称:", torch.cuda.get_device_name(0))  # 输出GPU型号