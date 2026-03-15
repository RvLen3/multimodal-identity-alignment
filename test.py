import torch

# 1. 加载 .pt 文件
# 建议加上 map_location='cpu'，防止原文件是在 GPU 上训练保存的，而你当前环境没有 GPU 导致报错
file_path = 'user_vectors.pt'
data = torch.load(file_path, map_location='cpu')

# 2. 查看文件里存的到底是什么类型的数据
print(f"数据类型: {type(data)}")

# 3. 根据类型深入查看内容

# 情况 A: 字典 (最常见，通常是模型的 state_dict 或包含了 optimizer 状态的 checkpoint)
if isinstance(data, dict):
    print(f"字典包含的 Key 数量: {len(data.keys())}")
    print("前 10 个 Key 及其对应值的形状:")
    for i, (key, value) in enumerate(data.items()):
        if i >= 10: 
            break
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: Tensor shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  - {key}: {type(value)}")

# 情况 B: 单个张量 (Tensor)
elif isinstance(data, torch.Tensor):
    print(f"张量形状: {data.shape}")
    print(f"张量数据类型: {data.dtype}")
    print(data) # 如果张量不大，可以直接打印看看

# 情况 C: 完整的模型实例
else:
    print(data) # 会打印出模型的网络结构