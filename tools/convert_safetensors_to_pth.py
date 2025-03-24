import torch
from safetensors.torch import load_file

def convert_safetensors_to_pth(safetensors_path, pth_path, device="cpu"):
    """
    将 safetensors 格式转换为 PyTorch 的 .pth 格式
    
    参数:
        safetensors_path : str - 输入的 .safetensors 文件路径
        pth_path : str - 输出的 .pth 文件路径
        device : str - 加载设备 [cpu/cuda]
    """
    # 加载 safetensors 权重
    state_dict = load_file(safetensors_path, device=device)
    
    # 保存为 PyTorch 格式
    torch.save(state_dict, pth_path)
    print(f"Successfully converted to {pth_path}")

if __name__ == "__main__":
    # 使用示例
    input_path = r"E:\project_global_populus\MAE_test_250324\3-weights\model.safetensors"  # 输入文件路径
    output_path = r"E:\project_global_populus\MAE_test_250324\3-weights\converted_model.pth"    # 输出文件路径
    
    # 执行转换 (默认使用CPU)
    convert_safetensors_to_pth(input_path, output_path)
    
    # 验证转换结果
    loaded = torch.load(output_path)
    # print("Converted keys:", loaded.keys())  # 查看转换后的权重键名