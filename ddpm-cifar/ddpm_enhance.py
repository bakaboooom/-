from diffusers import DDIMPipeline, DDPMScheduler
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import os

# ===================== 1. 加载模型和调度器（底层调用，避开参数问题） =====================
print("正在加载模型...")
model_path = r"D:\桌面\神秘肥猪夹\DDPM\ddpm-cifar10-32"
# 手动加载模型和调度器（替代Pipeline的封装）
try:
    pipeline = DDIMPipeline.from_pretrained(model_path, use_safetensors=True, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = pipeline.unet.to(device)  # 取出Unet模型
    scheduler = pipeline.scheduler    # 取出调度器
    print(f"✅ 模型加载成功，使用设备：{device}")
except OSError as e:
    pipeline = DDIMPipeline.from_pretrained(model_path, use_safetensors=False, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = pipeline.unet.to(device)
    scheduler = pipeline.scheduler
    print(f"✅ 模型加载成功（bin格式），使用设备：{device}")

# ===================== 2. 加载并预处理原图 =====================
image_path = "test_img5.jpg"
if not os.path.exists(image_path):
    print(f"错误：找不到图像文件 {image_path}，当前目录：{os.getcwd()}")
    exit()

try:
    original_image = Image.open(image_path).resize((32, 32)).convert("RGB")
    print("✅ 原图加载成功")
    # 原图转张量（归一化到[-1,1]）
    original_np = np.array(original_image).astype(np.float32) / 255.0
    original_tensor = torch.tensor(original_np).permute(2, 0, 1).unsqueeze(0)  # (1,3,32,32)
    original_tensor = (original_tensor - 0.5) * 2.0
    original_tensor = original_tensor.to(device)
except Exception as e:
    print(f"错误：加载原图失败 → {e}")
    exit()

# ===================== 3. 给原图加可控噪声（模拟低质图） =====================
noise_strength = 0.1  # 噪声强度，越小效果越好
noise = torch.randn_like(original_tensor).to(device)
noisy_tensor = original_tensor + noise_strength * noise  # 加噪后的原图

# 保存加噪图像
noisy_np = noisy_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
noisy_np = ((noisy_np / 2.0) + 0.5) * 255.0
noisy_np = noisy_np.clip(0, 255).astype(np.uint8)
Image.fromarray(noisy_np).save("noisy_img.jpg")
print("✅ 带噪原图已保存为noisy_img.jpg")

# ===================== 4. 手动实现DDPM去噪（核心：绕开Pipeline参数问题） =====================
print("正在增强图像（手动DDPM去噪）...")
num_inference_steps = 100  # 去噪步数
scheduler.set_timesteps(num_inference_steps, device=device)  # 设置时间步
timesteps = scheduler.timesteps

# 初始化去噪图像为加噪后的张量
denoised_image = noisy_tensor.clone()

# 逐步去噪（DDPM核心逻辑）
with torch.no_grad():
    for i, t in enumerate(timesteps):
        # 模型预测噪声
        noise_pred = model(denoised_image, t).sample
        # 调度器更新图像（去噪一步）
        denoised_image = scheduler.step(noise_pred, t, denoised_image).prev_sample
        # 打印进度
        if (i+1) % 20 == 0:
            print(f"去噪进度：{i+1}/{num_inference_steps}")

# ===================== 5. 转换并保存增强图像 =====================
# 张量转回PIL图像
enhanced_np = denoised_image.squeeze().permute(1, 2, 0).cpu().numpy()
enhanced_np = ((enhanced_np / 2.0) + 0.5) * 255.0  # 从[-1,1]转回[0,255]
enhanced_np = enhanced_np.clip(0, 255).astype(np.uint8)
enhanced_image = Image.fromarray(enhanced_np)
enhanced_image.save("enhanced_img.jpg")
print("✅ 增强图像已保存为enhanced_img.jpg")

# ===================== 6. 计算评价指标 =====================
def calculate_metrics(original, enhanced):
    original_np = np.array(original)
    enhanced_np = np.array(enhanced.resize((32, 32)).convert("RGB"))
    psnr = peak_signal_noise_ratio(original_np, enhanced_np)
    ssim = structural_similarity(original_np, enhanced_np, channel_axis=-1)
    return psnr, ssim

psnr, ssim = calculate_metrics(original_image, enhanced_image)
print(f"\n 增强效果指标：")
print(f"PSNR（峰值信噪比）：{psnr:.2f}（数值越高效果越好）")
print(f"SSIM（结构相似性）：{ssim:.2f}（越接近1效果越好）")
print("\n🎉 图像增强流程全部完成！")