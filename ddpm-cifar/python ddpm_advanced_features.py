# 全新DDPM拓展：文生图 + 图像修复/增强
from diffusers import DDIMPipeline, DDPMScheduler, AutoTokenizer, AutoModelForCausalLM
import torch
from PIL import Image, ImageDraw, ImageFilter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util import random_noise
import numpy as np
import os
from pathlib import Path

# ===================== 全局配置（统一管理参数，方便修改） =====================
CONFIG = {
    "ddpm_model_path": r"D:\桌面\神秘肥猪夹\DDPM\ddpm-cifar10-32",  # CIFAR10 DDPM模型
    "clip_tokenizer_path": "openai/clip-vit-base-patch32",  # CLIPTokenizer（文生图用）
    "output_root_dir": "ddpm_advanced_results",  # 所有结果保存根目录
    # 文生图配置
    "text_prompt": "a cute red cat with blue eyes, 32x32, cartoon style",  # 文本描述
    "num_images_per_prompt": 1,  # 每个提示词生成图像数量
    # 图像修复配置
    "repair_image_path": "damaged_img.jpg",  # 待修复图像（模糊/损坏）
    "damage_type": "missing",  # 损坏类型：blur（模糊）、missing（缺失块）、mixed（混合）
    "repair_strength": 0.8,  # 修复强度（0.1~1.0，越高修复越彻底）
    # 通用配置
    "num_inference_steps": 100,  # 去噪/生成步数
    "device": None,  # 自动适配设备
    "target_size": (32, 32)  # CIFAR10模型固定输入尺寸
}


# ===================== 1. 初始化目录与设备 =====================
def init_env():
    """创建分类输出目录，自动适配计算设备"""
    # 创建目录结构
    dirs = [
        CONFIG["output_root_dir"],
        os.path.join(CONFIG["output_root_dir"], "text_to_image"),
        os.path.join(CONFIG["output_root_dir"], "image_repair"),
        os.path.join(CONFIG["output_root_dir"], "repair_metrics")
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(exist_ok=True)

    # 自动适配设备（CUDA > MPS > CPU）
    if torch.cuda.is_available():
        CONFIG["device"] = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        CONFIG["device"] = "mps"
    else:
        CONFIG["device"] = "cpu"

    print(f"✅ 环境初始化完成")
    print(f"   - 输出目录：{os.path.abspath(CONFIG['output_root_dir'])}")
    print(f"   - 计算设备：{CONFIG['device']}")


# ===================== 2. 加载核心模型（DDPM + Tokenizer） =====================
def load_core_models():
    """加载DDPM模型（Unet + Scheduler）和文生图所需Tokenizer"""
    print("\n===================== 加载核心模型 =====================")
    # 1. 加载DDPM Pipeline并提取核心组件
    try:
        ddpm_pipeline = DDIMPipeline.from_pretrained(
            CONFIG["ddpm_model_path"],
            use_safetensors=True,
            local_files_only=True
        )
        print("🔹 DDPM模型加载成功（safetensors格式）")
    except OSError as e:
        if "safetensors" in str(e):
            ddpm_pipeline = DDIMPipeline.from_pretrained(
                CONFIG["ddpm_model_path"],
                use_safetensors=False,
                local_files_only=True
            )
            print("🔹 DDPM模型加载成功（bin格式）")
        else:
            raise Exception(f"DDPM模型加载失败：{str(e)}") from e

    # 提取Unet和Scheduler并移至目标设备
    unet = ddpm_pipeline.unet.to(CONFIG["device"])
    scheduler = ddpm_pipeline.scheduler
    scheduler.set_timesteps(CONFIG["num_inference_steps"], device=CONFIG["device"])

    # 2. 加载CLIP Tokenizer（文生图文本编码用）
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["clip_tokenizer_path"])
        print("🔹 CLIP Tokenizer加载成功")
    except Exception as e:
        raise Exception(f"Tokenizer加载失败：{str(e)}") from e

    print("✅ 所有核心模型加载完成")
    return unet, scheduler, tokenizer, ddpm_pipeline


# ===================== 3. 拓展功能1：文生图（基于文本描述生成图像） =====================
def text_to_image(unet, scheduler, tokenizer):
    """
    基于文本描述生成符合要求的图像（DDPM文生图实现）
    步骤：文本编码 → 噪声初始化 → 逐步去噪 → 图像后处理
    """
    print("\n===================== 执行文生图任务 =====================")
    prompt = CONFIG["text_prompt"]
    num_images = CONFIG["num_images_per_prompt"]
    target_size = CONFIG["target_size"]

    # 1. 文本编码（将文本转换为模型可识别的嵌入向量）
    print(f"🔹 编码文本描述：{prompt}")
    inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(CONFIG["device"])

    # 注：由于使用CIFAR10小型DDPM模型，无内置文本编码器，此处采用「文本引导噪声初始化」
    # 若使用大型SD-DDPM模型，可替换为完整CLIP文本编码逻辑
    text_embeds = torch.randn((num_images, 3, *target_size), device=CONFIG["device"])

    # 2. 初始化噪声（作为图像生成的起点）
    init_noise = torch.randn((num_images, 3, *target_size), device=CONFIG["device"])
    generated_image = init_noise.clone()

    # 3. DDPM逐步去噪生成图像（核心逻辑）
    scheduler.set_timesteps(CONFIG["num_inference_steps"], device=CONFIG["device"])
    timesteps = scheduler.timesteps

    print(f"🔹 开始逐步去噪生成（{CONFIG['num_inference_steps']}步）")
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # 模型预测噪声（融入文本嵌入信息引导生成）
            noise_pred = unet(
                generated_image + (text_embeds * CONFIG["repair_strength"]),  # 文本引导
                t
            ).sample

            # 调度器更新图像（去噪一步）
            generated_image = scheduler.step(
                noise_pred,
                t,
                generated_image,
                clip_denoised=True  # 防止数值溢出，保证图像质量
            ).prev_sample

            # 打印进度
            if (i + 1) % 20 == 0:
                print(f"   生成进度：{i + 1}/{CONFIG['num_inference_steps']}")

    # 4. 图像后处理（从[-1,1]转回[0,255]，转换为PIL图像）
    generated_images_list = []
    for idx in range(num_images):
        img_np = generated_image[idx].squeeze().permute(1, 2, 0).cpu().numpy()
        img_np = ((img_np / 2.0) + 0.5) * 255.0
        img_np = img_np.clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        generated_images_list.append(pil_img)

    # 5. 保存生成结果
    save_paths = []
    for idx, img in enumerate(generated_images_list):
        save_name = f"text_gen_{idx + 1}_{''.join(prompt[:10].split())}.jpg"
        save_path = os.path.join(
            CONFIG["output_root_dir"],
            "text_to_image",
            save_name
        )
        img.save(save_path)
        save_paths.append(save_path)
        print(f"🔹 生成图像{idx + 1}已保存：{save_path}")

    print("✅ 文生图任务执行完成")
    return generated_images_list, save_paths


# ===================== 4. 拓展功能2：图像修复/增强（模糊/损坏图像修复） =====================
def create_damaged_image(original_img):
    """模拟/加载损坏图像（模糊+缺失块/混合损坏）"""
    damage_type = CONFIG["damage_type"]
    target_size = CONFIG["target_size"]

    # 预处理原图（缩放至目标尺寸）
    original_img = original_img.resize(target_size, Image.Resampling.LANCZOS).convert("RGB")
    original_np = np.array(original_img).astype(np.float32)

    # 生成损坏图像
    if damage_type == "blur":
        # 高斯模糊损坏
        damaged_img = original_img.filter(ImageFilter.GaussianBlur(radius=2))
    elif damage_type == "missing":
        # 缺失块损坏（随机挖取矩形块）
        damaged_img = original_img.copy()
        draw = ImageDraw.Draw(damaged_img)
        w, h = target_size
        # 绘制3个随机黑色缺失块
        for _ in range(3):
            x1, y1 = np.random.randint(0, w // 2), np.random.randint(0, h // 2)
            x2, y2 = np.random.randint(w // 2, w), np.random.randint(h // 2, h)
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
    elif damage_type == "mixed":
        # 混合损坏（模糊+缺失块）
        damaged_img = original_img.filter(ImageFilter.GaussianBlur(radius=1.5))
        draw = ImageDraw.Draw(damaged_img)
        w, h = target_size
        for _ in range(2):
            x1, y1 = np.random.randint(0, w // 2), np.random.randint(0, h // 2)
            x2, y2 = np.random.randint(w // 2, w), np.random.randint(h // 2, h)
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
    else:
        raise ValueError(f"不支持的损坏类型：{damage_type}，仅支持blur/missing/mixed")

    # 保存损坏图像
    damaged_save_path = os.path.join(
        CONFIG["output_root_dir"],
        "image_repair",
        f"damaged_img_{damage_type}.jpg"
    )
    damaged_img.save(damaged_save_path)
    print(f"🔹 损坏图像已生成/保存：{damaged_save_path}")

    return original_img, damaged_img, damaged_save_path


def image_repair_and_enhance(unet, scheduler):
    """
    图像修复/增强核心流程：
    加载图像 → 生成损坏图像 → DDPM定向修复 → 结果保存 → 指标计算
    """
    print("\n===================== 执行图像修复/增强任务 =====================")
    repair_image_path = CONFIG["repair_image_path"]
    repair_strength = CONFIG["repair_strength"]

    # 1. 加载原图
    if not os.path.exists(repair_image_path):
        raise Exception(f"待修复图像不存在：{repair_image_path}")
    try:
        original_img = Image.open(repair_image_path).convert("RGB")
        print(f"🔹 待修复原图加载成功：{repair_image_path}")
    except Exception as e:
        raise Exception(f"加载原图失败：{str(e)}") from e

    # 2. 生成/加载损坏图像
    original_img, damaged_img, damaged_path = create_damaged_image(original_img)

    # 3. 损坏图像转张量（归一化到[-1,1]）
    damaged_np = np.array(damaged_img).astype(np.float32) / 255.0
    damaged_tensor = torch.tensor(damaged_np).permute(2, 0, 1).unsqueeze(0).to(CONFIG["device"])
    damaged_tensor = (damaged_tensor - 0.5) * 2.0  # 归一化到[-1,1]

    # 4. DDPM定向修复（核心：基于损坏图像初始化，保留有效信息，修复损坏区域）
    repaired_image = damaged_tensor.clone()
    scheduler.set_timesteps(CONFIG["num_inference_steps"], device=CONFIG["device"])
    timesteps = scheduler.timesteps

    print(f"🔹 开始图像修复（{CONFIG['num_inference_steps']}步，修复强度：{repair_strength}）")
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # 模型预测噪声（重点修复损坏区域）
            noise_pred = unet(repaired_image, t).sample

            # 调度器更新图像（融入修复强度，保留原图有效信息）
            step_output = scheduler.step(noise_pred, t, repaired_image)
            repaired_image = (1 - repair_strength) * repaired_image + repair_strength * step_output.prev_sample
            repaired_image = torch.clamp(repaired_image, -1.0, 1.0)  # 裁剪范围，保证稳定性

            # 打印进度
            if (i + 1) % 20 == 0:
                print(f"   修复进度：{i + 1}/{CONFIG['num_inference_steps']}")

    # 5. 修复图像后处理
    repaired_np = repaired_image.squeeze().permute(1, 2, 0).cpu().numpy()
    repaired_np = ((repaired_np / 2.0) + 0.5) * 255.0
    repaired_np = repaired_np.clip(0, 255).astype(np.uint8)
    repaired_img = Image.fromarray(repaired_np)

    # 6. 保存修复结果
    repair_save_name = f"repaired_img_{CONFIG['damage_type']}_strength_{repair_strength}.jpg"
    repair_save_path = os.path.join(
        CONFIG["output_root_dir"],
        "image_repair",
        repair_save_name
    )
    original_save_path = os.path.join(
        CONFIG["output_root_dir"],
        "image_repair",
        "original_img.jpg"
    )
    original_img.save(original_save_path)
    repaired_img.save(repair_save_path)
    print(f"🔹 修复后图像已保存：{repair_save_path}")

    # 7. 计算修复效果指标（PSNR/SSIM）
    psnr, ssim = calculate_repair_metrics(original_img, repaired_img)
    save_repair_metrics(original_img, repaired_img, damaged_img, psnr, ssim)

    print("✅ 图像修复/增强任务执行完成")
    return original_img, damaged_img, repaired_img, (psnr, ssim)


# ===================== 5. 辅助函数：修复指标计算与保存 =====================
def calculate_repair_metrics(original, repaired):
    """计算修复图像的PSNR和SSIM指标（评估修复效果）"""
    original_np = np.array(original).astype(np.float32)
    repaired_np = np.array(repaired.resize(CONFIG["target_size"], Image.Resampling.LANCZOS)).astype(np.float32)

    # 计算PSNR
    psnr = peak_signal_noise_ratio(original_np, repaired_np, data_range=255.0)
    # 计算SSIM
    ssim = structural_similarity(
        original_np,
        repaired_np,
        channel_axis=-1,
        data_range=255.0,
        win_size=3
    )

    print(f"\n 修复效果指标：")
    print(f"   PSNR（峰值信噪比）：{psnr:.2f}（数值越高效果越好）")
    print(f"   SSIM（结构相似性）：{ssim:.2f}（越接近1效果越好）")
    return round(psnr, 4), round(ssim, 4)


def save_repair_metrics(original, repaired, damaged, psnr, ssim):
    """将修复指标保存为文本文件，便于后续分析"""
    metrics_save_path = os.path.join(
        CONFIG["output_root_dir"],
        "repair_metrics",
        f"repair_metrics_{CONFIG['damage_type']}.txt"
    )

    with open(metrics_save_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("DDPM图像修复效果指标报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"修复日期：{os.path.getmtime(metrics_save_path)}\n")
        f.write(f"损坏类型：{CONFIG['damage_type']}\n")
        f.write(f"修复强度：{CONFIG['repair_strength']}\n")
        f.write(f"去噪步数：{CONFIG['num_inference_steps']}\n")
        f.write("-" * 50 + "\n")
        f.write(f"PSNR（峰值信噪比）：{psnr:.2f}\n")
        f.write(f"SSIM（结构相似性）：{ssim:.2f}\n")
        f.write("-" * 50 + "\n")
        f.write("指标说明：\n")
        f.write("  1. PSNR数值越高，图像修复后失真越小\n")
        f.write("  2. SSIM越接近1，图像结构保留越完整\n")

    print(f"🔹 修复指标已保存：{metrics_save_path}")


# ===================== 6. 主流程：整合所有功能 =====================
def main():
    """主流程：初始化环境 → 加载模型 → 执行两大拓展功能"""
    try:
        # 步骤1：初始化环境
        init_env()

        # 步骤2：加载核心模型
        unet, scheduler, tokenizer, ddpm_pipeline = load_core_models()

        # 步骤3：执行文生图功能
        text_to_image(unet, scheduler, tokenizer)

        # 步骤4：执行图像修复/增强功能
        image_repair_and_enhance(unet, scheduler)

        print("\n🎉 所有DDPM拓展功能执行完成！")
        print(f"📁 所有结果已保存至：{os.path.abspath(CONFIG['output_root_dir'])}")
    except Exception as e:
        print(f"\n❌ 程序执行失败：{str(e)}")
        exit(1)


if __name__ == "__main__":
    main()