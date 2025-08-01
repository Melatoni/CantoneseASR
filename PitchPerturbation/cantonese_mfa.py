import os
import json
import subprocess
from tqdm import tqdm
import logging

# 配置日志（记录转换过程和错误）
logging.basicConfig(
    filename='audio_conversion.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

def main():
    # 核心路径配置
    AUDIO_ROOT_DIR = "/Users/kennychan/Downloads/MCE_Dataset/Audio/"  # 音频根目录（包含所有子目录）
    WAV_DIR = "/Users/kennychan/Downloads/MCE_Dataset/wavs/"          # 转换后音频保存目录
    JSONL_PATH = "/Users/kennychan/Downloads/MCE_Dataset/MCE_ASR2_results_new_2.jsonl"  # 标注文件
    
    # 创建输出目录
    os.makedirs(WAV_DIR, exist_ok=True)
    logging.info(f"输出目录已准备：{WAV_DIR}")

    # 步骤1：扫描所有子目录，建立文件名到实际路径的映射
    audio_path_map = {}
    missing_dirs = []
    
    # 遍历根目录下的所有子目录（如100_MCE、14_MCE等）
    for subdir_name in os.listdir(AUDIO_ROOT_DIR):
        subdir_path = os.path.join(AUDIO_ROOT_DIR, subdir_name)
        
        # 跳过非目录文件
        if not os.path.isdir(subdir_path):
            continue
        
        # 检查子目录是否为空
        try:
            subdir_files = os.listdir(subdir_path)
        except PermissionError:
            logging.warning(f"无权限访问目录：{subdir_path}")
            missing_dirs.append(subdir_path)
            continue
        
        if not subdir_files:
            logging.info(f"子目录为空：{subdir_path}")
            continue
        
        # 记录子目录中的音频文件
        for filename in subdir_files:
            if filename.lower().endswith(".wav"):  # 只处理WAV文件
                full_path = os.path.join(subdir_path, filename)
                audio_path_map[filename] = full_path
    
    logging.info(f"扫描完成 - 共发现 {len(audio_path_map)} 个音频文件（分布在 {len(os.listdir(AUDIO_ROOT_DIR)) - len(missing_dirs)} 个子目录）")
    print(f"扫描完成 - 共发现 {len(audio_path_map)} 个音频文件")

    # 步骤2：从JSONL文件中提取需要转换的音频文件名
    try:
        with open(JSONL_PATH, "r", encoding="utf-8") as f:
            samples = [json.loads(line.strip()) for line in f if line.strip()]
        logging.info(f"成功加载JSONL文件，共 {len(samples)} 条记录")
    except FileNotFoundError:
        logging.error(f"JSONL文件不存在：{JSONL_PATH}")
        print(f"错误：未找到JSONL文件 {JSONL_PATH}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"JSONL格式错误：{str(e)}")
        print(f"错误：JSONL文件格式不正确 - {str(e)}")
        return

    # 提取并去重需要转换的文件名
    required_files = set()
    for sample in samples:
        if "AudioPath" not in sample:
            logging.warning("样本缺少AudioPath字段，已跳过")
            continue
        filename = os.path.basename(sample["AudioPath"])
        if filename.lower().endswith(".wav"):
            required_files.add(filename)
    
    logging.info(f"从JSONL中提取到 {len(required_files)} 个需转换的音频文件（去重后）")
    print(f"需转换的音频文件：{len(required_files)} 个（去重后）")

    # 步骤3：匹配存在的文件并转换
    success_count = 0
    fail_count = 0
    skip_count = 0
    missing_files = []

    # 遍历需要转换的文件
    for filename in tqdm(required_files, desc="音频转换进度"):
        # 检查文件是否存在
        if filename not in audio_path_map:
            missing_files.append(filename)
            continue
        
        input_path = audio_path_map[filename]
        output_path = os.path.join(WAV_DIR, filename)
        
        # 跳过已转换的文件
        if os.path.exists(output_path):
            skip_count += 1
            continue
        
        # 执行转换（16kHz，保持单声道）
        try:
            result = subprocess.run(
                ["sox", input_path, "-r", "16000", "-c", "1", output_path],
                check=True,
                capture_output=True,
                text=True
            )
            success_count += 1
            logging.info(f"转换成功：{filename}")
        except subprocess.CalledProcessError as e:
            fail_count += 1
            logging.error(f"转换失败 {filename}：{e.stderr}")
            print(f"转换失败 {filename}，详情见日志")

    # 输出统计结果
    print("\n===== 转换结果 =====")
    print(f"成功转换：{success_count} 个")
    print(f"已跳过（已转换）：{skip_count} 个")
    print(f"转换失败：{fail_count} 个")
    print(f"文件不存在：{len(missing_files)} 个")
    
    logging.info("\n===== 转换结果 =====")
    logging.info(f"成功转换：{success_count} 个")
    logging.info(f"已跳过（已转换）：{skip_count} 个")
    logging.info(f"转换失败：{fail_count} 个")
    logging.info(f"文件不存在：{len(missing_files)} 个，列表：{missing_files[:10]}（仅显示前10个）")

if __name__ == "__main__":
    main()
    print("转换流程已结束，详情见 audio_conversion.log")
