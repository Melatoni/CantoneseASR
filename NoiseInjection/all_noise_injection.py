# ==============================================================================
#           优化后，输出更干净的 all_noise_injection.py 脚本
# ==============================================================================
import os
import json
import numpy as np
import soundfile as sf
from tqdm import tqdm
import torch
import librosa
import random
from itertools import product
import logging # <-- 新增

# 屏蔽 transformers 的 INFO 级别日志，让控制台更干净
logging.getLogger("transformers").setLevel(logging.WARNING)

from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    AutoProcessor as QwenAudioProcessor,
    Qwen2AudioForConditionalGeneration,
)

# ... [配置参数部分和核心功能函数部分保持不变，这里省略] ...
# -------------------------- 1. 配置参数 (请根据您的环境修改) --------------------------
# --- 输入路径 ---
CLEAN_JSONL_PATH = "/mnt/data1/chendazhong/sampled_conerted_datasets.jsonl"
NOISE_DIR = "/mnt/data1/chendazhong/Whisper_MCE/librivox"
# --- 输出路径 ---
BASE_OUTPUT_DIR = "/mnt/data1/chendazhong/Whisper_MCE/noisy_experiment_final"
# --- 模型标识符 ---
WHISPER_MODEL_PATH = "cantonese_models/whisper-tiny-finetune"
QWEN_AUDIO_MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"
# --- 处理参数 ---
SNR_LEVELS_DB = [20, 10, 5, 0, -5, -10]
TARGET_SR = 16000
# --- 设备配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
WHISPER_TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# -------------------------- 2. 核心功能函数 (与之前相同) --------------------------
def load_audio_resample(file_path):
    try:
        audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
        return audio
    except Exception as e:
        # 打印更明确的警告信息
        print(f"\n[警告] 加载音频失败，将跳过: {file_path}. 错误: {e}")
        return None

def add_noise(clean_audio, noise_audio, snr_db):
    if len(noise_audio) < len(clean_audio):
        repeats = int(np.ceil(len(clean_audio) / len(noise_audio)))
        noise_audio = np.tile(noise_audio, repeats)[:len(clean_audio)]
    else:
        start = np.random.randint(0, len(noise_audio) - len(clean_audio) + 1)
        noise_audio = noise_audio[start:start + len(clean_audio)]
    clean_power = np.mean(clean_audio ** 2)
    noise_power = np.mean(noise_audio ** 2)
    if clean_power == 0: return noise_audio
    if noise_power == 0: return clean_audio
    scale = np.sqrt(clean_power / (noise_power * 10 ** (snr_db / 10)))
    noisy_audio = clean_audio + noise_audio * scale
    max_val = np.max(np.abs(noisy_audio))
    if max_val > 1.0:
        noisy_audio = noisy_audio / max_val
    return noisy_audio

def calculate_cer(reference: str, hypothesis: str) -> float:
    if not reference and not hypothesis: return 0.0
    if not reference: return 1.0
    if not hypothesis: return 1.0
    ref_len, hyp_len = len(reference), len(hypothesis)
    distance = np.zeros((ref_len + 1, hyp_len + 1), dtype=int)
    for i in range(ref_len + 1): distance[i][0] = i
    for j in range(hyp_len + 1): distance[0][j] = j
    for i, j in product(range(1, ref_len + 1), range(1, hyp_len + 1)):
        cost = 0 if reference[i-1] == hypothesis[j-1] else 1
        distance[i][j] = min(distance[i-1][j] + 1, distance[i][j-1] + 1, distance[i-1][j-1] + cost)
    return distance[ref_len][hyp_len] / ref_len

# -------------------------- 3. 模型加载与推理 (推理函数已优化) --------------------------
def load_models():
    print("正在加载ASR模型...")
    print(f"加载Whisper模型: {WHISPER_MODEL_PATH}")
    whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_PATH)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_MODEL_PATH, torch_dtype=WHISPER_TORCH_DTYPE,
    ).to(DEVICE)
    whisper_model.eval()
    print(f"加载Qwen2-Audio模型: {QWEN_AUDIO_MODEL_PATH}")
    qwen_processor = QwenAudioProcessor.from_pretrained(QWEN_AUDIO_MODEL_PATH, trust_remote_code=True)
    qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        QWEN_AUDIO_MODEL_PATH, torch_dtype=QWEN_TORCH_DTYPE,
        device_map="auto", trust_remote_code=True
    )
    qwen_model.eval()
    print("所有模型加载完毕。")
    return {"whisper": (whisper_model, whisper_processor), "qwen": (qwen_model, qwen_processor)}

def run_whisper_inference(model, processor, audio_input):
    """【优化】使用微调的Whisper模型进行推理 (ASR1)"""
    inputs = processor(audio_input, sampling_rate=TARGET_SR, return_tensors="pt")
    input_features = inputs.input_features.to(DEVICE).to(WHISPER_TORCH_DTYPE)
    with torch.no_grad():
        predicted_ids = model.generate(input_features, task="transcribe")
    asr_result = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return asr_result.strip()

def run_qwen_inference(model, processor, audio_input):
    """【优化】使用Qwen2-Audio模型进行推理 (ASR2)，消除attention_mask警告"""
    text_instruction = """
请转录以下粤语音频，严格遵循以下规则：
1. 完整保留粤语口语化表达（如“咁”“啲”“佢”等特色词汇），不替换为普通话词汇；
2. 音频中出现的英文单词、短语或缩写直接保留原文，不翻译为中文；
3. 按音频实际内容转录，不修正发音或语法“错误”（如口误、重复）；
4. 不添加任何标点符号、解释说明或额外内容，仅输出转录文本；
5. 若音频内容无法识别，直接输出“[无法识别]”。
"""
    conversation = [{"role": "user", "content": [
        {"type": "audio", "audio": audio_input, "sampling_rate": TARGET_SR},
        {"type": "text", "text": text_instruction}
    ]}]
    
    # 【优化】将音频和文本一起处理，让processor生成attention_mask
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    model_inputs = processor(text=text, audios=audio_input, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, do_sample=False, max_new_tokens=512)
        
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    prefixes_to_remove = [
        "好的，根据提供的所有信息，最准确的识别结果是：", "好的，这段粤语音频的转录结果是：",
        "音频的转录结果如下：", "转录结果是："
    ]
    # 【优化】使用更简洁的后处理
    for prefix in prefixes_to_remove:
        response = response.removeprefix(prefix).strip()
    response = response.strip('\"\'`')
    return response if response else "[无法识别]"


# -------------------------- 4. 主处理流程 (与之前相同) --------------------------
def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    models = load_models()
    print(f"从 {CLEAN_JSONL_PATH} 加载原始数据...")
    clean_records = []
    with open(CLEAN_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                if "audio" in record and "path" in record["audio"] and "sentence" in record:
                     if os.path.exists(record["audio"]["path"]):
                        clean_records.append(record)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"警告：跳过格式错误或缺少关键字段的行: {line.strip()} - {e}")
    print(f"找到 {len(clean_records)} 条有效的原始音频记录。")
    noise_paths = [os.path.join(NOISE_DIR, f) for f in os.listdir(NOISE_DIR) if f.endswith(('.wav', '.flac', '.mp3'))]
    if not noise_paths:
        raise FileNotFoundError(f"在目录 {NOISE_DIR} 中未找到任何噪音文件。")
    print(f"找到 {len(noise_paths)} 个噪音文件。")
    for snr_db in SNR_LEVELS_DB:
        print(f"\n{'='*20} 开始处理 SNR = {snr_db} dB {'='*20}")
        output_audio_dir = os.path.join(BASE_OUTPUT_DIR, f"noisy_audio_snr_{snr_db}")
        os.makedirs(output_audio_dir, exist_ok=True)
        output_jsonl_path = os.path.join(BASE_OUTPUT_DIR, f"results_snr_{snr_db}.jsonl")
        with open(output_jsonl_path, "w", encoding="utf-8") as f_out:
            progress_bar = tqdm(enumerate(clean_records), total=len(clean_records), desc=f"SNR={snr_db}dB")
            for i, clean_record in progress_bar:
                original_audio_path = None
                try:
                    original_audio_path = clean_record["audio"]["path"]
                    ground_truth = clean_record["sentence"]
                    clean_audio = load_audio_resample(original_audio_path)
                    if clean_audio is None: continue
                    noise_path = random.choice(noise_paths)
                    noise_audio = load_audio_resample(noise_path)
                    if noise_audio is None: continue
                    noisy_audio = add_noise(clean_audio, noise_audio, snr_db)
                    clean_basename = os.path.splitext(os.path.basename(original_audio_path))[0]
                    noisy_filename = f"{clean_basename}_snr{snr_db}_{i:04d}.wav"
                    noisy_audio_path = os.path.join(output_audio_dir, noisy_filename)
                    sf.write(noisy_audio_path, noisy_audio, TARGET_SR)
                    asr1_text = run_whisper_inference(models["whisper"][0], models["whisper"][1], noisy_audio)
                    asr2_text = run_qwen_inference(models["qwen"][0], models["qwen"][1], noisy_audio)
                    cer1 = calculate_cer(ground_truth, asr1_text)
                    cer2 = calculate_cer(ground_truth, asr2_text)
                    output_record = {
                        "Text": ground_truth, "NoisyAudioPath": noisy_audio_path,
                        "OriginalAudioPath": original_audio_path, "SNR_dB": snr_db,
                        "ASR1": asr1_text, "CER1": round(cer1, 4),
                        "ASR2": asr2_text, "CER2": round(cer2, 4)
                    }
                    f_out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                except Exception as e:
                    path_info = f"({original_audio_path})" if original_audio_path else ""
                    print(f"\n处理记录 {i} {path_info} 时发生严重错误: {e}，跳过此条目。")
        print(f"SNR = {snr_db} dB 处理完成。")
        print(f"带噪音频已保存至: {output_audio_dir}")
        print(f"结果JSONL已保存至: {output_jsonl_path}")
    print("\n所有处理任务已完成！")

if __name__ == "__main__":
    main()
