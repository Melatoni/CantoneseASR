import json
import random
import torch
import soundfile as sf
import librosa  # 新增：用于音频重采样
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig

def calculate_cer(reference: str, hypothesis: str) -> float:
    """计算字错率CER"""
    import numpy as np
    from itertools import product

    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 1.0
    if not hypothesis:
        return 1.0

    ref_len = len(reference)
    hyp_len = len(hypothesis)
    distance = np.zeros((ref_len + 1, hyp_len + 1), dtype=int)

    for i in range(ref_len + 1):
        distance[i][0] = i
    for j in range(hyp_len + 1):
        distance[0][j] = j

    for i, j in product(range(1, ref_len + 1), range(1, hyp_len + 1)):
        cost = 0 if reference[i-1] == hypothesis[j-1] else 1
        distance[i][j] = min(
            distance[i-1][j] + 1,
            distance[i][j-1] + 1,
            distance[i-1][j-1] + cost
        )

    return distance[ref_len][hyp_len] / ref_len

def main():
    input_file = "/mnt/data1/chendazhong/Whisper_MCE/converted_data.jsonl"
    output_file = "sampled_conerted_datasets.jsonl"
    model_path = "cantonese_models/whisper-tiny-finetune"
    sample_size = 5000
    target_sr = 16000  # 强制转为模型要求的16000Hz
    language = "yue"
    use_gpu = torch.cuda.is_available()

    # 加载模型和处理器
    print("加载模型和处理器...")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.fr、】
    
    model.generation_config = generation_config

    device = "cuda" if use_gpu else "cpu"
    model.to(device)
    model.eval()

    # 读取并随机抽取数据
    print("读取并抽取数据...")
    with open(input_file, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
    
    if len(all_data) < sample_size:
        sample_size = len(all_data)
        print(f"警告：数据量不足，将使用全部{sample_size}条数据")
    
    sampled_data = random.sample(all_data, sample_size)

    # 处理每条数据
    print("开始处理数据...")
    results = []
    for item in tqdm(sampled_data, total=sample_size):
        try:
            # 读取音频并强制重采样到16000Hz
            audio_path = item["audio"]["path"]
            audio, sample_rate = librosa.load(audio_path, sr=target_sr)  # 用librosa自动重采样

            # 预处理
            inputs = processor(audio, sampling_rate=target_sr, return_tensors="pt").to(device)

            # 生成ASR结果（不指定language参数，避免冲突）
            with torch.no_grad():
                predicted_ids = model.generate(**inputs, task="transcribe")
            asr_result = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            # 计算CER
            reference = item["sentence"]
            cer = calculate_cer(reference, asr_result)

            # 保存结果
            result = {
                "audio": item["audio"],
                "sentence": reference,
                "duration": item["duration"],
                "language": item["language"],
                "ASR1": asr_result,
                "CER1": round(cer, 4)
            }
            results.append(result)
        except Exception as e:
            print(f"处理音频{audio_path}时出错：{str(e)}，跳过该样本")
            continue

    # 保存结果
    print(f"保存结果到{output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"完成！共处理{len(results)}/{sample_size}条数据")

if __name__ == "__main__":
    main()