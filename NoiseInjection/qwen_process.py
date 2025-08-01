import json
import librosa
import torch
import numpy as np
from itertools import product
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from tqdm import tqdm

# --- 1. 模型加载 ---
print("正在加载模型 Qwen/Qwen2-Audio-7B-Instruct...")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
print("模型加载完成。")


# --- 2. 文件路径定义 ---
input_file = "/mnt/data1/chendazhong/sampled_conerted_datasets.jsonl"
output_file = "/mnt/data1/chendazhong/qwen_audio_results_for_comparison_v2.jsonl"


# --- 3. CER 计算函数（与CER1完全一致） ---
def calculate_cer(reference: str, hypothesis: str) -> float:
    """计算字错率CER（与Whisper模型的CER1计算逻辑完全一致）"""
    # 移除空格处理，与CER1保持一致
    # 不做 reference.replace(" ", "") 和 hypothesis.replace(" ", "") 操作

    # 处理特殊情况
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 1.0
    if not hypothesis:
        return 1.0

    # 初始化编辑距离矩阵（与CER1相同的实现）
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    distance = np.zeros((ref_len + 1, hyp_len + 1), dtype=int)

    # 填充边界条件
    for i in range(ref_len + 1):
        distance[i][0] = i  # 删除操作
    for j in range(hyp_len + 1):
        distance[0][j] = j  # 插入操作

    # 计算编辑距离
    for i, j in product(range(1, ref_len + 1), range(1, hyp_len + 1)):
        cost = 0 if reference[i-1] == hypothesis[j-1] else 1
        distance[i][j] = min(
            distance[i-1][j] + 1,         # 删除
            distance[i][j-1] + 1,         # 插入
            distance[i-1][j-1] + cost     # 替换
        )

    # 返回字错率（编辑距离 / 参考文本长度）
    return distance[ref_len][hyp_len] / ref_len


# --- 4. 音频处理与ASR函数 ---
def process_audio(audio_path, text_instruction):
    """使用Qwen2-Audio模型处理单个音频文件并返回ASR结果。"""
    try:
        sampling_rate = processor.feature_extractor.sampling_rate
        audio_data, sr = librosa.load(audio_path, sr=sampling_rate)

        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_path": audio_path},
                {"type": "text", "text": text_instruction}
            ]}
        ]
        
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        inputs = processor(
            text=text,
            audios=audio_data,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        
        inputs = inputs.to(model.device, dtype=torch.bfloat16)

        generate_ids = model.generate(**inputs, max_new_tokens=512)
        
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return response
    except Exception as e:
        print(f"处理音频时出错: {audio_path}, 错误: {str(e)}")
        return f"处理错误: {str(e)}"


# --- 5. ASR结果后处理函数 ---
def postprocess_response(response):
    """清理模型输出，只保留纯净的转录文本。"""
    prefixes_to_remove = [
        "好的，根据提供的所有信息，最准确的识别结果是：",
        "好的，这段粤语音频的转录结果是：",
        "音频的转录结果如下：",
        "转录结果是："
    ]
    response = response.strip()
    for prefix in prefixes_to_remove:
        if response.startswith(prefix):
            response = response.replace(prefix, "", 1).strip()

    response = response.strip('\"\'`')
    return response if response else "[无法识别]"


# --- 6. 主处理循环 ---
print(f"开始处理文件: {input_file}")
results_to_write = []
with open(input_file, "r", encoding="utf-8") as f_in:
    lines = f_in.readlines()
    for line in tqdm(lines, desc="Qwen2-Audio ASR (New Prompt)"):
        try:
            data = json.loads(line)

            audio_path = data["audio"]["path"]
            ground_truth_sentence = data["sentence"]

            text_instruction = """
请转录以下粤语音频，严格遵循以下规则：
1. 完整保留粤语口语化表达（如“咁”“啲”“佢”等特色词汇），不替换为普通话词汇；
2. 音频中出现的英文单词、短语或缩写直接保留原文，不翻译为中文；
3. 按音频实际内容转录，不修正发音或语法“错误”（如口误、重复）；
4. 不添加任何标点符号、解释说明或额外内容，仅输出转录文本；
5. 若音频内容无法识别，直接输出“[无法识别]”。
"""

            raw_asr_result = process_audio(audio_path, text_instruction)
            clean_asr_result = postprocess_response(raw_asr_result)
            cer_score = calculate_cer(ground_truth_sentence, clean_asr_result)

            data["ASR2"] = clean_asr_result
            data["CER2"] = round(cer_score, 4)  # 与CER1保持相同的小数点位数
            results_to_write.append(data)

        except json.JSONDecodeError:
            print(f"警告: 跳过无法解析的行: {line.strip()}")
        except KeyError as e:
            print(f"警告: 跳过缺少关键字段的行: {e} in {line.strip()}")


# --- 7. 写入新文件 ---
print(f"处理完成，正在将结果写入到 {output_file}...")
with open(output_file, "w", encoding="utf-8") as f_out:
    for item in results_to_write:
        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"任务完成！结果已保存至 {output_file}")
    