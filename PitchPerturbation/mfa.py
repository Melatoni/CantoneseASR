import os
import json
import re

# --- 1. 配置路径 (请根据你的实际情况修改) ---
# 输入：包含新格式数据的JSONL文件
INPUT_JSONL = "/Users/kennychan/Downloads/MCE_Dataset/converted_data.jsonl" 
# 输出：生成的.lab文件将保存在这里
LAB_DIR = "/Users/kennychan/Downloads/MCE_Dataset/all_data_lab_files"
# 输入：你的粤语词典文件
CUSTOM_LEXICON_PATH = "/Users/kennychan/Downloads/MCE_Dataset/cantonese_lexicon.txt"

# 自动创建输出目录
os.makedirs(LAB_DIR, exist_ok=True)


# --- 2. 加载自定义词典 (这部分无需修改) ---
def load_custom_lexicon(path):
    """加载自定义词典，返回 {汉字: 读音} 的字典"""
    lexicon = {}
    if not os.path.exists(path):
        print(f"警告：自定义词典文件不存在 → {path}")
        return lexicon
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                print(f"自定义词典格式错误（行 {line_num}）：{line}")
                continue
            # 我们只需要词汇表，所以只存键
            lexicon[parts[0]] = True 
    print(f"成功加载自定义词典，共 {len(lexicon)} 个条目")
    return lexicon

# --- 3. 主程序块 (核心修改在这里) ---
if __name__ == '__main__':
    # 加载词典，建立我们已知的词汇表
    custom_lexicon = load_custom_lexicon(CUSTOM_LEXICON_PATH)
    known_chars = set(custom_lexicon.keys())

    # 读取包含新格式数据的JSONL文件
    try:
        with open(INPUT_JSONL, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误：输入文件不存在 {INPUT_JSONL}")
        exit()

    print(f"\n--- 开始处理 {len(samples)} 条记录 ---")
    
    # 遍历所有样本，生成.lab文件
    for sample_idx, sample in enumerate(samples):
        # --- 核心修改点 开始 ---
        # 从新的数据结构中安全地提取路径和文本
        audio_info = sample.get("audio", {})
        audio_path = audio_info.get("path")
        text = sample.get("sentence", "")
        
        # 如果缺少关键信息，则跳过此条记录
        if not audio_path or not text:
            print(f"警告: 记录 {sample_idx+1} 缺少 'audio.path' 或 'sentence' 字段，已跳过。")
            continue
        # --- 核心修改点 结束 ---

        # 提取文件名（不含扩展名），这部分代码很棒，无需修改
        # 无论原始文件是.mp3还是.wav，它都能正确处理
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        lab_path = os.path.join(LAB_DIR, f"{audio_basename}.lab")
        
        # --- 文本清理和过滤部分 (无需修改) ---
        # 1. 初步清理：移除所有非汉字和非字母数字的字符
        cleaned_text = re.sub(r'[^\u4e00-\u9FFFa-zA-Z0-9]', '', text)
        
        # 2. 关键过滤：只保留在自定义词典中存在的字
        valid_chars = [char for char in cleaned_text if char in known_chars]
        
        # 3. 格式化：在有效字符之间添加空格
        lab_content = ' '.join(valid_chars)
        
        # 4. 写入.lab文件
        if lab_content:
            with open(lab_path, "w", encoding="utf-8") as f:
                f.write(lab_content)
        else:
            # 如果原始句子里的所有字都不在词典里，则打印警告
            print(f"警告: 样本 '{audio_basename}' 的文本 '{text}' 清理后无有效词典内字符，跳过。")

    print(f"\n--- 处理完成 ---")
    print(f"生成的 .lab 文件已保存在: {LAB_DIR}")

