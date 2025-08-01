import os
import librosa
import soundfile as sf
from tqdm import tqdm

def convert_and_consolidate_audio(source_folders, target_folder, target_sr=16000):
    """
    将多个源文件夹中的WAV和MP3文件转换为16kHz的WAV并整合到目标文件夹
    
    参数:
        source_folders: 源文件夹列表
        target_folder: 目标文件夹路径
        target_sr: 目标采样率，默认16000Hz
    """
    # 创建目标文件夹
    os.makedirs(target_folder, exist_ok=True)
    
    # 跟踪所有文件名，避免重复
    all_filenames = set()
    
    # 遍历所有源文件夹
    for src_folder in source_folders:
        # 检查源文件夹是否存在
        if not os.path.exists(src_folder):
            print(f"警告: 源文件夹不存在 - {src_folder}")
            continue
            
        print(f"正在处理文件夹: {src_folder}")
        
        # 递归遍历文件夹中的所有文件
        for root, _, files in os.walk(src_folder):
            # 过滤出WAV和MP3文件（不区分大小写）
            audio_files = [
                f for f in files 
                if f.lower().endswith(('.wav', '.mp3'))
            ]
            
            # 处理每个音频文件
            for filename in tqdm(audio_files, desc=f"处理 {os.path.basename(src_folder)}"):
                src_path = os.path.join(root, filename)
                
                # 处理文件名（统一转为.wav扩展名）
                name, ext = os.path.splitext(filename)
                target_filename = f"{name}.wav"  # 无论原格式，都转为wav
                counter = 1
                while target_filename in all_filenames:
                    target_filename = f"{name}_{counter}.wav"
                    counter += 1
                all_filenames.add(target_filename)
                
                target_path = os.path.join(target_folder, target_filename)
                
                try:
                    # 加载音频（自动处理wav和mp3）并转换采样率
                    # 注意：librosa加载mp3需要ffmpeg支持，若报错请安装ffmpeg
                    audio, sr = librosa.load(src_path, sr=target_sr)
                    
                    # 保存为16kHz的WAV文件
                    sf.write(target_path, audio, target_sr)
                    
                except Exception as e:
                    print(f"处理文件失败 {src_path}: {str(e)}")

if __name__ == "__main__":
    # 源文件夹列表（包含clips文件夹，其中有mp3文件）
    source_folders = [
        "/Users/kennychan/Downloads/MCE_Dataset/clips"  # 包含mp3文件的文件夹
    ]
    
    # 目标文件夹（所有文件将整合到这里，统一为wav格式）
    target_folder = "/Users/kennychan/Downloads/MCE_Dataset/consolidated_wavs"
    
    # 执行转换和整合
    print("开始处理音频文件（支持WAV和MP3）...")
    convert_and_consolidate_audio(source_folders, target_folder, target_sr=16000)
    print(f"所有文件已处理完成，保存至: {target_folder}")
    