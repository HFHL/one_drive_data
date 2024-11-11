import os
import json
import pandas as pd
import soundfile as sf
from datasets import Dataset, Features, Value, Audio, Sequence
import gc  # 引入垃圾回收模块
from soundfile import LibsndfileError

os.environ['COPYFILE_DISABLE'] = '1'

parent_directory_name = os.path.basename(os.path.dirname(os.path.abspath("./data")))


PROCESSED_FILES_RECORD = "processed_files.txt"  # 保存已处理文件路径的记录文件

# 定义函数，加载已处理文件列表
def load_processed_files(record_file):
    """
    从记录文件加载已经处理的文件列表，返回已处理文件的集合。
    """
    if os.path.exists(record_file):
        with open(record_file, 'r') as f:
            return set(f.read().splitlines())  # 读取文件中的所有路径，并返回集合
    return set()

# 定义函数，保存已处理的文件路径
def save_processed_file(record_file, file_path):
    """
    将已处理的文件路径保存到记录文件中。
    """
    with open(record_file, 'a') as f:
        f.write(file_path + '\n')

# 定义函数，处理每个文件夹，按批次返回数据
def process_directory_in_batches(directory_path, datasetname, batch_size=20, record_file=PROCESSED_FILES_RECORD):
    data_batch = []
    processed_files = load_processed_files(record_file)  # 加载已处理文件列表
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.flac'):
                audio_path = os.path.join(root, file)
                
                # 如果该文件已处理，则跳过
                if audio_path in processed_files:
                    continue
                
                json_path = audio_path.replace('.flac', '.json')

                try:
                    # 读取音频数据和采样率
                    audio_data, sample_rate = sf.read(audio_path)
                    audio_len = len(audio_data) / sample_rate  # 计算音频时长（秒）
                except LibsndfileError as e:
                    print(f"Error reading {audio_path}: {e}")
                    continue  # 跳过无法读取的文件

                # 读取对应的JSON文件
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                else:
                    metadata = {}

                # 使用相对路径生成唯一的 index
                audio_id = os.path.relpath(audio_path, directory_path)
                index = audio_id.replace(os.sep, '_').split('.')[0]
                
                # 创建单个音频文件的记录
                record = {
                    'index': index,
                    'datasetname': datasetname,
                    'audio': {
                        'array': audio_data,
                        'sampling_rate': sample_rate
                    },
                    'audio_len': audio_len
                }

                # 处理文本字段
                if 'text' in metadata:
                    record['text'] = metadata['text']
                else:
                    record['text'] = ''

                # 处理raw_text字段，包含所有重要信息
                raw_text = []
                if metadata:
                    # 添加基本文本描述
                    raw_text.append(f"text: {metadata.get('text', '')}")
                    
                    # 添加标签信息
                    if 'tag' in metadata:
                        raw_text.append(f"tags: {', '.join(metadata['tag'])}")
                    
                    # 添加强标注事件信息
                    if 'strong_event' in metadata:
                        for event in metadata['strong_event']:
                            start, end, label, mid = event
                            raw_text.append(f"event: {label} ({start:.2f}s-{end:.2f}s)")
                    
                    # 添加原始数据中的类别标签
                    if 'original_data' in metadata and 'class_names' in metadata['original_data']:
                        raw_text.append(f"classes: {', '.join(metadata['original_data']['class_names'])}")

                record['raw_text'] = raw_text

                data_batch.append(record)

                # 保存当前文件路径到记录文件
                save_processed_file(record_file, audio_path)

                # 批次处理，达到指定大小时返回该批数据
                if len(data_batch) >= batch_size:
                    yield data_batch
                    data_batch = []

    # 如果还有剩余数据，返回最后一批
    if data_batch:
        yield data_batch

# 定义函数，保存为 .parquet 文件
def save_parquet_files(data, output_dir, datasetname, subdir, batch_num):
    total_records = len(data)
    
    # 确保输出子目录存在
    output_subdir = os.path.join(output_dir, subdir)
    os.makedirs(output_subdir, exist_ok=True)
    
    # 获取当前分段的起止 index 值用于命名
    first_index = os.path.basename(data[0]['index'])
    last_index = os.path.basename(data[-1]['index'])
    
    # 生成文件名，例如：train-0-train-100.parquet
    filename = f"{subdir}-{first_index}-{last_index}-batch{batch_num}.parquet"
    output_path = os.path.join(output_subdir, filename)
    
    # 转换为 DataFrame
    df = pd.DataFrame(data)
    
    # 定义 Hugging Face datasets 的 features
    features = Features({
        'index': Value('string'),
        'datasetname': Value('string'),
        'audio': Audio(),  # audio字段定义为Hugging Face的audio类型，内嵌音频数据
        'audio_len': Value('float32'),  # 音频长度（秒）
        'text': Value('string'),
        'raw_text': Sequence(Value('string'))  # raw_text定义为sequence类型
    })

    # 转换为 Huggingface Dataset 格式
    dataset = Dataset.from_pandas(df, features=features)

    # 保存为 .parquet 文件
    dataset.to_parquet(output_path)
    print(f"Saved {output_path}")

# 定义主函数，手动指定train、test、valid目录的输出路径
def main(train_dir, test_dir, valid_dir, train_output_dir, test_output_dir, valid_output_dir, datasetname=parent_directory_name):
    # 处理 train 目录
    batch_num = 0
    for data_batch in process_directory_in_batches(train_dir, datasetname):
        save_parquet_files(data_batch, train_output_dir, datasetname, "train", batch_num)
        del data_batch  # 释放内存
        gc.collect()  # 手动进行垃圾回收
        batch_num += 1

    # 处理 test 目录
    if test_dir:
        batch_num = 0
        for data_batch in process_directory_in_batches(test_dir, datasetname):
            save_parquet_files(data_batch, test_output_dir, datasetname, "test", batch_num)
            del data_batch  # 释放内存
            gc.collect()  # 手动进行垃圾回收
            batch_num += 1

    # 处理 valid 目录
    if valid_dir:
        batch_num = 0
        for data_batch in process_directory_in_batches(valid_dir, datasetname):
            save_parquet_files(data_batch, valid_output_dir, datasetname, "valid", batch_num)
            del data_batch  # 释放内存
            gc.collect()  # 手动进行垃圾回收
            batch_num += 1

# 使用方法
if __name__ == "__main__":
    train_dir = "./train"  # 现在只需要指定train的顶级目录
    test_dir = "./test"  # 指定test目录路径
    valid_dir = "./eval"  # 指定valid目录路径
    
    train_output_dir = "./data/data/"  # 指定train生成的parquet文件路径
    test_output_dir = "./data/data"  # 指定test生成的parquet文件路径
    valid_output_dir = "./data/data"  # 指定valid生成的parquet文件路径
    
    main(train_dir, test_dir, valid_dir, train_output_dir, test_output_dir, valid_output_dir)
