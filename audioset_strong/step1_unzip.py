import os
import tarfile

os.environ['COPYFILE_DISABLE'] = '1'
PROCESSED_FILE_RECORD = "unziped_record.txt"

def load_processed_files(record_file):
    """
    加载已经处理的 tar 文件列表。
    """
    if os.path.exists(record_file):
        with open(record_file, 'r') as f:
            return set(f.read().splitlines())
    return set()

def save_processed_file(record_file, file_path):
    """
    保存已处理的 tar 文件路径到记录文件。
    """
    with open(record_file, 'a') as f:
        f.write(file_path + '\n')

def extract_tar_files_in_batches(directory, batch_size=5, record_file=PROCESSED_FILE_RECORD):
    """
    批量解压目录中的 .tar 文件，带有检查点机制，避免重复解压已处理的文件。
    """
    # 加载已经处理的文件
    processed_files = load_processed_files(record_file)

    # 找到目录中的 tar 文件，并且排除已经处理过的文件
    tar_files = [f for f in os.listdir(directory) if f.endswith('.tar') and os.path.join(directory, f) not in processed_files]
    
    total_files = len(tar_files)
    if total_files == 0:
        print(f"目录 {directory} 中没有新的 tar 文件。")
        return

    # 分批处理文件
    for i in range(0, total_files, batch_size):
        batch_files = tar_files[i:i+batch_size]
        print(f"正在解压批次 {i//batch_size + 1}: {batch_files}")
        
        for tar_file in batch_files:
            tar_path = os.path.join(directory, tar_file)
            try:
                with tarfile.open(tar_path, 'r') as tar:
                    # 过滤掉 ._ 开头的文件
                    members = [m for m in tar.getmembers() if not os.path.basename(m.name).startswith('._')]
                    tar.extractall(path=directory, members=members)
                print(f"成功解压 {tar_file}。")
                # 记录已处理的文件，使用完整路径
                save_processed_file(record_file, tar_path)
            except Exception as e:
                print(f"解压 {tar_file} 时出错: {e}")

def clean_dot_underscore_files(directory):
    """
    清理目录中的 ._ 开头的文件
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('._'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                except Exception as e:
                    print(f"删除 {file_path} 时出错: {e}")

def process_directories(base_directory, batch_size=5):
    """
    处理 train, test 和 valid 目录，并解压其中的 tar 文件。
    """
    for sub_dir in ['train', 'test', 'eval']:
        full_path = os.path.join(base_directory, sub_dir)
        if os.path.isdir(full_path):
            print(f"正在处理目录: {full_path}")
            extract_tar_files_in_batches(full_path, batch_size=batch_size)
            clean_dot_underscore_files(full_path)
        else:
            print(f"目录 {full_path} 未找到。")

if __name__ == "__main__":
    # 替换为你的基础目录路径
    base_dir = "./"
    
    process_directories(base_dir)
    