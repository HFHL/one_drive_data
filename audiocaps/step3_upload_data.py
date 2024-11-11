import os
from huggingface_hub import HfApi, create_repo

os.environ['COPYFILE_DISABLE'] = '1'

# 获取当前目录的父目录名
parent_directory_name = os.path.basename(os.path.dirname(os.path.abspath("./data")))

# 打印父目录名字
print(f"Parent directory name: {parent_directory_name}")

# 创建 repo_id 并拼接父目录名字
repo_id = f"CLAPv2/{parent_directory_name}"

print(repo_id)

# 初始化 Hugging Face API
api = HfApi()

# 上传文件夹到拼接后的 repo_id
api.upload_folder(
    folder_path="./data",
    repo_id=repo_id,
    repo_type="dataset",
)