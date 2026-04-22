from huggingface_hub import login, upload_folder
import os
HF_TOKEN = os.environ["HF_TOKEN"]
login(token=HF_TOKEN)

upload_folder(folder_path="./source/statements", repo_id="kubovotny/ECB_TEXTS", repo_type="dataset", token=HF_TOKEN)
