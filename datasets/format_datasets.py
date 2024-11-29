from datasets import DatasetDict, Audio
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")

path = os.path.realpath(__file__)
csv_path = os.path.join(os.path.dirname(path), "jsut_ver1.1_basic5000.csv")

# CSVファイルからデータセットを作成
dataset = DatasetDict.from_csv({"train": csv_path})

# audioカラムのパスを絶対パスに変換
def convert_to_absolute_path(example):
    example["audio"] = os.path.join(os.path.dirname(path), example["audio"])
    return example

dataset["train"] = dataset["train"].map(convert_to_absolute_path)

# 音声データのカラムをAudio型にキャスト
dataset = dataset.cast_column("audio", Audio())

# データセットをHuggingFace Hubにプッシュ（JSUTコーパスの再配布は許可されていないのでプライベートに）
dataset.push_to_hub("Atotti/jsut-corpus-datasets", private=True, token=token)
