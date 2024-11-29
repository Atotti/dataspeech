このメモはAtotti用に書いているので，適宜読み替えてください．

## JSUTコーパスをHuggingFaceのdatasetsフォーマットに変換

1. JSUTコーパスの`transcript_utf8.txt`を適当に文字列置換等で相対パスに変換 & csvに変換
2. HuggingFaceのトークンを`.env`に設定．このとき，write権限が必要
3. `fromat_datasets.py`でHuggingFaceのdatasetsフォーマットに変換
```bash
uv run datasets/format_datasets.py
```
1. HuggingFace上にリポジトリが作成されていることを確認 [Atotti/jsut-corpus-datasets](https://huggingface.co/datasets/Atotti/jsut-corpus-datasets/tree/main)

## 注釈付けの実行
1. 以下のコマンドを実行して，注釈付けを行う
```bash
uv run main.py "Atotti/jsut-corpus-datasets" --configuration "default" --audio_column_name "audio" --text_column_name "text" --cpu_num_workers 8 --repo_id "jsut-corpus-tags-v1" --apply_squim_quality_estimation
```
