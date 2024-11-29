このメモはAtotti用に書いているので，適宜読み替えてください．

実行の過程で，依存関係解決のため，python-versionを切り替えたため，この通りに動かない可能性がある．(主にnumpy2.0と，torch+cu121関係)

## JSUTコーパスをHuggingFaceのdatasetsフォーマットに変換
※手順1, 2は既に実施済みのcsvがgitに載っているので，基本的には新規作成は不要．
1. JSUTコーパスの`transcript_utf8.txt`を適当に文字列置換等で相対パスに変換
2. 同様に文字列置換等で適当に`transcript_utf8.txt`をcsvに変換．(置換済みのファイルがjsut_ver1.1_basic5000.csvにあるので，新規作成する場合はこれを参考に)
3. HuggingFaceから自分のアカウントのアクセストークンを取得する．このとき，write権限が必要なので注意．
4. [huggingface-cli](https://huggingface.co/docs/huggingface_hub/main/guides/cli)をインストールし，ログインする．以降このトークンを利用してコードからprivateなリポジトリにアクセス可能になる．
    ```bash
    huggingface-cli login
    ```
5. `fromat_datasets.py`でHuggingFaceのdatasetsフォーマットに変換
    ```bash
    uv run datasets/format_datasets.py
    ```
6. HuggingFace上にリポジトリが作成されていることを確認 [Atotti/jsut-corpus-datasets](https://huggingface.co/datasets/Atotti/jsut-corpus-datasets/tree/main)


## 注釈付けの実行
### 1. 以下のコマンドを実行して，注釈付けを行う．
CPU12コアでbasic5000処理するのに1h30m程度かかったので，GPU使えるようにしといた方が良さそう．
```bash
uv run main.py "Atotti/jsut-corpus-datasets" --configuration "default" --audio_column_name "audio" --text_column_name "text" --cpu_num_workers 8 --repo_id "jsut-corpus-tags-v1" --apply_squim_quality_estimation
 ```
HuggingFaceにpushされるが，publicになっていたので，privateに変更する．引数与えられるか，コードを修正するかした方が良い(publicだとdatasetcardが見れるので，良いのだが...)

### 2. 以下のコマンドを実行して，連続変数を離散的なキーワードにマッピングする．
すぐに完了する．(Map continuous annotations to key-words)

※ 単一話者のデータセットであるので，`--avoid_pitch_computation`を指定した．
```bash
uv run ./scripts/metadata_to_text.py "Atotti/jsut-corpus-tags-v1" --repo_id "jsut-corpus-keywords-v1" --configuration "default" --cpu_num_workers "8"--avoid_pitch_computation --apply_squim_quality_estimation
```
HuggingFaceにpushされるが，publicになっていたので，privateに変更する．(publicだとdatasetcardが見れるので，良いのだが...)

### 3. 自然言語による説明を生成する
このステップでは個別の特徴をLLMに入力して，自然言語の説明を生成させる．そのためLLMを実行する必要があるが，そんなGPUリソースは存在しない．そのため，解決案としては，GPT4o-mini等をAPI経由で利用する方法と，手元のGPU(RTX 3050 8GB)でgoogle/gemma-2b-itをbfloat16で実行する方法が考えられる．(普段A100で30BくらいのLLM動かしてるのってなかなかできないことなんだなあ...)

gemma-2b-itはbfloat16で動かすことで，8GBのGPUでも動かせるようになる．ここでLLM知識が役立つとは...([Total Size 4.67GB](https://huggingface.co/spaces/hf-accelerate/model-memory-usage))

以下のスクリプトは[run_prompt_creation_jenny.sh](https://github.com/huggingface/dataspeech/blob/main/examples/prompt_creation/run_prompt_creation_jenny.sh)を参考にし，gemma-2b-itをbfloat16で動かすように調整したものである．

ここで，JSUTコーパスの話者の名前を"Tomoko"とする．(適当に選択)gemmaは日本語も解釈できるが，予期せぬ挙動を避けるためにアルファベットで話者名を指定している．

GPUを使えば，現実的な時間で完了する．torchでGPUを使えるようにするのに若干てまどったが，torch.cuda.is_available()がTrueを返すようになればGPUが使える．
```bash
uv run ./scripts/run_prompt_creation.py --speaker_name "Tomoko" --is_single_speaker --is_new_speaker_prompt --dataset_name "Atotti/jsut-corpus-keywords-v1" --dataset_config_name "default" --model_name_or_path "google/gemma-2b-it" --per_device_eval_batch_size 4 --attn_implementation "sdpa" --dataloader_num_workers 4 --torch_dtype "bfloat16" --load_in_4bit --push_to_hub --hub_dataset_id "tomoko-tts-tagged-v1" --preprocessing_num_workers 4 --output_dir "./output"
```

最後に，HuggingFaceを確認する．
これにて，JSUTコーパスの注釈付けが完了した．
