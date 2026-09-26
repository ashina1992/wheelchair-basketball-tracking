# models/

自前で学習した検出モデルの重み(`best.pt` など)を置く場所。重みファイルは Git 管理の対象外(`.gitignore`)。

- Colab(`notebooks/01_train_detector.ipynb`)で学習した `best.pt` をここに置く
- 解析: `python scripts/analyze.py <video> --profile ottawa2026 --weights models/best.pt --out ...`
- Web アプリ / Docker: 環境変数 `WBT_WEIGHTS=models/best.pt` を指定する(Docker イメージにはビルド時にこのフォルダが含まれる)
