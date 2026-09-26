# 車いすバスケ スタッツ・トラッキング(API + フロントエンド + 解析ワーカー、CPU 推論)
FROM python:3.11-slim

# tesseract: スコアボード OCR、libgl1/libglib2.0-0: OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU 版 torch を先に入れる(GPU 版より大幅に小さい)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
# 編集可能モードで入れる(profiles/ web/ demo/ をリポジトリ直下から参照するため)
RUN pip install --no-cache-dir -e ".[api]"

COPY profiles ./profiles
COPY web ./web
COPY demo ./demo

# 既製の検出モデルをイメージに含める(起動時のダウンロードを避ける)。
# 自前で学習した重みを使う場合は models/ に置いて WBT_WEIGHTS=models/best.pt を指定する
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
COPY models ./models

ENV WBT_STORAGE=/data \
    WBT_WEIGHTS=yolov8s.pt \
    PORT=8000
VOLUME /data
EXPOSE 8000

CMD ["sh", "-c", "uvicorn wbt.api.app:create_app --factory --host 0.0.0.0 --port ${PORT}"]
