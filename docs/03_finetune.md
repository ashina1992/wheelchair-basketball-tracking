# 03. ファインチューニングと評価

最終更新: 2026-10-17

## 1回目(100 枚)

### 学習条件

- ベースモデル: yolov8s(COCO 学習済み)
- データ: train 60 枚(3 試合)/ val 20 枚(Canada v USA)/ test 20 枚(USA v Japan)。分割は試合単位(`docs/02_dataset.md`)
- imgsz 1280、100 エポック、batch 8、cos_lr、左右反転あり・上下反転なし
- Google Colab(T4)、`notebooks/01_train_detector.ipynb`

### test セットでの評価

`python scripts/eval_detector.py data/dataset/yolo --split test --weights yolov8s.pt yolov8x.pt models/batch1.pt`
(既製モデルは person → player、sports ball → ball に読み替えて、同じ方法で計算。結果は `docs/results/detector_eval_batch1.json`)

| クラス | 既製 yolov8s | 既製 yolov8x | **自前 batch1(yolov8s)** | 正解の数 |
|---|---|---|---|---|
| player AP50 | 0.244 | 0.279 | **0.950** | 174 |
| player AP50-95 | 0.087 | 0.114 | **0.767** | |
| ball AP50 | 0.004 | 0.188 | **0.963** | 16 |
| ball AP50-95 | 0.003 | 0.094 | **0.417** | |
| rim AP50 | – | – | **1.000** | 18 |
| referee AP50 | – | – | **0.958** | 49 |

### 考察

- **player**: 既製モデルの AP が低いのは、主に箱の定義の違い(既製モデルは上半身だけを囲むことが多く、IoU 0.5 に届かない)と、ベンチ・観客を player として出すことによる。「選手+車いす全体」を1つの箱とする定義で学習したことで、接地点(コート座標変換に使う)まで含む箱が出るようになった
- **ball**: 既製の yolov8s ではほぼ検出できなかった(広角ショットで直径 40px 前後)。60 枚の学習で AP50 0.96 まで上がった。AP50-95 が 0.42 にとどまるのは、小さい物体では数 px のずれで IoU が大きく下がるため
- **注意**: test セットは 1 試合・20 枚(ボール 16 個)と小さく、数値のばらつきが大きい。また test の試合は学習データと同じ会場の中継で、別会場への汎化はこの評価では測れていない。2回目(合計 300 枚)で test を増やして評価し直す

## 2回目(合計 300 枚)

【未実施】
