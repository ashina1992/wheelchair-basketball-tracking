# デプロイ手順

Docker イメージ1つに、フロントエンド・API・解析ワーカーがすべて入っている(`Dockerfile`)。
GPU は不要(CPU 推論)。解析は 1 分の映像あたり CPU で 6〜7 分程度かかる(Ryzen 5 5625U での実測: 82 秒の映像で約 520 秒)。

## ローカルで動かす

```bash
docker build -t wbt .
docker run --rm -p 8000:8000 -v wbt-data:/data wbt
# → http://localhost:8000
```

- 同梱のデモ(`demo/`、映像なし・解析結果のみ)が最初から表示される
- 映像付きのデモを見たい場合は、`/data/demo/<名前>/` に `analysis.json` と `video.mp4` を置く

## 公開環境の構成方針

| 項目 | 方針 | 理由 |
|---|---|---|
| 公開する内容 | 同梱デモ(解析結果のみ)の閲覧を主にする | 試合映像そのものは再配布しない(YouTube 利用規約・放映権) |
| アップロード解析 | 有効にするが、`WBT_MAX_SECONDS=60` で先頭 60 秒に制限する | CPU 推論が重く、長い動画で他の利用者を待たせないため |
| アップロード上限 | `WBT_MAX_UPLOAD_MB=200` | ディスク容量の保護 |
| マシン | 2 vCPU / 4GB メモリ以上 | torch + YOLOv8s の推論に 2GB 程度使う |
| 永続化 | `/data` にボリュームをマウント | ジョブの状態と結果をファイルで保持するため |

## 候補のサービス

| サービス | 向き・不向き |
|---|---|
| **Fly.io**(推奨) | Docker イメージをそのまま動かせ、ボリュームも使える。常時起動の小さな VM で、バックグラウンドの解析ワーカーが止められない |
| Render(Web Service + Disk) | 同様に動く。ディスクは有料プランが必要 |
| Google Cloud Run | リクエストがないと CPU が絞られ、バックグラウンド解析が進まない(「CPU を常に割り当てる」設定が必要)。ファイルシステムも永続しない。**デモ閲覧だけ**なら安く運用できる |
| 自前の VPS | `docker run` するだけ。HTTPS はリバースプロキシ(Caddy 等)で付ける |

### Fly.io の例

```bash
fly launch --no-deploy            # アプリ名・リージョン(nrt)を選ぶ。Dockerfile が自動で使われる
fly volumes create wbt_data --size 3 --region nrt
# fly.toml に以下を追記
#   [mounts]
#     source = "wbt_data"
#     destination = "/data"
#   [env]
#     WBT_MAX_SECONDS = "60"
#     WBT_MAX_UPLOAD_MB = "200"
#   [[vm]]
#     size = "shared-cpu-2x"
#     memory = "4gb"
fly deploy
```

## 自前モデルに差し替える

1. Colab で学習した `best.pt` を `models/` に置く
2. `WBT_WEIGHTS=models/best.pt` を環境変数に設定してビルド・デプロイする
3. 同梱デモも自前モデルで解析し直す: `python scripts/analyze.py <video> --profile ottawa2026 --weights models/best.pt --out demo/jpn_pol` のあと、`demo/jpn_pol/` から映像以外(`analysis.json`, `*.csv`)だけを残す

## ライセンス上の注意

検出に使っている ultralytics は **AGPL-3.0**。これを組み込んだサービスをネットワーク越しに公開する場合、
サービスのソースコードを AGPL-3.0 で公開する必要がある(本リポジトリを GitHub で公開し、AGPL-3.0 とすれば満たせる)。
