# results/runs/ — 実験の記録(1実験 = 1ディレクトリ)

```
results/runs/
└─ 20261012_a0_numeric_fwd5/        ← YYYYMMDD_<実験名>
   ├─ config.yaml     この実験に使った設定(目的変数、特徴量、モデル、分割、乱数シード、yfinance取得日)
   ├─ metrics.json    評価指標(val / test を分ける。testは最終評価の1回だけ)
   ├─ notes.md        目的 / 結果 / 気づき / 次の仮説(数行でよい)
   ├─ git_commit.txt  実行時のコミットハッシュ(未コミットの変更があれば "dirty" と書く)
   └─ artifacts/      予測値・モデルなどの大きなファイル(git管理外)
```

- config.yaml・metrics.json・notes.md・git_commit.txt は**git管理する**。`artifacts/` は管理しない(`.gitignore`)。
- 論文に載せる最終の表・図だけを `results/tables/`、`results/figures/` に置く(スクリプトで再生成できること)。
- 実験名の例: `a0_numeric_fwd5`(数値のみ・5営業日平均)、`a2_finbert_fwd5`(銘柄ニュース・5営業日平均)。A0〜A3 は docs/RESEARCH_PLAN.md §5。
- 失敗した実験も消さずに notes.md に理由を残す(あとで「やらなかったこと」の根拠になる)。
