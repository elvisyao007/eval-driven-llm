# DeepDoc Eval v1 — Phase 1 観察ノート

実施日: 2026-06-12  
実施者: yaoqichan  
DeepDocバージョン: RAGFlow main (sparse-checkout @ 2026-06-12, `2934f9e` 相当)  
ハードウェア: CPU only (RTX 5090 は本 Phase では使用しない)

---

## 1. 環境構築で当たった坑

### 坑 1: `requires-python = ">=3.13"` だがシステムは Python 3.12
`pyproject.toml` の制約は全体プロジェクト向け。deepdoc モジュール自体は 3.12 でも問題なく動いた。  
**対処**: Python 3.12 の venv で進める。3.13 への移行は Phase 2 でも据え置き可。

### 坑 2: `common/settings.py` が完全な RAGFlow サービススタック (rag.utils, memory.utils 等) をインポート
deepdoc が settings から実際に使うのは `PARALLEL_DEVICES: int = 0` と `DOC_ENGINE_INFINITY: bool = False` の 2 変数のみ。  
**対処**: `common/settings.py` を最小スタブに差し替え (オリジナルは `.orig` でバックアップ)。

### 坑 3: `t_ocr.py` の sys.path.insert が `common` インポートより**後**に来る
`t_recognizer.py` は path insert が先なので `python t_recognizer.py` で動く。  
`t_ocr.py` は `common.misc_utils` のインポートが path insert より先に書かれているため、カレントディレクトリに `common/` がないと失敗する。  
**対処**: 常に `PYTHONPATH=/mnt/data/ragflow-deepdoc` を指定して実行。

### 坑 4: NLTK データ不足 (TSR 実行時)
`rag.nlp.rag_tokenizer` (TSR の `blockType` 判定に使用) が `punkt_tab`, `wordnet`, `averaged_perceptron_tagger_eng`, `stopwords` を要求。  
**対処**: `python -c "import nltk; nltk.download(...)"` で手動ダウンロード。

### 坑 5: `infinity-sdk==0.7.0` が必要 (`rag.nlp.rag_tokenizer`)
deepdoc/__init__.py → TableStructureRecognizer → rag_tokenizer → `infinity.rag_tokenizer` の連鎖。  
**対処**: pip install infinity-sdk==0.7.0 で解決。

---

## 2. 実行環境まとめ

| 項目 | 値 |
|------|----|
| venv パス | `/mnt/data/eval-driven-llm/.venv-deepdoc` |
| Python | 3.12.3 |
| DeepDoc ソース | `/mnt/data/ragflow-deepdoc/deepdoc` |
| 実行コマンドプレフィックス | `PYTHONPATH=/mnt/data/ragflow-deepdoc .venv-deepdoc/bin/python ragflow-deepdoc/deepdoc/vision/t_*.py` |
| OCR 実行時の追加フラグ | `CUDA_VISIBLE_DEVICES=""` (CPU 強制) |
| HF モデルキャッシュ | `~/.cache/huggingface/hub/models--InfiniFlow--deepdoc` |
| モデル初回ダウンロード時間 | ~33s (10 ファイル, det.onnx + rec.onnx + layout*.onnx + tsr.onnx 等) |

---

## 3. 解析速度 (CPU, モデルキャッシュ済み)

| タスク | サンプル | ページ数 | 実時間 | /ページ |
|--------|----------|----------|--------|---------|
| Layout 認識 | 01_nta (確定申告書) | 4p | 4.3s | ~1.1s |
| Layout 認識 | 03_stat_kakei (家計調査) | 24p | 14.0s | ~0.6s |
| TSR | 01_nta (確定申告書) | 4p | 17.9s | ~4.5s |
| OCR | 01_nta (確定申告書) | 4p | 15.5s | ~3.9s |
| OCR | 03_stat_kakei (家計調査) | 24p | 79.9s | ~3.3s |

**判定**: CPU で 3–4s/page は実用的に許容範囲。 Phase 2 でのバッチ処理 (1000 ページ規模) では GPU 化を推奨。  
GPU 化の優先度: OCR > TSR > Layout (コスト順)。

---

## 4. 日文文書への初期観察

### 4.1 文字認識 (OCR)

| 現象 | 例 | 重大度 |
|------|-----|--------|
| 令 → 今 の誤認識 (一貫して発生) | `今和6年分` (正: `令和6年分`) | 中 — 年号が毎回化ける |
| 所 → 覆 の誤認識 | `覆得及` (正: `所得及`) | 高 — キーワード破壊 |
| 複雑な罫線内のテキスト分割ミス | セル内容が断片化して複数行に | 低 — 後処理で結合可能 |
| ルビ・小字の脱落 | 注釈が取れない場合あり | 低 |

**全体**: 90% 前後の文字は読めるが「令和」「所得」など頻出法律用語が化ける問題は下流 RAG の信頼性に直結。 文字辞書の日本語ファインチューンまたは後処理辞書が要る。

### 4.2 レイアウト認識

- 確定申告書 (01_nta): 複雑な格子状フォームを概ね「表」として検出。  
  ただしフォームの小区画 (個人番号欄など) を複数の独立ブロックとして分割することがある。
- 家計調査 (03_stat_kakei): テキストブロックと統計表の境界を概ね正しく識別。  
  多段組みの一部は隣接カラムが結合されてしまうケースがある。
- 出力フォーマット: **JPEG 画像のみ** (バウンディングボックスを描画済み)。  
  JSON/テキスト形式のメタデータは `t_recognizer.py` からは出力されない。  
  → Phase 2 で eval pipeline に接続するには `PdfParser` Python API を直接使う必要がある。

### 4.3 表構造認識 (TSR)

- 確定申告書ページ 0 で 174 セルを抽出。HTML テーブルとして出力。
- セル数は妥当だがセル内テキストは OCR 精度に依存するため、`今和` `覆得` などの誤りがそのまま伝播する。
- 空セルや区分線専用行の扱いは曖昧 (セルに "-" や空白が入る)。
- テーブル HTML は `construct_table` が `rag_tokenizer.tokenize()` で blockType を決定するが、日本語テキストはほぼ英語 tokenizer を素通りするだけで有意な分類は得られていない。

### 4.4 出力フォーマット構造 (Phase 2 接続準備)

```
raw_output/
  layout/  ← JPG (検出結果を描画した画像)  ← eval には使えない
  tsr/     ← JPG + HTML (セル内容をテーブルに整形)  ← 表比較には使える
  ocr/     ← JPG + .txt (全テキスト一行ずつ)  ← チャンク化の素材
```

`PdfParser` を直接 import して使えば bounding boxes + text chunks + tables のメタデータ付き構造体が返る (raw_output のパース不要)。Phase 2 はこの API を使う。

---

## 5. Phase 2 へのアクション候補

1. **文字認識の後処理辞書**: 令/今、所/覆など頻出誤認識パターンの修正マッピング (日本語法律用語辞書との照合)。
2. **GPU 化**: OCR が律速 (~3.3s/page CPU)。RTX 5090 で 10–30× の高速化が見込める。`DEVICE=gpu` で ONNX GPU プロバイダーに切り替えが必要。
3. **PdfParser API での直接統合**: `t_*.py` ではなく `from deepdoc.parser import PdfParser` を使い、チャンク + 位置メタデータを取得してパイプラインに流す。
4. **横評価 (v2)**: MinerU / PaddleOCR との比較は v2 のスコープ。今は DeepDoc 単独で充分。
5. **スキャン文書 OCR**: sample 05 (国立公文書館) は本 Phase では未実行。Phase 2 で測定。

---

## 6. 検証チェックリスト (範囲外の確認)

- [x] `t_ocr.py --help` 動作: ✅
- [x] `t_recognizer.py --help` 動作: ✅
- [x] layout モード (sample 01, 4p): ✅ — 4 JPG 出力
- [x] layout モード (sample 03, 24p): ✅ — 24 JPG 出力
- [x] tsr モード (sample 01, 4p): ✅ — 4 HTML + 4 JPG 出力
- [x] OCR (sample 01, 4p): ✅ — 4 .txt + 4 JPG 出力
- [x] OCR (sample 03, 24p): ✅ — 24 .txt + 24 JPG 出力
- [x] `samples/SOURCES.md` に全来源記録: ✅
- [x] 顧客データなし: ✅ (全て政府公開PDF)
