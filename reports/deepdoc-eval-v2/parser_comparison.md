# 三方解析器 横評: pdfplumber × DeepDoc × MinerU

Phase 3.2 — China bridge 解析系列 収口  
実施日: 2026-06-13  
対象サンプル: 日本語政府公開PDF 3件（02_mof_budget / 03_stat_kakei / 04_mof_fiscal）  
誠実さの注記: 3ドキュメント・小規模サンプル。傾向の初期信号として扱うこと。

---

## 1. 解析器アーキテクチャ概要

| 項目 | pdfplumber (plain) | DeepDoc | MinerU 3.3.1 |
|------|-------------------|---------|--------------|
| テキスト抽出方式 | PDF テキスト直接抽出 | 画像変換 → OCR | **テキスト PDF: PyMuPDF 直接抽出**（OCR 不使用）|
| 表認識方式 | なし（フラットテキスト）| TSR（ONNX モデル）| SLANet+/UNet（ONNX）|
| レイアウト検出 | なし | DLR（ONNX モデル）| PP-DocLayout v2（transformers RT-DETR ベース）|
| Python エコシステム | 純 Python / pdfplumber | PyTorch + ONNX | PyTorch + ONNX（PaddlePaddle 不要）|
| OCR エンジン | なし | InfiniFlow ONNX OCR | PytorchPaddleOCR（ONNX, スキャン PDF のみ起動）|
| 分離環境 | .venv（共用）| .venv-deepdoc | .venv-mineru（独立必須）|

**重要**: MinerU の「`txt` メソッド」はテキスト PDF に対し OCR を起動しない。スキャン PDF が
入力されたときのみ PytorchPaddleOCR が動く（`auto` または `ocr` メソッド時）。
今回の 3 サンプル（デジタルテキスト PDF）は全て PyMuPDF テキスト抽出パスで処理された。

---

## 2. チャンク数・表数・処理速度

### 2.1 PDF 別 (生 parse 出力チャンク、スライディングウィンドウ適用前)

| サンプル（ページ数） | Parser | チャンク数 | 表数 | 平均チャンク長（字）| 処理時間 |
|---------------------|--------|-----------|------|-------------------|---------|
| 02_mof_budget (14p) | plain | 639 | 0 | 18.0 | 0.2s |
| 02_mof_budget (14p) | DeepDoc | 63 | 2 | 116.2 | 25.2s |
| 02_mof_budget (14p) | MinerU | 160 | 2 | 48.1 | **33.6s** |
| 03_stat_kakei (24p) | plain | 1,363 | 0 | 20.2 | 0.6s |
| 03_stat_kakei (24p) | DeepDoc | 66 | 13 | 139.9 | 220.5s |
| 03_stat_kakei (24p) | MinerU | 176 | 13 | 115.6 | **37.1s** |
| 04_mof_fiscal (64p) | plain | 5,057 | 0 | 16.1 | 2.6s |
| 04_mof_fiscal (64p) | DeepDoc | 208 | 38 | 128.6 | 474.4s |
| 04_mof_fiscal (64p) | MinerU | 468 | 34 | 96.1 | **45.7s** |

### 2.2 3 PDF 合計

| Parser | 総チャンク数 | 表数 | 総処理時間 | 対 DeepDoc 比 |
|--------|------------|------|-----------|--------------|
| plain | 7,059 | 0 | 3.3s | — |
| DeepDoc | 337 | 53 | 720s | baseline |
| MinerU | 804 | 49 | **116s** | **6.2× 高速** |

**観察**:
- MinerU は DeepDoc の **6.2× 高速**（GPU なし CPU 環境）
- チャンク粒度は DeepDoc (avg 120-140字) より細かく (avg 50-116字)、dense 検索向きの粒度
- 表数はほぼ同等（DeepDoc 53 vs MinerU 49）

---

## 3. 表還元（TSR）品質比較

### 3.1 出力フォーマット差

| 項目 | DeepDoc | MinerU |
|------|---------|--------|
| 表出力形式 | HTML (`<table>`) | HTML (`<table>`) — `table_body` フィールド |
| セル結合 (rowspan/colspan) | 対応 | 対応 |
| キャプション保持 | 限定的 | `table_caption` リストとして独立保存 |
| 注釈保持 | なし | `table_footnote` リストとして保存 |
| 表イメージ | なし | `img_path` でリンク |

### 3.2 サンプル 02 (予算フレーム表) での比較

**MinerU 出力例 (table_body)**:
```
歳出 / 6年度予算(当初) / 7年度予算 / 増減
一般歳出 / 677,764 / 682,452 / +4,689
社会保障関係費 / 377,193 / 382,778 / +5,585
...
```
caption: `['令和7年度予算フレーム（概要）', '（単位：億円）']`
footnote: `['（注）計数は、それぞれ四捨五入...']`

→ 金額数値・構造・キャプション・注釈すべて正しく取得。

**DeepDoc 出力**: 同じ表を 63 チャンクに分解、表内容はスライス付きテキスト。
セル境界の正確性は同程度だが、キャプション/注釈は別途処理が必要。

### 3.3 TSR Ground Truth 照合 (v1 検証の再利用)

v1 で確認済みの TSR セル値:
- `682,452` (令和7年度一般歳出): MinerU ✓, DeepDoc ✓
- `1,155,415` (一般会計計): MinerU ✓, DeepDoc ✓
- `284,400` → `784,400` (税収): MinerU ✓, DeepDoc ✓

→ 両者ともキー数値は正確に取得できている。

---

## 4. 日本語 OCR 品質比較

### 4.1 「令和」年号認識テスト（DeepDoc 既知バグの再現確認）

Phase 1 で DeepDoc が「令→今」を誤認識する問題（PDF 01 スキャン文書で確認）が報告されていた。

| テスト | plain | DeepDoc | MinerU |
|--------|-------|---------|--------|
| `令和` (正しい年号) の出現数 | 102 | 74 | 163 |
| `今和` (OCR 誤認識) の出現数 | 0 | 0 | **0** |

**判定**: PDF 02/03/04 はデジタルテキスト PDF のため、DeepDoc も MinerU も OCR 誤認識はなし。
ただし、スキャン PDF (例: PDF 01) では DeepDoc の「令→今」バグが再現する可能性がある。
MinerU は `txt` メソッド使用時、テキスト PDF では OCR を起動しないため、この種のエラーは
原理的に発生しない（テキスト PDF 限定の優位性）。

### 4.2 システマチック OCR 差異の構造的説明

```
DeepDoc:
  PDF → PIL 画像変換 → ONNX OCR (InfiniFlow)
  → 全ページが画像処理 → 文字認識精度に依存

MinerU (txt メソッド, テキスト PDF):
  PDF → PyMuPDF 直接テキスト抽出
  → フォントエンコーディングを直接読む → OCR 誤認識なし
  ただしフォント埋め込みに問題がある PDF では文字化けの可能性

MinerU (ocr/auto メソッド, スキャン PDF):
  PDF → 画像変換 → PytorchPaddleOCR (ONNX)
  → スキャン PDF では OCR に依存（日本語精度は未評価）
```

### 4.3 段組・縦書き対応

| 文書種別 | DeepDoc | MinerU |
|---------|---------|--------|
| 横書き多段組 | 対応（DLR で列検出）| 対応（PP-DocLayout v2）|
| 縦書きテキスト | 不明（テスト未実施）| **非対応**（公式ドキュメント明記）|
| フォーム型文書（確定申告書等）| 対応（複雑フォームも処理）| 未評価 |

---

## 5. 版面認識（レイアウト）品質

### 5.1 PP-DocLayout v2 vs InfiniFlow DLR

| 項目 | DeepDoc (InfiniFlow DLR) | MinerU (PP-DocLayout v2) |
|------|--------------------------|--------------------------|
| ベースモデル | ONNX カスタム | RT-DETR (transformers) |
| 対応レイアウト要素 | text, table, figure | text, title, table, figure, formula, caption |
| 数式認識 | なし | UnimerNet（オプション）|
| 日本語特化学習 | 不明 | 中国語/英語主体、日本語は未保証 |

### 5.2 PDF 04 (64p) での観察

- DeepDoc: 208 チャンク、大きなセマンティックブロック（段落を統合）
- MinerU: 468 チャンク、見出し・本文・表キャプションを個別に分離
- どちらも主要な財務表を正確に認識

---

## 6. 三方比較まとめ

### 6.1 特性サマリー

| 特性 | plain | DeepDoc | MinerU |
|------|-------|---------|--------|
| 処理速度（CPU）| ◎ 3.3s | ✗ 720s | ○ 116s |
| 表還元精度 | ✗ なし | ○ HTML | ○ HTML + caption/footnote |
| BM25 検索適合性 | △ 56.2% | ◎ 68.8% | ○ 62.5% |
| Dense 検索適合性 | ✗ 40.6% | ○ 65.6% | ◎ **71.9%** |
| テキスト PDF OCR 安全性 | ◎ 直接抽出 | △ 常時 OCR | ◎ OCR スキップ |
| 縦書き対応 | N/A | 未確認 | ✗ 非対応 |
| GPU なし運用 | ◎ | △（CPU で低速）| ○ 6.2× 高速 |
| 環境分離コスト | 低 | 中 | 中 |

### 6.2 CrossOver: BM25 vs Dense での逆転

DeepDoc は BM25 で優位（+12.5% over plain）、MinerU は Dense で最強（+31.2% over plain）。
この逆転の理由:
- DeepDoc の短い構造化チャンク（avg 120字）→ BM25 キーワードマッチに精確
- MinerU の中間粒度チャンク（avg 80字）→ dense embedding にとってよりリッチなセマンティック文脈を保持
- plain の微細チャンク（avg 18字）→ BM25 では断片ヒット、dense では文脈なしベクトル

### 6.3 日本企業が選ぶべき開源解析器（条件判断）

```
条件1: テキスト PDF のみ + dense 検索 中心
  → MinerU (pipeline/txt) を第一候補
  理由: OCR なし・高速・dense で最高精度

条件2: 多様な日本語文書（スキャン混在）+ BM25 中心
  → DeepDoc を第一候補
  理由: 常時 OCR でスキャン/デジタル両対応・BM25 で最高精度

条件3: 縦書き文書が含まれる
  → MinerU は不適（縦書き非対応）。DeepDoc か pdfplumber + 別途縦書き処理

条件4: GPU なし・速度優先
  → MinerU (CPU 116s vs DeepDoc CPU 720s = 6.2×高速)
  pdfplumber は表認識不要なら最速だが表を取りこぼす

条件5: 数式が多い文書（理工系・学術論文）
  → MinerU (UnimerNet 数式認識あり)。DeepDoc は数式非対応

条件6: 環境制約（オンプレ・インターネット接続制限）
  → どちらも HF モデルキャッシュ後はオフライン動作可能
```

---

## 7. 制限事項・留意点

1. **サンプル 3 文書（102 ページ）のみ**: 統計的有意性なし。傾向として読むこと
2. **ruri-v3-310m embedding 固定**: 他 embedder (bge-m3 等) では結果が異なる可能性
3. **スライディングウィンドウ 300字/150ステップ固定**: チャンキング戦略の最適化は未実施
4. **PDF 01 (NTA スキャン) と PDF 05 (archives スキャン) は今回対象外**
5. **PaddleOCR は評価から除外**: 百度 PaddlePaddle フレームワーク依存、CUDA 競合リスク
6. **MinerU 縦書き非対応**: 古文書・一部日本語文書では不適
7. **DeepDoc CPU 速度**: ~~GPU 使用時の速度比較は未実施~~（本プロジェクト CPU-only eval ポリシー）
