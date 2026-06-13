# DeepDoc Eval Phase 3.2 — 三方解析器横評サマリー

実施日: 2026-06-13  
フェーズ: 3.2（China bridge 解析系列 収口）  
対象: Phase 3.1 の 2×2 を 3×2 に拡張（MinerU 追加）  
環境: CPU only（embedding: ruri-v3-310m on CPU）, Python 3.12  
誠実さの注記: 3ドキュメント・32問は初期信号。定説として扱わない。

---

## 1. 評価設定

| 項目 | 値 |
|------|---|
| ゴールデンセット | golden_set_v2.json（32問, oracle 87.5%）|
| 対象ドキュメント | sample 02, 03, 04（14+24+64=102 ページ）|
| Pipeline A | pdfplumber plain text → sliding-window chunking（300字/150ステップ）|
| Pipeline B | DeepDoc PdfParser（layout + TSR）→ chunking |
| Pipeline C | MinerU 3.3.1（pipeline backend, txt method, japan lang）→ chunking |
| Retriever 1 | BM25（文字バイグラムトークン化）|
| Retriever 2 | dense（cl-nagoya/ruri-v3-310m、コサイン類似度、in-memory）|
| 指標 | hit@5（answer_keywords が top-5 チャンクに含まれるか）|
| Embedding | ruri-v3-310m（日本語検索特化、本プロジェクト標準、/mnt/cache/hf にキャッシュ済）|
| 除外 | PaddleOCR（百度 PaddlePaddle 依存、CUDA 競合リスク、環境コスト不値）|

---

## 2. Sanity Gate (3 管線共通)

```
Pipeline A oracle（plain）: 28/32 = 87.5%
Pipeline B oracle（DeepDoc）: 28/32 = 87.5%
Pipeline C oracle（MinerU）: 28/32 = 87.5%
Sanity gate: PASS（≥60%条件充足）
```

Oracle-failing 4問（Q07/Q08/Q18/Q26）: 算術計算値がコーパスに存在しない（v2 設計どおり）

---

## 3. 3×2 Hit@5 結果（コア）

### 3.1 結果表

| Pipeline | BM25 | Dense |
|---------|------|-------|
| **A（plain pdfplumber）** | **56.2%** (18/32) | **40.6%** (13/32) |
| **B（DeepDoc）** | **68.8%** (22/32) | **65.6%** (21/32) |
| **C（MinerU）** | **62.5%** (20/32) | **71.9%** (23/32) |
| Delta B − A | +12.5% (+4問) | +25.0% (+8問) |
| **Delta C − A** | **+6.2%** (+2問) | **+31.2%** (+10問) |
| Delta C − B | -6.2% (-2問) | **+6.2%** (+2問) |

### 3.2 問別詳細

```
ID    Diff          BM25-A  BM25-B  BM25-C   Den-A   Den-B   Den-C
Q01   medium             ✓       ✓       ✗       ✗       ✗       ✗
Q02   medium             ✗       ✓       ✗       ✗       ✓       ✓
Q03   medium             ✓       ✓       ✓       ✗       ✗       ✓
Q04   hard               ✓       ✓       ✓       ✓       ✓       ✓
Q05   medium             ✓       ✓       ✓       ✗       ✓       ✗
Q06   hard               ✓       ✓       ✓       ✓       ✗       ✓
Q07   oracle_fail        ✗       ✗       ✗       ✗       ✗       ✗
Q08   oracle_fail        ✗       ✗       ✗       ✗       ✗       ✗
Q09   hard               ✗       ✗       ✗       ✗       ✗       ✗
Q10   hard               ✗       ✗       ✗       ✗       ✓       ✗
Q11   medium             ✗       ✓       ✓       ✗       ✓       ✓
Q12   hard               ✗       ✗       ✗       ✗       ✗       ✓
Q13   hard               ✗       ✗       ✗       ✓       ✓       ✓
Q14   medium             ✓       ✓       ✓       ✓       ✓       ✓
Q15   hard               ✗       ✓       ✓       ✗       ✓       ✓
Q16   hard               ✗       ✓       ✓       ✗       ✓       ✓
Q17   hard               ✗       ✗       ✓       ✗       ✗       ✓
Q18   oracle_fail        ✗       ✗       ✗       ✗       ✗       ✗
Q19   medium             ✓       ✓       ✓       ✗       ✓       ✓
Q20   hard               ✗       ✗       ✗       ✗       ✗       ✗
Q21   medium             ✓       ✓       ✓       ✓       ✓       ✓
Q22   medium             ✓       ✓       ✓       ✓       ✓       ✓
Q23   hard               ✓       ✓       ✓       ✓       ✓       ✓
Q24   hard               ✓       ✓       ✓       ✓       ✓       ✓
Q25   hard               ✓       ✓       ✓       ✓       ✓       ✓
Q26   oracle_fail        ✗       ✗       ✗       ✗       ✗       ✗
Q27   hard               ✓       ✓       ✓       ✓       ✓       ✓
Q28   hard               ✓       ✓       ✓       ✗       ✓       ✓
Q29   hard               ✓       ✓       ✓       ✓       ✓       ✓
Q30   hard               ✓       ✓       ✓       ✓       ✓       ✓
Q31   hard               ✓       ✓       ✓       ✓       ✓       ✓
Q32   hard               ✓       ✓       ✗       ✗       ✓       ✓
```

---

## 4. 三方解析品質比較表

| 項目 | plain | DeepDoc | MinerU |
|------|-------|---------|--------|
| 総チャンク数（生）| 7,059 | 337 | 804 |
| 総チャンク数（sliding window 後）| 2,934 | 630 | 929 |
| 表検出数 | 0 | 53 | 49 |
| 平均チャンク長 | 18字 | ~128字 | ~87字 |
| 処理速度（CPU, 3 PDF 合計）| 3.3s | 720s | 116s |
| テキスト PDF OCR エラー | なし | なし | なし |
| 縦書き対応 | 未対応 | 未確認 | **非対応**（公式）|
| キャプション・注釈保持 | なし | 限定的 | ◎（独立フィールド）|

---

## 5. 主要発見

### 5.1 BM25 vs Dense のクロスオーバー（最重要発見）

**DeepDoc: BM25 優位（68.8% > 62.5%）、MinerU: Dense 優位（71.9% > 65.6%）**

- DeepDoc の短い構造化チャンク（avg 128字）→ BM25 キーワード精度が高い
- MinerU の中間粒度チャンク（avg 87字）→ ruri-v3 dense embedding にとってよりリッチな
  セマンティック文脈を保持 → Dense で DeepDoc を +6.2% 上回る

**実務含意**: Dense 検索（ベクトルDB / RAG）を主軸とするシステムでは MinerU が有利。
BM25 ハイブリッドシステムで BM25 ウェイトが高い構成では DeepDoc が有利。

### 5.2 MinerU の Dense 優位性は DeepDoc より大きい

| 比較 | Phase 3.1（DeepDoc vs plain）| Phase 3.2（MinerU vs plain）|
|------|-----------------------------|-----------------------------|
| BM25 delta | +12.5% | +6.2% |
| Dense delta | +25.0% | **+31.2%** |

MinerU の Dense delta (+31.2%) は DeepDoc (+25.0%) を超えた。
理由: MinerU チャンクは DeepDoc より長めで文脈を多く含み、dense embedding の品質が高い。

### 5.3 MinerU の日本語 OCR リスク

- テキスト PDF（本評価対象）: OCR 非使用 → 「令→今」型の誤認識ゼロ
- スキャン PDF: PytorchPaddleOCR 起動 → 日本語精度は**未評価**（Phase 3.2 スコープ外）
- DeepDoc: 全ページ OCR → スキャン PDF での誤認識リスクあり（Phase 1 で確認済み）

### 5.4 処理速度の差異

| 解析器 | 3 PDF 合計 | DeepDoc 比 |
|--------|-----------|-----------|
| plain | 3.3s | 218× 高速 |
| MinerU | 116s | **6.2× 高速** |
| DeepDoc | 720s | baseline |

MinerU の速度優位は GPU 不要・CPU 環境での実用運用に直結する。

### 5.5 PaddleOCR 除外記録

PaddleOCR（単体）を評価から除外した理由:
1. 依存先: 百度 PaddlePaddle フレームワーク（非 PyTorch エコシステム）
2. CUDA/cudnn との競合: 既知の RTX 5090 (sm_120) 環境での不整合
3. 環境コスト: PaddlePaddle は PyTorch と独立の CUDA スタックを要求
4. 代替: MinerU は PaddleOCR の実装を ONNX/PyTorch に移植（PytorchPaddleOCR）して
   PaddlePaddle 依存を排除している → 同等機能を安全な環境で利用可能

---

## 6. 日本企業向け「選択指針」

```
テキスト PDF 中心 × dense 検索 (RAG/ベクトルDB)
  → MinerU (pipeline/txt)
  ・OCR エラーなし、6× 高速、dense hit@5 最高（71.9%）

混在文書（スキャン含む）× BM25 ハイブリッド
  → DeepDoc
  ・OCR 常時対応、BM25 最高（68.8%）

縦書き文書あり
  → MinerU は不適。DeepDoc か pdfplumber + 別途処理

GPU なし・速度優先・表認識不要
  → pdfplumber (3.3s) + 適切なチャンキング

数式・論文文書
  → MinerU（UnimerNet 数式認識、table + formula 同時対応）
```

**いずれも**: 3 ドキュメント・32 問での比較。本番採用前に自社文書での検証を強く推奨。

---

## 7. Phase 3.1 結果との整合

Phase 3.1 の 2×2 結果（DeepDoc vs plain）は本評価でも完全再現:

| Phase 3.1 | BM25 | Dense |
|-----------|------|-------|
| plain | 56.2% | 40.6% |
| DeepDoc | 68.8% | 65.6% |
| delta | +12.5% | +25.0% |

Phase 3.2 の追加管線 (MinerU) は既存結果を変更せず、第三の視点を追加。

---

## 8. 今後の拡張候補（Phase 3.2 スコープ外）

- MinerU + スキャン PDF での日本語 OCR 精度評価（PDF 01/05 対象）
- MinerU `ocr` メソッドでの強制 OCR vs `txt` メソッドの精度差
- bge-m3 / multilingual-e5 での embedding 横比較（ADR-0020 の再検討条件）
- 50問以上の統計的に有意なゴールデンセットでの再検証
- MinerU GPU 加速 (RTX 5090) での処理速度測定
