# Model Selection Benchmark v2 — Summary

> All numbers from real runs on RTX 5090 32 GB. No placeholders.
> Judge: **qwen3:32b** (not a competitor — see DECISIONS.md ADR for rationale).
> Cross-validation judge: **gemma4:31b** on 25 item subset.

## Quality / Latency / Resource (three-dimension table)

| Model | Params | Quant | Faithfulness | Hit Rate | n | Judge | Cold TPS | Warm TPS | Cold Wall(s) | VRAM Δ(MiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| elyza-jp-8b | 8.0B | Q4_K_M | 0.7788 | 0.4000 | 45 | qwen3:32b | 157.34 | 219.63 | 1.4 | 6347 |
| gemma4-31b | 31.3B | Q4_K_M | 0.7971 | 0.6222 | 45 | qwen3:32b | 62.75 | 60.16 | 3.17 | 26042 |
| nemotron-nano-9b-jp | 8.9B | Q4_K_M | 0.6388 | 0.6222 | 45 | qwen3:32b | 179.26 | 157.54 | 1.92 | 11362 |
| swallow-8b | 8.0B | Q8_0 | 0.7090 | 0.5333 | 45 | qwen3:32b | 169.33 | 147.97 | 2.92 | 9549 |

> faithfulness = fraction of answer claims entailed by provided context (judge-based, 1 run).
> hit_rate = fraction of items where answer matches reference answer (judge-based).

## Judge Agreement (cross-validation)

- Primary judge: **qwen3:32b**
- Cross judge:   **gemma4:31b** (also a competitor; cross-judges **only to validate primary judge reliability**, not to score itself)
- Subset size:   25 items (random seed 42)
- Hit agreement rate: **0.96**
- Cohen's κ (hit): **0.9196**
- Faithfulness agreement rate (|Δ|<0.2): **0.68**
- Disagreements: 1 items

### Disagreement cases (primary_hit ≠ cross_hit)

| id | primary_hit | cross_hit | primary_faith | cross_faith |
|---|---|---|---|---|
| msv2-008 | 0.0 | 1.0 | 1.000 | 1.000 |

## Constraint → Recommended model (decision table)

| Constraint | Recommended | Rationale |
|---|---|---|
| VRAM ≤ 10 GB + best Japanese quality | **swallow-8b** | 8B models fit 5–10 GB; winner by hit_rate |
| VRAM ≤ 10 GB + fastest tokens/s | **elyza-jp-8b** | highest warm TPS among ≤10 GB models |
| VRAM ≤ 20 GB + highest quality | **gemma4-31b** | highest hit_rate across all models; 31B needs ~20 GB |
| Fastest cold start (lowest TTFT) | **elyza-jp-8b** | smallest cold_start_wall_s |
| Multilingual (JP + EN) required | **gemma4-31b** | natively multilingual; 8B JP fine-tunes may degrade EN |
| Apache 2.0 license required | **gemma4-31b** | Apache 2.0; verify 8B model licenses before production |

## Protocol integrity notes

- **qwen3:32b** judges all competitors. It is NOT a competitor (deployment/content layer separation — see DECISIONS.md).
- **gemma4:31b** cross-judges a 25-item subset to validate primary judge reliability. Its own main scores are judged by qwen3:32b, never itself.
- All latency and VRAM numbers from live Ollama runs on RTX 5090 32 GB.
- Golden set: 45 context-grounded QA items, neutral tech/Japanese knowledge, no customer data.
- Golden set source: golden_qa_v2.jsonl (committed to repo).

## Discriminability Analysis

Golden set: **45 items** (from `golden_qa_v2.jsonl`)

| Category | Count | % |
|---|---|---|
| All models correct (zero discrimination) | 13 | 29% |
| Partial discrimination (some models wrong) | 23 | 51% |
| All models wrong | 9 | 20% |

### Discriminating questions (partial correct)

| ID | Difficulty | #Correct/4 | Query (preview) |
|---|---|---|---|
| msv2-005 | ? | 3/4 | この料金表によると、会員が平日の午後7時に2時間ジムを利用した場合の料金はいくらですか? |
| msv2-011 | ? | 3/4 | このフロー図に従うと、エラーコード「E-03」が発生した場合、最終的にユーザーに表示されるメッセージは何ですか |
| msv2-013 | ? | 1/4 | このレポートによると、プロジェクトAとプロジェクトBを同時進行できない理由は何ですか?また、もしCチームが2名 |
| msv2-016 | ? | 1/4 | このメールの送信者Aは受信者Bに対してどのような立場（上位・同位・下位）にあると推定されますか?文中の根拠を3 |
| msv2-018 | ? | 3/4 | この仮想的な数学的定義に基づくと、集合S={2,3,4,6,8,9,12}において、「強偶数」に該当する要素を |
| msv2-020 | ? | 3/4 | この架空の交通規則によると、ドライバーXが時速65kmで走行中に歩行者用信号が点滅を始めた場合、Xはどう行動す |
| msv2-022 | ? | 2/4 | この文書に書かれた3つの事実命題から、論理的に導出できる結論を1つ答えてください。推論過程も示してください。 |
| msv2-023 | ? | 3/4 | この規約によると、ユーザーが「プレミアムプラン」から「ベーシックプラン」にダウングレードした場合、ストレージは |
| msv2-024 | ? | 1/4 | このドキュメントによると、「緊急メンテナンス」と「定期メンテナンス」でユーザーへの通知タイミングが異なる理由は |
| msv2-027 | ? | 1/4 | この架空の選挙規則によると、候補者Aは当選できますか?理由を規則の条番号を引用して答えてください。【状況】Aは |
| msv2-028 | ? | 2/4 | この架空の薬品データシートに基づき、患者X（体重75kg、腎機能障害あり：GFR 35 mL/min）に対する |
| msv2-029 | ? | 1/4 | この架空のネットワーク設定に基づいて、ホストAからホストCへの通信が許可されているか答えてください。また判断根 |
| msv2-032 | ? | 1/4 | この仮想的な暗号規則で「HELLO」を暗号化した場合の出力を答えてください。 |
| msv2-033 | ? | 1/4 | この架空の優先度ルールに従い、受信した5件のリクエスト（R1〜R5）を処理すべき順序に並べてください。 |
| msv2-035 | ? | 1/4 | この複雑な配送規則に従い、注文Xの送料を計算してください。【注文X】重量2.5kg、サイズ60cm（3辺合計） |
| msv2-036 | ? | 3/4 | この架空の判定システムの説明に従い、入力値が「温度=38℃、湿度=80%、風速=3m/s」のときの最終判定を求 |
| msv2-037 | ? | 3/4 | この文書において、以下の3つの主張のうち文書の内容から「真」と言えるものと「偽」と言えるものをそれぞれ判定し、 |
| msv2-038 | ? | 3/4 | この架空の公共交通乗り継ぎ規則に基づき、乗客が地点Pから地点Sまで最も少ない乗り換え回数で行く経路と、その総所 |
| msv2-039 | ? | 2/4 | 以下の架空のSQLクエリは、どのようなデータを返しますか？日本語で説明してください。また、結果に「Tanaka |
| msv2-040 | ? | 3/4 | この架空の会計規則による「のれん」の計上額を計算してください。また、その後5年間の年次償却額も示してください。 |
| msv2-042 | ? | 1/4 | この架空の給付金制度によると、世帯Zは申請できますか？また、申請できる場合の給付額（月額）を計算してください。 |
| msv2-043 | ? | 1/4 | この架空の評価基準に従い、応募者P（得点詳細：筆記試験72点、実技試験88点、面接評価B、勤務年数4年）が「優 |
| msv2-045 | ? | 3/4 | この架空の気象予測モデルに従い、翌日の「積雪リスク」を「高/中/低/なし」で判定してください。【入力データ】気 |

## 区分度説明 — v1の失敗とv2の修正（方法論透明性のための記録）

### v1 での問題：区分度ゼロ

v1 の golden set（20問）は中立的な知識問題で構成されていた（絶対零度、富士山の標高、光速など）。
これらは LLM の学習データに広く含まれる事実であり、**文脈（context）を読まなくても正解できる**。

結果：

| 指標 | v1 の値 |
|---|---|
| 全モデル正解（区分度ゼロ） | 18/20 問（90%） |
| 部分正解（区分度あり） | 2/20 問（10%） |
| Cohen's κ（cross-val） | **1.0**（判定が同じなので一致率は自明） |
| hit_rate 最高 vs 最低 | 1.00 vs 0.90（差が小さすぎる） |

κ=1.0 は「クロスバリデーション判定が完全に一致した」ではなく、
**「全問正解なので何を judge しても同じ答えになった」という零方差の自明解**である。
この状態では 8B モデルと 31B モデルの差を計測することは不可能。

### v2 での修正：難度設計による区分度の回復

問題の根本原因は「モデルが context を読まずに知識から答えられる」こと。
v2 の修正原則：

1. **架空のルール・仕様を context に書く**：モデルが学習データから答えられないよう、
   実在しない規則（架空の交通法、架空の薬品データ、架空の選挙規定）を文脈として与える。
2. **多段推論を要求する**：1ステップの事実検索では足りない問題設計（条件分岐の連鎖、
   依存関係グラフの追跡、再帰的評価）。
3. **完全性を採点条件にする**（`expected_points`）：部分的な正解は不正解。
4. **日本語のニュアンス**：敬語から話者の立場を推定する、曖昧表現を文脈から解釈する。
5. **数値・論理計算の正確性**：複数の似た数値が context に混在し、正しい数値を選ぶ必要がある。

結果（v2, 45問）：

| 指標 | v2 の値 |
|---|---|
| 全モデル正解（区分度ゼロ） | 13/45 問（29%） |
| 部分正解（区分度あり） | 23/45 問（51%） |
| 全モデル不正解 | 9/45 問（20%） |
| Cohen's κ（cross-val） | **0.920**（実質的一致 — 零方差ではない） |
| hit_rate 最高 vs 最低 | 0.6222 vs 0.4000（差が可視化された） |

κ=0.920 は「主判定者（qwen3）と交差検証判定者（gemma4）が概ね一致しつつ
実際の不一致も存在する」ことを示す。これが区分度のある評価における正常な状態。

### v1・v2 の並存方針

v1 の結果は削除しない。修正前後の比較そのものが「評価設計の誠実さ」を示す。
区分度の低い評価結果を隠蔽せず、問題点を診断し、方法論を改善した過程を記録として残す。
これは ADR-0009（grounded-but-wrong メトリック artifact の訂正）と同じ透明性の原則に従う。
