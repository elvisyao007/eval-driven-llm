# EXPERIMENT_hybrid.md

> **状态:ARCHIVED(2026-06)。动机经 `reports/recall_metric_analysis.md` 证伪,未执行。见 DECISIONS.md ADR-0008。**

> 实验目标(诚实表述):在 JQaRA **固定 100 候选集**内,测稀疏-稠密互补性能否把正确文档排进 top-k(改善 context_recall),并将此结论与「第一阶段召回改善」明确切割。
>
> **本实验不能、也不会声称提升了第一阶段 recall。** JQaRA 是 reranking 数据集(每 query 100 候选),候选集由数据集固定给定,我们不做 full-corpus 检索。任何 context_recall 的变化都来自**重排序**,不是召回。这条约束必须一致地传播到 code / DECISIONS.md / README / blog。

---

## 0. 工作约定

- 工作目录:`/mnt/data/eval-driven-llm`
- 判定模型锁定:`gemma4:31b`(保持与现有 blog-02 baseline 可比;**generator 和 judge 不得同时更换**)
- 生成模型:`qwen3:32b`(与现有 baseline 一致)
- 两模型不能同时驻留 32GB GPU → 两段式:全生成 → VRAM 卸载 → 全判定
- 所有数字必须来自真实 eval,不编造
- 确定性优先:fusion 默认用 RRF(rank-only,天然确定性);判定/生成的随机性按现有 baseline 配置固定

---

## 1. 第 0 步(GATE):候选集 ceiling 先验 —— 必须先跑,结果决定实验是否继续

### 为什么这是 gate
context_recall=0.41 的瓶颈有两种可能,机制完全不同:

1. **正确文档在 100 候选内,但没排进 top-k** → 是排序问题 → hybrid 重排**能救** → 实验继续。
2. **正确文档根本不在 100 候选内** → 是数据集候选集本身的上限 → hybrid **救不了**(我们不做第一阶段检索)→ 实验应停止,改写「天花板」叙事。

### 计算什么
对每个评测集(生成集 100 / 检索全集 1667),计算:

```
ceiling = (正确文档落在给定 100 候选内的 query 数) / (总 query 数)
```

- 这是 context_recall 在本数据集设定下的**理论上限**。
- 同时输出 top-k 命中分布:正确文档当前排在第几位(rank 分布直方图)。这直接告诉我们「重排有多少空间」。

### 决策规则(写死)
- 若 `ceiling - 当前 context_recall(0.41)` 的差距 **≥ 0.15** → 排序空间大,实验全量继续。
- 若差距在 **0.05 ~ 0.15** → 空间有限,只跑最小闭环(A0/A1/A2/H1/H2),按结果再定。
- 若差距 **< 0.05** → hybrid 在本数据集上救不了 context_recall。**停止 hybrid 实验**,改产出「JQaRA 候选集天花板」诚实 blog,把精力转向真正的瓶颈(见 §6)。

> 产物:`reports/ceiling_check.md` —— ceiling 数字 + rank 分布图 + 上述决策结论。

---

## 2. 实验配置(8 行,ceiling gate 通过后才跑)

| ID | 描述 | 检索/重排 | 用途 |
|---|---|---|---|
| A0 | dense-only baseline | ruri-v3 稠密相似度排序 | 对照基线 |
| A1 | sparse-only (BM25, local IDF) | BM25 + SudachiPy,IDF 仅在 100 候选内统计 | 稀疏基线 |
| A2 | sparse-only (BM25, global IDF) | BM25 + SudachiPy,IDF 用全语料统计 | IDF ablation 卖点 |
| A3 | sparse-only (BM25, MeCab 分词) | BM25 + MeCab,其余同 A1 | 分词器 ablation 卖点 |
| H1 | hybrid RRF (dense + BM25-local) | RRF rank-only 融合 A0 + A1 | 主结果 |
| H2 | hybrid RRF (dense + BM25-global) | RRF 融合 A0 + A2 | IDF 对 hybrid 的影响 |
| H3 | hybrid weighted-norm | min-max 归一化加权融合;**α = 0.5 固定,事先声明,不在测试集上调** | 融合方法稳定性对比 |
| H4 | hybrid RRF + 现有 reranker | H1 输出再过现有 reranker | 重排叠加 |
| R0 | dense + 现有 reranker | A0 输出过现有 reranker | 对照(已有 reranker 单独效果) |

注:reranker 模型 ID 跑 R0/H4 前先 `ollama list` 确认精确 tag,记入 DECISIONS.md。  
注:RRF 的优势是 rank-only、免调参;H3 的 α 事先固定为 0.5 正是为了展示这个方法论卖点:
RRF 不需要在测试集上调权重,H3 的加权融合若要可比就必须做同样约束。

### 最小可行闭环(MVL)
先跑 **A0 / A1 / A2 / H1 / H2**(+ R0 管道校验,见 §3)。

**为什么 MVL 必须包含 A2/H2、不能只用 A1/H1：**  
A1 使用 local IDF（IDF 仅在 100 候选内统计）。100 篇文档的 document frequency 是个位数，
`IDF = log((N − df + 0.5) / (df + 0.5))` 在这个规模下几乎无区分力，BM25 退化成近似 TF 加权。
若 MVL 的主 hybrid 信号（H1）建立在退化的 local IDF 上，可能得出「hybrid 无效」的假阴性，
而真实原因是统计工具坏了，不是方法无效。

因此：**H2（dense + BM25-global IDF）是主结果**；H1/A1 作为对照，专门用来  
证明 local IDF 在 reranking 场景下的退化——这本身是有价值的 ablation 卖点。

其余配置（A3/H3/H4/R0 的 full 跑）是 ablation/稳定性；确认 MVL 有信号后再补。

---

## 3. 度量与产物

**主力评测集：检索全集 1667 query（不是生成集 100 query）**  
hybrid/重排是确定性的，不需要生成模型。Step 0 已确认提升空间是结构性的
（oracle − dense@5_within_100 ≈ 0.19），H vs A0 的 delta 预计只有 0.01–0.03，
在 100 query 上与噪声无法区分。1667 query 才有足够的统计功效。
生成集 100 query 作为补充运行（与 blog-02 baseline 保持可比）。

**主指标（按重要性排序）：**
1. `recall@5 within 100` — 每配置 vs A0，1667 query，**主要 delta 基准**
2. `recall@10 within 100` — 同上
3. `P@1 within 100`
4. `context_recall@5`（生成集 100 query，与 blog-02 可比）

**显著性检验（针对每个 H/A 配置 vs A0）：**
- per-query recall@5 差值的 paired bootstrap（10000 次重采样，seed=42），报告 delta 的 95% CI
- 只有 95% CI 下界 > 0（置信区间不跨零）的 delta 才可在 blog/报告里下「hybrid 有效」的结论
- 若所有 H 的 delta 均不显著，如实报告：提升空间是结构性的（k=5 截断），不是排序问题

**R0 管道校验（必须先通过，才算管道正确）：**
- A0（dense-only within 100）必须复现 ceiling_check 的：
  `dense_recall@5_within_100 = 0.4224`（gen set）/ `0.4368`（retrieval set，±0.001）
- R0（dense + 现有 reranker within 100）必须复现 blog-01 的 **P@1 = 0.8440**（±0.001）
- 若任一数字对不上，说明 within-100 接法有 bug，先停下排查，不继续其他配置

**确定性校验：**
- RRF 跑两次，确认数字完全一致（rank-only 应零方差）
- BM25（含分词和 IDF）跑两次同上

**产物落点：**
- `reports/ceiling_check.md`（Step 0，已完成）
- `reports/hybrid_results.md`（配置对比表 + delta + 95% CI）
- `DECISIONS.md` 追加（ADR-0007）：tokenizer 选择、IDF 选择、fusion 方法、reranker model ID

---

## 4. 诚实约束清单(传播到所有产物)

- [ ] 不声称提升第一阶段 recall;明确写「固定候选集内的重排」
- [ ] ceiling 数字与 context_recall 并列展示,让读者看到天花板
- [ ] 若 hybrid 改善有限,如实写,不藏负结果
- [ ] ablation(local/global IDF、SudachiPy/MeCab、RRF/weighted)作为方法论严谨性展示,不是为了凑配置
- [ ] 只有 95% CI 不跨零的 delta 才在 blog 里下「hybrid 有效」的结论;不显著的结果如实写

---

## 5. Claude Code 执行顺序(scope-frozen)

1. **Step 0**:写 ceiling 计算脚本 → 跑生成集 100 + 检索全集 1667 → 产出 `reports/ceiling_check.md` → **停下来,等人看决策结论**。
2. **Step 1**(gate 通过后):实现 R0/A0/A1/A2/H1/H2（MVL 五配置 + R0 管道校验）→ 先跑 R0/A0 校验管道 → 再跑 A1/A2/H1/H2 → 主力跑检索全集 1667 query + 显著性检验 → 产出 MVL 结果。
3. **Step 2**:确认 MVL 有信号后,补 A3/H3/H4 + 生成集全量。
4. 每步更新 DECISIONS.md。

> Step 0 是硬 gate。不要在没看到 ceiling 决策结论前直接跳到 Step 1。

---

## 6. 若 ceiling gate 判定「救不了」时的转向(预案)
context_recall 真正瓶颈不在排序而在候选集本身 → 那 0.41 是 JQaRA 给定候选的属性,不是你系统的缺陷。此时:
- 产出诚实 blog:「为什么我放弃了 hybrid 攻 context_recall——一个 reranking 数据集教我的事」(反主流、有数据撑腰)。
- 把工程精力转回**生成层**真瓶颈:grounded-but-wrong 33/100。这才是 0.41 之外更值钱的问题。
