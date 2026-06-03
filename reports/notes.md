# Retrieval benchmark notes — JQaRA v0, 2026-06-04

## Run summary

| Metric | dense-only | dense+rerank | delta |
|---|---|---|---|
| P@1 | 0.8308 | 0.8440 | **+0.0132** |
| MRR | 0.8858 | 0.8927 | **+0.0069** |
| nDCG@5 | 0.7135 | 0.7051 | -0.0083 |
| nDCG@10 | 0.6870 | 0.6770 | **-0.0099** |
| Recall@5 | 0.4256 | 0.4158 | -0.0098 |
| Recall@10 | 0.5738 | 0.5634 | **-0.0104** |

Per-query P@1 flips: **133 fixed, 111 broke, 1423 unchanged.**  
Net +22 correct queries at P@1.

---

## FIXED by reranker — example 1: QA20CAPR-1024

**Query:** 代表曲に『古い日記』や『あの鐘を鳴らすのはあなた』がある、
人気番組『アッコにおまかせ!』の司会を務めるタレントは誰でしょう?
(Who hosts the variety show "Akko ni Omakase!" and has hit songs "Furui Nikki"
and "Ano Kane wo Narasu no wa Anata"?)

**Dense ranked #1 (wrong):**  
Title: アッコにおまかせ!  
Text snippet: …番組では地デジ化キャラクター…1972年の「第14回日本レコード大賞」で和田が…  
(A passage about the show's broadcasting history; "和田" appears but only in passing.)

**Rerank promoted to #1 (correct):**  
Title: アッコにおまかせ!  
Text snippet: …総合司会を務める和田アキ子の冠番組で、民放の日曜12…  
(Opens with "Wada Akiko's variety show, hosted by her".)

**Why the reranker fixed it:** Both passages are from the same article. The
cross-encoder scored the query "who is the host?" jointly against each passage
and promoted the one that names the host in its first clause. The dense embedder
ranked both passages similarly (same article, similar embedding) and put the
historical/broadcasting passage first — not wrong by embedding similarity, wrong
by answer presence.

---

## FIXED by reranker — example 2: QA20QBIK-1014

**Query:** 1970年代に流行した、ヒールの高いロングブーツのことを、ある都市の
若者の間から流行したことにちなんで何ブーツという?
(High-heeled long boots popular in the 1970s — named after a city where they
became trendy among youth — are called what kind of boots?)

**Dense ranked #1 (wrong):**  
Title: 長靴  
Text snippet: …1969年にアサヒゴムが「ハイレイン」のブランドで細め長めの婦人用ゴム長靴…  
(About rubber rain boots; mentions "ロングブーツ" surface term.)

**Rerank promoted to #1 (correct):**  
Title: ロンドンブーツ  
Text snippet: …1970年代に流行した、かかとが高いブーツのこと…  
(Direct definitional match: "high-heeled boots popular in the 1970s".)

**Why the reranker fixed it:** The dense model latched onto the shared surface
term "ロングブーツ" and retrieved a general rain-boot history passage. The
cross-encoder recognized the query is asking *what are they called?* and
promoted the passage that both defines the named footwear type AND matches all
semantic constraints (1970s, high heel, city origin).

---

## BROKE by reranker — example: QA20CAPR-1015

**Query:** 星条旗といえばアメリカの国旗ですが、太極旗といえばどこの国旗でしょう?
(The Stars and Stripes is America's flag — which country's flag is the Taegeukgi?)

**Dense ranked #1 (correct):**  
Title: 大韓民国の国旗  
Text snippet: …清国の外交官の馬建忠が太極旗を…  
(Passage explicitly about the Korean flag / Taegeukgi.)

**Rerank demoted to #1 (wrong):**  
Title: 花旗  
Text snippet: 花旗(かき)は、アメリカ合衆国の国旗(星条旗)を指す中国語由来の古い呼び方…  
(A passage about the Chinese name for the Stars and Stripes.)

**Why the reranker broke it:** The query uses a contrastive structure: "A is X,
but what country is B?" The cross-encoder scored the "花旗 / Stars and Stripes"
passage highly because it contains both 星条旗 and アメリカ, which both appear
prominently in the query. It over-weighted the query's contrastive premise over
its actual question target (太極旗). The dense model correctly focused on 太極旗
as the pivot term.

---

## Interpretation

The rerank trade-off is consistent with ADR-0002's design note:

> "adding sparse mainly lifts Recall@10, barely moves P@1, and can shuffle
>  mid-ranks enough to lower P@3."

Here the cross-encoder lifts P@1 (+0.0132) and MRR (+0.0069) at the cost of
recall@10 (-0.0104) — it promotes a single best answer but re-orders mid-ranks
in ways that occasionally push a relevant passage below the cutoff. For a
"single-answer QA" task this is the correct trade-off. For a multi-document
retrieval task (high recall needed), the trade-off would need to be revisited.
