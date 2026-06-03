"""Build a tiny FROZEN toy golden set + corpus to validate the eval loop.

This is plumbing validation only — NOT a model benchmark. Run:
    python scripts/build_toy_golden.py
"""
import json
from pathlib import Path

from elv.eval.ids import content_hash, stable_doc_id

# (title, text) passages on distinct topics
PASSAGES = [
    ("vLLM", "vLLMはPagedAttentionにより大規模言語モデルの推論スループットを高める推論サーバである。"),
    ("RAG", "RAGは検索で取得した文書を文脈として与え、言語モデルの回答精度を高める手法である。"),
    ("リランキング", "リランキングは一次検索で得た候補をクロスエンコーダで並べ替え、上位の適合率を上げる。"),
    ("オンプレミス", "オンプレミス運用ではモデルとデータが社内に留まり、データが外部に出ない利点がある。"),
    ("量子化", "量子化はモデルの重みを低ビットで表現し、必要なGPUメモリを削減して単一GPUでの実行を可能にする。"),
    ("eval", "評価ハーネスは凍結したゴールデンセットに対し再現可能な検索指標を算出する基盤である。"),
    ("Faiss", "Faissは大規模ベクトルの近傍探索を高速に行うライブラリで内積やL2距離に対応する。"),
    ("Qdrant", "Qdrantはメタデータでフィルタ可能なベクトルDBで、ユーザ権限に応じた検索の絞り込みに使える。"),
    ("ruri", "ruri-v3は日本語検索でJMTEB上位の埋め込みモデルで、ローカルで自前ホストできる。"),
    ("天気", "明日の関東地方は晴れのち曇りで、最高気温は二十度前後の見込みである。"),
    ("料理", "出汁は昆布と鰹節からとると、味噌汁や煮物の風味が大きく向上する。"),
    ("野球", "九回裏の逆転打でホームチームが勝利し、観客は大いに沸いた。"),
    # distractors: heavy surface overlap with some queries but off-topic. A
    # count-based dense cosine over-rewards repeated bigrams; set-based Jaccard
    # reranking handles them differently — any flip below is a real difference
    # between the two scorers, not a hand-tuned result.
    ("GPUゲーム", "最新GPUはゲームのモデル描画を高速化し、単一GPUでも大きな画面を滑らかに動かせる。"),
    ("検索広告", "検索連動型広告は検索クエリに応じて広告を出し、適合率より収益を上げる方法が問われる。"),
    ("データ移行", "データを社外のクラウドに出す移行作業では、停止時間を短くする方式の検討が必要である。"),
    ("DB入門", "ベクトルとは限らない一般的なデータベースは、ユーザごとに権限を制限できる仕組みを持つ。"),
]

# query -> list of (title) that are relevant
QUERIES = {
    "q1": ("単一GPUで大きなモデルを動かすにはどうすればよいか", ["量子化", "vLLM"]),
    "q2": ("検索の上位適合率を上げる方法は", ["リランキング"]),
    "q3": ("データを社外に出さずにLLMを使う方式は", ["オンプレミス"]),
    "q4": ("ユーザごとに検索範囲を制限できるベクトルDBは", ["Qdrant"]),
    "q5": ("日本語検索に強い埋め込みモデルは", ["ruri"]),
    "q6": ("評価で再現性を担保する基盤は何か", ["eval"]),
}

root = Path(__file__).resolve().parents[1]
gdir = root / "data" / "golden" / "_toy" / "v0"
cdir = root / "data" / "corpus" / "_toy"
gdir.mkdir(parents=True, exist_ok=True)
cdir.mkdir(parents=True, exist_ok=True)

title_to_id, corpus = {}, []
for title, text in PASSAGES:
    doc_id = stable_doc_id(title, text)
    title_to_id[title] = doc_id
    corpus.append({"doc_id": doc_id, "title": title, "text": text})

(cdir / "passages.jsonl").write_text(
    "\n".join(json.dumps(c, ensure_ascii=False) for c in corpus) + "\n", "utf-8")

with (gdir / "queries.jsonl").open("w", encoding="utf-8") as fh:
    for qid, (query, rel_titles) in QUERIES.items():
        fh.write(json.dumps(
            {"id": qid, "query": query,
             "relevant_doc_ids": [title_to_id[t] for t in rel_titles]},
            ensure_ascii=False) + "\n")

manifest = {
    "name": "_toy", "version": "v0",
    "source": "synthetic toy set — plumbing validation only, NOT a benchmark",
    "corpus_path": str(cdir / "passages.jsonl"),
    "n_queries": len(QUERIES), "n_passages": len(corpus),
    "doc_id_hashes": {c["doc_id"]: content_hash(c["title"], c["text"]) for c in corpus},
}
(gdir / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
print(f"froze toy golden set: {gdir}  ({len(QUERIES)} q, {len(corpus)} passages)")
