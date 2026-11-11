import functools
from pathlib import Path
from runtime.common import chunk_text, device, guarded, model_path, page_header


@functools.lru_cache(maxsize=1)
def model_and_tokenizer():
    from transformers import AutoModel, AutoTokenizer
    directory = model_path()
    return AutoModel.from_pretrained(directory).to(device()).eval(), AutoTokenizer.from_pretrained(directory)


def embeddings(texts):
    import torch
    import torch.nn.functional as F
    model, tokenizer = model_and_tokenizer()
    results = []
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            encoded = tokenizer(texts[start:start+16], max_length=512, padding=True, truncation=True, return_tensors="pt").to(device())
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).bool()
            pooled = hidden.masked_fill(~mask, 0).sum(1) / encoded["attention_mask"].sum(1).unsqueeze(-1)
            results.append(F.normalize(pooled, p=2, dim=1).cpu())
    return torch.cat(results)


@guarded
def search(files, query, count):
    if not files or not query.strip():
        raise ValueError("请上传文本文件并输入问题")
    if len(files) > 10:
        raise ValueError("单次最多10份文档")
    corpus = []
    for filename in files:
        path = Path(filename)
        if path.suffix.lower() not in (".txt", ".md") or path.stat().st_size > 1024*1024:
            raise ValueError("仅支持每份不超过1MB的UTF-8 TXT/Markdown")
        text = path.read_text(encoding="utf-8-sig")
        corpus.extend((path.name, i+1, part) for i, part in enumerate(chunk_text(text)))
    if not corpus or len(corpus) > 500:
        raise ValueError("文档为空或超过500个段落，请减少文档内容")
    passage = embeddings(["passage: " + part for _, _, part in corpus])
    question = embeddings(["query: " + query[:2000]])
    scores = (passage @ question.T).squeeze(1)
    indices = scores.argsort(descending=True)[:min(int(count), len(corpus))].tolist()
    return [[corpus[i][0], corpus[i][1], round(float(scores[i]), 4), corpus[i][2]] for i in indices]


def build_ui():
    import gradio as gr
    with gr.Blocks(title="E5 文档检索") as ui:
        page_header()
        gr.Markdown("从文档里找出相关原文，支持中英文跨语言检索；结果是原文片段，不生成答案。")
        files = gr.File(file_count="multiple", file_types=[".txt", ".md"], type="filepath", label="上传UTF-8文档")
        query = gr.Textbox(label="想查找什么")
        count = gr.Slider(1, 10, value=3, step=1, label="返回段落数")
        button = gr.Button("查找相关原文", variant="primary")
        result = gr.Dataframe(headers=["文件", "段落", "相似度", "原文"], datatype=["str", "number", "number", "str"], wrap=True, label="检索结果")
        button.click(search, [files, query, count], result, api_name="search")
    return ui
