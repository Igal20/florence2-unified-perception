# Unified Sports Perception — the recipe

> One small VLM. One forward pass. Detection, OCR, team affiliation and a scene caption — all natively associated, all in one structured token sequence.

This repository is the **methodology** behind the IMVC 2026 talk *"Unified Sports Perception"* by Igal Dmitriev and Ofir Liba (WSC Sports). It is **not a model release** and **not a code release** — it is the recipe.

**Why a recipe and not code?** Because the value isn't in a particular weights file. The value is in a small set of decisions: which tokens to add, how to serialize an annotation into a token sequence, what to freeze and what to train in two surgical stages, which tokens to weight in the loss. Those decisions are domain-independent — once you internalise them, you can rebuild the pipeline on *your* data, with *your* labels, with whichever LLM-coding assistant you prefer.

The talk's promise was a recipe you can hand to your favourite LLM. This is that recipe.

📖 **Companion blog post — the token-vocabulary surgery in detail:**
[Expanding Florence-2's Vocabulary — An Advanced Guide to Adding Custom Tokens During Fine-Tuning](https://medium.com/@ygal20/expanding-florence-2s-vocabulary-an-advanced-guide-to-adding-custom-tokens-during-fine-tuning-138fab660b64)

---

## How to use this recipe

You have three options:

1. **Read it yourself**, file by file, and implement what makes sense. The five documents are short and ordered.
2. **Hand it to an LLM** ("read all the files under `docs/` and scaffold a training pipeline that follows this recipe on my dataset"). The docs are written to be unambiguous for a coding assistant.
3. **Use it as a reference** while watching the talk recording — the chapters mirror the slide order.

Whichever path, work through the docs in this order:

| # | Doc | What it answers |
|---|---|---|
| 1 | [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md) | What does one annotation look like? What grammar string does the model learn to emit? |
| 2 | [`docs/TOKENS.md`](docs/TOKENS.md) | Which custom tokens to add, and *exactly* how to register them in Florence-2's tokenizer (this is where most teams trip). Pairs with the Medium post. |
| 3 | [`docs/TWO_STAGE_TRAINING.md`](docs/TWO_STAGE_TRAINING.md) | What's frozen, what's trainable, and which loss to use in Stage 1 vs Stage 2. |
| 4 | [`docs/INFERENCE.md`](docs/INFERENCE.md) | How to generate, parse, and visualise the structured output. |
| 5 | [`docs/ADAPT_TO_YOUR_DOMAIN.md`](docs/ADAPT_TO_YOUR_DOMAIN.md) | How to port the recipe from sports to retail / medical / manufacturing / etc. |

---

## What the model produces

After training, one image goes in. One forward pass later, you get a single token sequence that looks like this:

```
<MULTIMODAL_VISUAL_CAPTION><stype>In-Game<nath>3
  <player_1><bbox><loc_412><loc_205><loc_589><loc_734></bbox><team>A
    <ocr>23<loc_488><loc_412><loc_521><loc_412><loc_521><loc_456><loc_488><loc_456>
  </player_1>
  <player_2><bbox><loc_152><loc_198><loc_312><loc_701></bbox><team>B
    <ocr>7<loc_212><loc_390><loc_239><loc_390><loc_239><loc_430><loc_212><loc_430>
  </player_2>
  <player_3><bbox><loc_720><loc_215><loc_870><loc_690></bbox><team>A</player_3>
<gdesc>Three basketball players competing for a rebound under the basket during an NBA game.</s>
```

Detection, OCR (jersey numbers with polygons), team affiliation (`<team>A`/`<team>B`), entity association, and a natural-language description — all in one structured output, all bound to each player by construction. Parsing is a handful of regex passes.

The leading `<MULTIMODAL_VISUAL_CAPTION>` is the **input task prompt** that activates this behaviour (registered as a custom Florence-2 task prompt — see [`docs/TOKENS.md`](docs/TOKENS.md#the-task-prompt-token)). The decoder itself only generates the portion from `<stype>` onwards; everything before that is the input you pass to the processor.

That single string is the heart of the recipe. The entire training pipeline exists to teach a Florence-2 decoder to emit it correctly, given an image.

---

## Why the approach works (one paragraph)

A classical sports-perception cascade runs four models (detector → ReID → OCR → captioner) and stitches them together with heuristics (IoU thresholds, association rules, error-prone parsers). A hosted VLM API works in a notebook but breaks on cost and latency at production scale. **Both can be replaced by a single fine-tuned Florence-2 decoder** that learns the hierarchical grammar above. The trick is to teach it the grammar separately from teaching it where to look — that's the two-stage curriculum the docs describe.

---

## What this repo gives you and what it doesn't

| Provided | Not provided |
|---|---|
| The hierarchical token grammar | Pre-trained weights |
| The two-stage training methodology (freezing strategy, losses, hyper-parameters) | Production training code |
| The custom-token registration recipe (cross-linked to the Medium post) | A working dataset |
| Per-stage hyper-parameter starting points | A reference implementation you can `pip install` |
| A worked example for adapting to a non-sports domain | A hosted inference endpoint |

The expectation is that you bring your own annotated data (≈5–20K images is enough), implement the recipe with the LLM of your choice, and own the resulting model end-to-end. We get to share what worked; you get the IP and the data sovereignty.

---

## Citations & links

* 📖 **Medium — adding custom tokens:** <https://medium.com/@ygal20/expanding-florence-2s-vocabulary-an-advanced-guide-to-adding-custom-tokens-during-fine-tuning-138fab660b64>
* 🤗 **Florence-2-large:** <https://huggingface.co/microsoft/Florence-2-large>
* 📄 **Florence-2 paper:** Xiao *et al.*, *Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks*, CVPR 2024.
* 🎤 **IMVC 2026 talk:** *Unified Sports Perception* — Igal Dmitriev, Ofir Liba.

---

## Get in touch

Tried this on your data? Got it working on a non-sports domain? Stuck on Stage 2? Ping Igal on LinkedIn — we genuinely want to know which multi-task perception problem this pattern unlocks for you.
