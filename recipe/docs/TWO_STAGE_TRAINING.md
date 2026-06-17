# The two-stage training curriculum

## Why two stages, not one

Single-shot fine-tuning Florence-2 on the hierarchical token sequence conflates two very different learning problems:

1. **Grammar** — what tokens come in what order. The model must learn that `<bbox>` is always followed by exactly 4 `<loc_*>` and a `</bbox>`, that `<ocr>` is always followed by text and then 8 `<loc_*>`, that `<player_N>` must close with `</player_N>`, etc.
2. **Grounding** — which pixels each token refers to. Bounding boxes must align with actual entities, OCR text must match jersey numbers, the scene class must match the visual context.

If both happen at once the gradients fight each other. The decoder invents grammar while the cross-attention is being asked to point at specific pixels, the vision encoder is pulled away from its pretrained representation in service of the grammar, convergence is slow, and on small datasets you simply never converge to a coherent output.

**The fix is to decouple.** Stage 1 teaches the grammar with the vision encoder fully frozen. Stage 2 teaches grounding via decoder-only LoRA, with the vision encoder still frozen — the decoder learns *where to look* in the unchanged visual features.

## Why two stages and not three

The full production pipeline (the one shown in the talk) adds **Stage 3**: surgical LoRA on the late DaViT vision stages (1/16 and 1/32 resolution) at a very low LR, to nudge the visual representation for the specific domain. We deliberately leave Stage 3 out of this recipe because:

* Florence-2's FLD-5B pretraining is already excellent on sports and general scenes. Stage 3 buys 2-4% accuracy at the cost of considerably more engineering and a real risk of corrupting the visual generalisation that lets the same weights work on unseen leagues.
* Two stages get you 90% of the quality with a fraction of the work.
* If you need Stage 3 later, the recipe accommodates it cleanly — it's another freeze-then-LoRA cycle, this time targeting attention projections inside the late vision blocks. Same pattern as Stage 2.

If you have a strong reason to drift the visual representation — radically different visual domain (medical microscopy, satellite imagery, etc.) — add Stage 3. Otherwise don't.

---

## Visual orientation — what's frozen, what's trained

Before the per-stage details, here is the whole story in one diagram. Both columns show the same Florence-2 architecture top-to-bottom; only the colour of each box changes between stages.

![Florence-2 two-stage fine-tuning diagram showing which components are frozen vs trainable in Stage 1 (Vocabulary Alignment) and Stage 2 (Decoder Refinement)](images/florence2_two_stage_freezing.png)

**The logic behind this freezing pattern** — three decisions explain almost everything:

1. **The vision encoder is *never* trainable in this recipe.** Florence-2's DaViT was pre-trained on FLD-5B (a vast visual corpus) and already separates entities, jersey-like patches, scene context, and text-bearing regions extremely well for natural images. Letting the vision encoder drift in the service of grammar learning (Stage 1) or grounding (Stage 2) is a recipe for catastrophic forgetting — the model gets better on your training distribution and worse on every league, lighting condition, or sport you haven't seen. Stage 3 of the production curriculum (deliberately omitted here) does adapt the late DaViT stages with surgical LoRA, but only when the visual domain is genuinely different from natural images.

2. **Stage 1 makes the new vocabulary part of the model.** The freshly-registered custom tokens (`<MULTIMODAL_VISUAL_CAPTION>`, `<stype>`, `<team>`, all the `<player_N>` pairs, etc.) start with random-ish initialisations — they have no semantic content. To learn good embeddings for them, the shared embedding matrix must be **trainable**, and because Florence-2 ties four pointers to that one matrix (`encoder.embed_tokens`, `decoder.embed_tokens`, `model.shared`, `lm_head`), training the shared matrix automatically trains the lm_head too. The full decoder is also trainable in Stage 1 because the decoder has to learn the *grammar* — what tokens come in what order — and that requires real plasticity, not just adapter-sized capacity. Only the encoder layers and layernorms are frozen, so the encoder doesn't drift while the decoder learns its new vocabulary.

3. **Stage 2 makes the new vocabulary visually grounded — cheaply.** After Stage 1 the model can emit syntactically valid output but the boxes/OCR may point at the wrong things. Stage 2 fixes this with tiny LoRA adapters on the decoder's attention projections — and *only* those projections, because cross-attention is the mechanism by which the decoder "looks at" the (frozen) visual features. Everything else is frozen so the Stage 1 vocabulary alignment is locked in. Only ~0.3% of parameters end up trainable. This stage is fast, hard to overfit, and the resulting LoRA can be merged back into the base linear weights at save time for zero-overhead inference.

The `freeze_*` helper names in the diagram (`freeze_vision_encoder`, `freeze_bart_encoder_layers`, `inject_decoder_lora`) are short utility functions you implement once and reuse across both stages — see each stage's *"How freezing actually works"* paragraph below for the exact pattern.

---

## Stage 1 — Vocabulary alignment

### Freezing policy

The notation "**frozen**" below means `param.requires_grad = False`. Florence-2 **ties four parameter pointers to one weight matrix** — `model.shared`, `encoder.embed_tokens`, `decoder.embed_tokens`, and `lm_head` all share the same underlying tensor — so the table treats each pointer as its own row to make the tie explicit.

| Component | Stage 1 |
|---|---|
| DaViT vision encoder (`model.vision_tower`) | **frozen** |
| BART encoder transformer layers (`encoder.layers`) | **frozen** |
| BART encoder layernorms (`layernorm_embedding`, final `layer_norm`) | **frozen** |
| BART encoder `embed_tokens` | **trainable** (tied) |
| Shared embedding (`model.language_model.model.shared`) | **trainable** — the rows for the new custom tokens need to learn |
| BART decoder `embed_tokens` | **trainable** (tied) |
| BART decoder layers (self-attn, cross-attn, FFN, layernorms) | **trainable** (full fine-tune) |
| `lm_head` | **trainable** (tied) |

**How freezing actually works in Stage 1.** Because the shared embedding is trainable and `lm_head` shares its weight tensor by tying, training one trains all four pointers. You do not have to chase pointers individually — you only need to freeze the things that are *not* tied to the shared matrix:

```python
freeze_vision_encoder(model)         # vision_tower.parameters().requires_grad = False
freeze_bart_encoder_layers(model)    # encoder.layers + layernorm_embedding + final layer_norm
                                     #   -- explicitly does NOT touch encoder.embed_tokens
# Decoder layers, shared embedding, decoder embed_tokens, lm_head: default requires_grad = True
```

The two helpers above are 3-5 lines each — implement them yourself (or have your LLM scaffold them from the freezing-policy table). The critical detail is that `freeze_bart_encoder_layers` walks `encoder.layers` and the two layernorms but **leaves `encoder.embed_tokens` untouched**, because that parameter is tied to `model.shared` and freezing it would also freeze the lm_head and the decoder's `embed_tokens` — defeating the whole point of Stage 1.

### Loss

Uniform cross-entropy. **Do not weight tokens in Stage 1** — biasing the loss before the model knows the grammar slows convergence and produces uneven token-level errors.

### Early stop

Every few hundred steps, generate on ~50 validation images and run cheap regex checks against the expected grammar (well-formed `<bbox>`, balanced `<player_N>...</player_N>`, `<nath>` count matches the number of player blocks, etc.). When ≥ 95% of generations *parse* — regardless of whether the content is right — Stage 1 is done. In practice this triggers within 1-2 epochs on a few thousand annotations.

### Hyper-parameters that matter

| Knob | Suggested | Why |
|---|---|---|
| Learning rate | `5e-5` | Larger LR destabilises the new token embeddings. |
| Warmup | 10% linear | Gives the freshly-added embedding rows time to find direction before the LR peaks. |
| Batch size | `16` (Florence-2-large, 24 GB) | Halve if you OOM on a smaller GPU. |
| Epochs | `3` (cap) | With early-stop, you almost always finish in 1-2. |
| Schema-compliance threshold | `0.95` | Higher = stricter early-stop = more epochs. |

### What "done" looks like

Schema-compliance metric climbing past 95% within 1-2 epochs. Validation loss converging around 0.3-0.6 (varies by dataset size). At that point the checkpoint is a clean Hugging Face folder (model + processor + the expanded tokenizer + Florence-2's custom `.py` files) — load it with a plain `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`.

---

## Stage 2 — Decoder refinement (LoRA + hierarchical loss)

### Freezing policy

Same notation as Stage 1 — "**frozen**" means `param.requires_grad = False`. Stage 2 freezes *everything* in the base model and only the freshly-injected LoRA `lora_A` / `lora_B` parameters end up trainable.

| Component | Stage 2 |
|---|---|
| DaViT vision encoder (`model.vision_tower`) | **frozen** |
| BART encoder (layers + layernorms + `embed_tokens`) | **frozen** |
| Shared embedding (`model.language_model.model.shared`) | **frozen** — locks the Stage 1 vocabulary alignment |
| BART decoder `embed_tokens` | **frozen** (tied) |
| BART decoder `embed_positions` | **frozen** |
| `lm_head` | **frozen** (tied) |
| BART decoder base weights (self-attn, cross-attn, FFN, layernorms) | **frozen** |
| LoRA on decoder self-attention (q_proj, v_proj, out_proj) | **trainable** |
| LoRA on decoder cross-attention (q_proj, v_proj, out_proj) | **trainable** |

Only ~0.3% of all parameters are trainable. Forward / backward is correspondingly cheap.

**How freezing actually works in Stage 2.** You do not need a separate `freeze_embeddings` helper for Stage 2 — the LoRA injection block (shown below) starts with `for p in model.parameters(): p.requires_grad = False`, which freezes everything atomically: the entire vision encoder, the BART encoder, **all 4 tied pointers** (shared, encoder.embed_tokens, decoder.embed_tokens, lm_head), `embed_positions`, the decoder layers — every parameter in the model. Only the newly-created `LoRALinear.lora_A` and `LoRALinear.lora_B` end up `requires_grad=True` after injection, because they were just instantiated and inherit the PyTorch default.

If you prefer defensive explicitness, you can still call a separate `freeze_embeddings(model)` helper after the LoRA injection — it'll be a no-op because the tied pointers are already frozen, but it makes intent visible to anyone reading the training script.

### Why decoder cross-attention specifically

Self-attention LoRA helps the decoder reason about token-to-token consistency (closing `<player_N>` correctly, putting OCR after bbox, etc.).

**Cross-attention LoRA is the real lever.** The decoder cross-attends from the token sequence into the (frozen) visual feature map. Adapting cross-attention is how the decoder learns *where to look* in the visual features for jersey numbers, scene context, team colours — without changing the visual features themselves. This is the entire spatial-binding mechanism from the talk.

### Loss — token-weighted cross-entropy with two boosts

Per-token weighted CE with three weight classes and two on-top boosts:

| Token class | Suggested weight | Examples |
|---|---|---|
| HIGH | `5.0` | `<bbox>`, `</bbox>`, `<player_N>`, `</player_N>`, `<stype>`, `<nath>`, `</s>`, all `<loc_*>` |
| OCR (highest) | `12.0` | `<ocr>` and the content tokens that follow it (jersey-number text + 8 `<loc_*>`) |
| LOW | `0.7` | `<gdesc>` and the long natural-language span after it |
| Default | `1.0` | Everything else |

**Building the per-token weight tensor.** The weights above are per-token-*class*, but cross-entropy applies them per-token-*id*. You need to convert each class's token names into integer ids using the loaded (and registration-completed) tokenizer, then build the weight mask per batch by table-lookup against the labels. The full pattern is short:

```python
# ---- Once, after the tokenizer has the custom tokens registered (see TOKENS.md): ----
vocab = tokenizer.get_vocab()

HIGH_IDS = {vocab[t] for t in ("<bbox>", "</bbox>", "<stype>", "<nath>", "</s>") if t in vocab}
HIGH_IDS |= {vocab[f"<player_{i}>"]  for i in range(1, 9) if f"<player_{i}>"  in vocab}
HIGH_IDS |= {vocab[f"</player_{i}>"] for i in range(1, 9) if f"</player_{i}>" in vocab}
HIGH_IDS |= {vocab[f"<loc_{i}>"]     for i in range(1000) if f"<loc_{i}>"     in vocab}   # the 1000 location bins

OCR_IDS              = {vocab["<ocr>"]}
LOW_IDS              = {vocab["<gdesc>"]}
CONTENT_TRIGGER_IDS  = {vocab["<ocr>"], vocab["<stype>"]}   # tokens that open a boosted content span

# ---- Per batch, inside the loss forward: ----
weights = torch.ones_like(labels, dtype=torch.float32)              # default = 1.0
for tid in HIGH_IDS: weights[labels == tid] = HIGH_WEIGHT            # 5.0
for tid in OCR_IDS:  weights[labels == tid] = OCR_WEIGHT             # 12.0  (trigger itself; content boost extends this)
for tid in LOW_IDS:  weights[labels == tid] = LOW_WEIGHT             # 0.7
# ... then layer the content + positional boosts (next two bullets) on top of `weights` ...
weights[labels == IGNORE_INDEX] = 0.0                                # mask padding (the standard -100 sentinel)

per_token_ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none", ignore_index=IGNORE_INDEX)
loss = (per_token_ce * weights.flatten()).sum() / weights.flatten().sum().clamp(min=1.0)
```

**Three pitfalls in this construction that will silently corrupt training:**

* **Don't prefix-match on token strings.** Use exact dictionary lookups (`vocab[token]`) and verify each lookup succeeds. Substring matches on `<player_` will conflate `<player_1>` with `<player_10>` (if you ever raise `MAX_PLAYER_INDEX`); substring matches on `<loc_` will conflate `<loc_1>` with `<loc_100>`. Wrong ids in the id-sets means wrong tokens get weighted, which is undetectable by every standard training-loss curve.
* **Don't cache the id-sets across a tokenizer change.** If you add or remove any custom token (a new `<player_9>`, a new attribute token, etc.) you must (a) re-run the registration block in [`TOKENS.md`](TOKENS.md), (b) re-load the model + processor from the new snapshot, and (c) **rebuild these id-sets from the new tokenizer**. The string-to-id mapping changes after every `add_special_tokens` call.
* **Don't assume `<loc_*>` ids are contiguous.** Florence-2 happens to assign them contiguously today, but the canonical source of truth is the dictionary lookup. Iterating with `range(loc_0_id, loc_999_id + 1)` will break the moment the underlying tokenizer is rebuilt or you swap in a different base model.

Two boosts layered on top:

* **Content boost.** Tokens immediately following an `<ocr>` or `<stype>` trigger inherit the trigger's weight for the next ~19 tokens. This is how the actual jersey-number text and the scene-class string get weighted (they aren't structural tokens themselves, but they're the content that matters).
* **Positional boost.** Tokens inside `<player_N>` get scaled by `1 + (N - 1) * rate`. With `rate = 0.4`: player 1 = 1.0×, player 5 = 2.6×, player 8 = 3.8×. Later players are harder (cross-attention degrades by position in the sequence); the boost compensates.

### The LoRA-injection pattern

Stage 2's pipeline is: load the Stage 1 checkpoint → freeze everything → inject LoRA into the decoder's self- and cross-attention → optimise only the LoRA parameters. The injection is the only non-trivial piece — here is the minimal pattern:

```python
import math, torch, torch.nn as nn

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank delta."""
    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.original = original
        self.scaling  = alpha / rank
        self.lora_A   = nn.Parameter(torch.empty(rank, original.in_features))
        self.lora_B   = nn.Parameter(torch.zeros(original.out_features, rank))   # zero -> identity at init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop     = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.drop(x) @ self.lora_A.T @ self.lora_B.T * self.scaling


def inject_decoder_lora(model, rank=16, alpha=32.0, dropout=0.05) -> int:
    """Wrap every q_proj / v_proj / out_proj in BART decoder self- AND cross-attention."""
    for p in model.parameters():
        p.requires_grad = False                                                  # freeze everything first
    decoder = model.language_model.model.decoder
    count = 0
    for layer in decoder.layers:
        for attn_name in ("self_attn", "encoder_attn"):                          # encoder_attn == cross-attn
            attn = getattr(layer, attn_name, None)
            if attn is None: continue
            for proj_name in ("q_proj", "v_proj", "out_proj"):                   # NOT k_proj
                orig = getattr(attn, proj_name)
                setattr(attn, proj_name, LoRALinear(orig, rank=rank, alpha=alpha, dropout=dropout))
                count += 1
    return count   # ~72 wrapped projections on Florence-2-large
```

After injection, build the optimizer over `[p for n, p in model.named_parameters() if "lora_" in n]` only.

At save time, merge each `LoRALinear` back into a plain `nn.Linear` (`W ← W + (B @ A) * scaling`), then `save_pretrained`. The resulting checkpoint is a clean Hugging Face model — no LoRA library needed at inference.

### Hyper-parameters that matter

| Knob | Suggested | Why |
|---|---|---|
| Learning rate | `1e-4` | Higher than Stage 1 because we're training tiny LoRA adapters, not big embeddings. |
| LoRA rank | `16` | Sweet spot — `r=4` underfits, `r=32` adds compute without quality gain. |
| LoRA alpha | `32.0` | Standard scaling = α / r = 2.0. |
| LoRA dropout | `0.05` | Keep small; LoRA already regularises. |
| Batch size | `8` | Smaller than Stage 1 because the weighted-loss bookkeeping is memory-heavy. |
| Epochs | `5` | Larger datasets benefit from 7-10; LoRA at `r=16` rarely overfits. |
| OCR token weight | `12.0` | Don't drop below 8 unless your OCR text is large and easy to read. |
| Positional boost rate | `0.4` | Drop to `0.15-0.2` if you have ≤ 4 entities; the boost grows quickly with player index. |

### What "done" looks like

Validation loss decreasing slowly over 3-5 epochs. Generation samples on the validation set look right — boxes tighten, OCR text is correct, the scene-description text is coherent. CER on jersey numbers drops to single digits.

If Stage 2's loss explodes after warmup, lower the OCR token weight from 12 to 8 — on noisy annotations the loss is sometimes dominated by one mis-predicted OCR span.

---

## Resume / restart cookbook

| Situation | What to do |
|---|---|
| Fresh start | Run Stage 1 on the base Florence-2 with custom tokens enabled, then Stage 2 on the Stage-1 best checkpoint. |
| Stage 1 succeeded, want to sweep Stage 2 hyper-params | Re-run only Stage 2, pointing at the same Stage-1 best checkpoint. |
| Stage 2 plateaued / regressed | Fall back to the Stage-1 best checkpoint, lower LR or OCR token weight, retry Stage 2. |
| Token list changed | **Re-run Stage 1 from base Florence-2.** Adding or removing tokens after Stage 1 corrupts the new embedding rows. |
| Dataset grew | Just re-run both stages — the curriculum is fast enough that re-training from scratch is cleaner than incrementally fine-tuning. |
