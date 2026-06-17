# Custom tokens — the recipe

> Companion to the Medium post:
> [**Expanding Florence-2's Vocabulary — An Advanced Guide to Adding Custom Tokens During Fine-Tuning**](https://medium.com/@ygal20/expanding-florence-2s-vocabulary-an-advanced-guide-to-adding-custom-tokens-during-fine-tuning-138fab660b64)
>
> Read the post first for the conceptual picture. This doc is the executable checklist.

## Which tokens to register

For the sports recipe, the custom token list is:

| Token | Role |
|---|---|
| `<stype>` | Scene-type classification prefix. The class name is plain text after it. |
| `<gdesc>` | General-description prefix. Free English text follows until `</s>`. |
| `<nath>` | Number-of-athletes prefix. A small integer follows. |
| `<bbox>` … `</bbox>` | Bounding-box bracket. Exactly 4 `<loc_*>` inside. |
| `<team>` | Per-player team-affiliation prefix. A short class label follows (e.g. `"A"`, `"B"`). Use a small closed vocabulary; never the actual team name. |
| `<ocr>` | OCR span trigger. Text + 8 `<loc_*>` follow. |
| `<player_1>` … `<player_8>` and `</player_1>` … `</player_8>` | One open/close pair per entity index. Cap = 8 entities per image. |

For a different domain, rename them (`<shelf_type>`, `<prod_1>`, …) but keep the *shape* — one prefix per scene field, paired open/close tags per entity. Add domain-specific singleton tokens like `<in_stock>` / `<out_of_stock>` only if the value is categorical and small-cardinality.

## What you must NOT add

* **Do not add new `<loc_*>` tokens.** Florence-2 already ships with 1000 of them (`<loc_0>` … `<loc_999>`), and the FLD-5B pretraining has them solidly placed in semantic space. Adding new location tokens introduces fresh, untrained embeddings that compete with the pretrained ones — strictly worse than reusing the existing bins.
* **Do not register raw text content as tokens.** Jersey numbers, scene-type class names, the free-form description — all should come out of the tokenizer's existing sub-word vocabulary (typically as 1–3 BPE pieces each). Only the *structural* prefixes / brackets are added as new special tokens.

## Why the registration step is fragile

When you call the tokenizer's `add_special_tokens` API, the vocabulary grows but the model's embedding table *does not*. Forgetting to grow the embedding table is the most common failure mode — the next forward pass throws an index-out-of-range from the embedding lookup. Worse, Florence-2 ties **four** weight pointers together (encoder embedding, decoder embedding, shared embedding, lm_head). Growing the embeddings on a sub-module instead of the top-level model breaks the tie silently — the model trains but the lm_head produces garbage.

The minimal, correct registration is this:

```python
# Run ONCE, on the base microsoft/Florence-2-large, before any training.
# Save the resulting processor & model -- subsequent runs load the saved
# checkpoint with add_custom_tokens=False.
from transformers import AutoModelForCausalLM, AutoProcessor

CUSTOM_TOKENS = [
    "<stype>", "<gdesc>", "<nath>", "<bbox>", "</bbox>", "<team>", "<ocr>",
    "<player_1>", "</player_1>", "<player_2>", "</player_2>",
    "<player_3>", "</player_3>", "<player_4>", "</player_4>",
    "<player_5>", "</player_5>", "<player_6>", "</player_6>",
    "<player_7>", "</player_7>", "<player_8>", "</player_8>",
]

model     = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
processor = AutoProcessor.from_pretrained        ("microsoft/Florence-2-large", trust_remote_code=True)

missing = [t for t in CUSTOM_TOKENS if t not in processor.tokenizer.get_vocab()]
if missing:
    processor.tokenizer.add_special_tokens({"additional_special_tokens": missing})
    model.resize_token_embeddings(len(processor.tokenizer))   # <-- the line that's easy to forget
```

That is the entire registration. Save the model and processor immediately afterwards. From this point on, every later run loads the saved snapshot and skips the registration block (`add_custom_tokens=False`-equivalent).

## What `resize_token_embeddings` actually does

* Grows the shared embedding matrix from `(V, D)` to `(V + k, D)`.
* New rows are initialised with the **mean of existing embeddings** — a sensible warm start for fine-tuning. (Don't init them with zeros; the lm_head logits will collapse.)
* Walks all four tied pointers (`encoder.embed_tokens`, `decoder.embed_tokens`, `model.shared`, `lm_head`) — **but only if called on the top-level model**. Never call it on a sub-module like `model.language_model`.

## Smoke-test before training

Right after the registration block above, encode each new token and confirm it tokenises to exactly one id:

> for each token in `CUSTOM_TOKENS`: assert `processor.tokenizer.encode(token, add_special_tokens=False)` returns a list of length 1.

If any token tokenises to multiple ids (e.g. `["<", "stype", ">"]`), the registration silently failed and Stage 1 will train on the wrong target.

## Changing the token list later

If you add or remove a token after training has started, **you must re-run Stage 1 from the base Florence-2**. Resuming from an old checkpoint with new tokens leaves the new rows zero-initialised and silently corrupt. Florence-2 also does not support shrinking the embedding table cleanly, so removing tokens has the same restart cost as adding them.

## Common pitfalls

* **Forgetting `resize_token_embeddings`.** Throws `IndexError: index out of range in self` on the first forward pass.
* **Calling `resize_token_embeddings` on a sub-module.** Trains without error; lm_head silently produces garbage. Always resize on the top-level `AutoModelForCausalLM`.
* **Saving the model but forgetting to save the processor.** At reload time the tokenizer doesn't know the custom tokens; tokenisation falls back to per-character splits and Stage 2 trains on a completely different target than Stage 1 did.
* **Including raw content (e.g. specific team names) in the token list.** The model will memorise team names instead of learning a relative team-affiliation concept, breaking generalisation to unseen leagues.
* **Resuming Stage 1 with the registration block enabled, on a checkpoint that already has the tokens.** The `add_special_tokens` call is a no-op (the tokens already exist), and `resize_token_embeddings` is a no-op (the table is already the right size). Safe, but noisy in logs — you can skip the block entirely on resumes.
