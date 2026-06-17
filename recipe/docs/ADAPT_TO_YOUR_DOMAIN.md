# Adapting the recipe to your own domain

> The recipe is sports because that's our domain. The *pattern* fits any multi-task perception problem where the output is naturally hierarchical: a few scene-level fields, plus a list of entities, where each entity has attributes (bounding boxes, IDs, classes, sub-regions with text).

This doc walks through porting the recipe to a non-sports domain. The running example is **retail shelf inspection**: one shelf photo in, a structured output with shelf-level metadata plus a list of product instances, each with a bounding box, an OCR-extracted brand/price, optional structured attributes (`in_stock` / `out_of_stock`), and a free-form shelf description.

## Step 1 — Pin down your output schema

List the fields you want the model to emit and group them as scene-level vs entity-level:

| Category | Sports example | Retail example |
|---|---|---|
| Scene-level class | scene type (In-Game, Interview, …) | shelf type (Beverages, Snacks, …) |
| Scene-level count | num athletes | num products |
| Scene-level description | general description | shelf description |
| Per-entity bounding box | player bbox | product bbox |
| Per-entity OCR text + polygon | jersey number + polygon | brand/price text + polygon |
| Per-entity categorical attributes | (none) | `<in_stock>` / `<out_of_stock>` |

Make this list exhaustive before touching anything else. Every later step assumes it's frozen.

## Step 2 — Rename your tokens

Replace the sports-specific token names with your domain's names — but keep the *shape* identical:

| Sports | Retail |
|---|---|
| `<stype>` | `<shelf_type>` |
| `<gdesc>` | `<shelf_desc>` |
| `<nath>` | `<nprod>` |
| `<bbox>` … `</bbox>` | `<bbox>` … `</bbox>` (unchanged — generic geometric token) |
| `<ocr>` | `<ocr>` (unchanged) |
| `<player_1>` … `<player_8>` and closing pairs | `<prod_1>` … `<prod_20>` and closing pairs |

A few notes:

* Shelves typically hold more entities than a sports scene shows players — raise the per-entity cap from 8 to 20 (or whatever your typical count + slack is). Florence-2 handles hundreds of structural tokens without difficulty.
* Categorical attributes like `in_stock` become **new singleton tokens** (`<in_stock>` and `<out_of_stock>`). One token per value — never embed the value as plain text.
* Add the new tokens to the custom-token list and run the registration recipe in [`TOKENS.md`](TOKENS.md) on the base Florence-2.

## Step 3 — Update the grammar your serialiser produces

Mirror the new structure. The *shape* of the output string stays the same — header tokens, then one block per entity, then a final description:

```
<MULTIMODAL_VISUAL_CAPTION><shelf_type>{CLASS}<nprod>{N}
  <prod_1><bbox><loc_*x4></bbox><in_stock><ocr>{BRAND/PRICE}<loc_*x8></prod_1>
  <prod_2><bbox><loc_*x4></bbox><out_of_stock></prod_2>
  ...
<shelf_desc>{FREE TEXT}</s>
```

When updating the serialiser, the only domain-specific decisions are:

1. Which field maps to which token prefix.
2. The order of per-entity blocks. (For shelves, **left-to-right, top-to-bottom** is sensible; the model will learn whatever order you pick as long as it's consistent.)
3. Where structured attribute tokens go inside the entity block. (Convention: right after the bbox, before any OCR.)

Run the data-prep smoke test from [`DATA_PREPARATION.md`](DATA_PREPARATION.md) on a handful of converted annotations and **read the resulting strings with your eyes**. If you can't parse them visually, the model can't either.

## Step 4 — Update the parser

The parser at inference time is the inverse of the serialiser. Rename the regex patterns to match your new token names:

* `<shelf_type>([^<]+)` for the scene class.
* `<nprod>(\d+)` for the entity count.
* `<shelf_desc>([^<]*)` for the description.
* `<prod_(\d+)>(.*?)</prod_\1>` for each entity block.
* `<bbox><loc_\d+><loc_\d+><loc_\d+><loc_\d+></bbox>` for boxes (unchanged).
* `<ocr>([^<]+)<loc_\d+>...<loc_\d+>` for OCR (unchanged shape, 8 `<loc_*>`).
* `<in_stock>` / `<out_of_stock>` as plain substring matches inside each entity block.

## Step 5 — Update the schema-compliance early-stop regexes

The Stage 1 early-stop checks are short regex patterns (`<shelf_type>` is present, balanced `<prod_N>...</prod_N>`, `<nprod>` count matches the number of entity blocks, well-formed bbox, etc.). Rename them to match your tokens. Keep the rule that ≥ 95% of validation generations must pass before Stage 1 exits.

## Step 6 — Update the weighted-loss token classes

For the Stage 2 hierarchical loss, sort your new tokens into the three weight classes:

| Class | Tokens for the retail example |
|---|---|
| HIGH | `<shelf_type>`, `<nprod>`, `<bbox>`, `</bbox>`, all `<prod_N>` / `</prod_N>`, `<in_stock>`, `<out_of_stock>`, all `<loc_*>` |
| OCR | `<ocr>` (plus content boost on the next ~19 tokens) |
| LOW | `<shelf_desc>` (plus content boost on the long free-text span after it) |

Structural attribute tokens like `<in_stock>` belong to HIGH because they're categorical markers — getting them wrong is a hard error, not a fuzzy one.

## Step 7 — Reuse the training pipeline as-is

You should not need to touch the freezing policy, the LoRA-injection pattern, the training loop, or the optimiser construction. They are grammar-agnostic — they operate on token-id sets and parameter names, not on domain semantics.

Concretely, after you swap the token names and rebuild the token-id sets for the loss, every step of the curriculum in [`TWO_STAGE_TRAINING.md`](TWO_STAGE_TRAINING.md) runs unchanged.

## Step 8 — Tune the hyper-parameters for your domain

| Default for sports | Likely change for a new domain |
|---|---|
| `MAX_PLAYERS = 6` | Match your typical entity count + 1 or 2 for slack. Shelves: 15-20. |
| `OCR_TOKEN_WEIGHT = 12` | Lower (8) if text is large and easy (price tags). Raise (15+) if text is tiny (serial numbers). |
| `POSITIONAL_BOOST_RATE = 0.4` | Reduce to `0.15-0.2` if you have many entities (>10) — the boost grows quickly. |
| `TARGET_RESOLUTION = 768 × 768` | Keep. Florence-2 is native at 768; up-scaling rarely helps and slows training. |
| `STAGE2.EPOCHS = 5` | Larger datasets benefit from 7-10. LoRA at `r=16` rarely overfits. |

If you have a radically different visual domain (medical microscopy, satellite imagery, factory IR images, X-ray), seriously consider adding a third stage with surgical LoRA on the late DaViT vision blocks. The sports recipe leaves Stage 3 out because Florence-2's natural-image pretraining already covers most sports scenes; that argument doesn't hold for non-natural imagery.

## Other domains the same recipe fits

| Domain | "Entities" | OCR-like field | Scene-level fields |
|---|---|---|---|
| Retail shelf inspection | Products | Brand / price label | Shelf type, total count |
| Medical microscopy | Cells | Cell-ID or class label | Tissue type, density |
| Construction site safety | Workers | PPE class / hard-hat colour | Site phase, hazard score |
| Manufacturing line | Parts | Serial number | Line status, defect count |
| Satellite imagery | Buildings / vehicles | (often none) | Land-use class, coverage % |
| Document understanding | Form fields | Field text | Document type, page number |
| Wildlife monitoring | Animals | (none, or tag IDs) | Habitat type, group size |

The two-stage curriculum, the LoRA-injection pattern, and the hierarchical loss are domain-agnostic. The only domain-specific work is renaming tokens, rewriting the serialiser / parser regexes, and tweaking the weight classes. Plan for **one engineering day** to port everything end-to-end, then a few days of training and tuning.

## Sanity checklist before you launch a training run

* [ ] Custom tokens registered and saved with the processor — verified by encoding each token and confirming it returns exactly one id.
* [ ] Serialiser produces a consistent ordering across the entire dataset (`<gdesc>` always last, attributes always in the same slot, etc.).
* [ ] Eyeballed 10 random serialised strings — all parse visually.
* [ ] Token-id sets for the weighted loss have been recomputed from the *new* tokenizer (not copy-pasted from the sports recipe).
* [ ] Early-stop regexes rewritten with the new token names.
* [ ] Held-out benchmark set defined and excluded from train/val.
* [ ] Dataset is at least ~3-5K labelled images for Stage 1, ideally 5-20K for Stage 2 to be meaningful.

If all seven boxes are ticked, you're ready to launch. Stage 1 typically finishes in a few hours on a single A10 / 3090; Stage 2 in another few hours.
