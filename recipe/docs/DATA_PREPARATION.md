# Data preparation

The entire pipeline is anchored by one decision: **what string does the decoder learn to emit for one image?** Once that string is settled, the rest of the recipe falls into place. Get this wrong and Stage 1 will never reach high schema compliance no matter what hyper-parameters you sweep.

## The annotation contract

One image, one annotation file with the same stem, containing these fields (names are suggestions — yours can differ as long as they map cleanly to the grammar below):

| Field | Required | What it is |
|---|---|---|
| `image_size` | yes | `[width_px, height_px]` — needed to normalise every coordinate. |
| `scene_type` | yes | One of a small closed vocabulary (we used 9 sport-context classes). Anything outside the vocabulary collapses to `"Other"`. |
| `general_description` | recommended | One free-form English sentence about the scene. If empty, no `<gdesc>` block is emitted. |
| `players` (entities) | yes | A list of entity dicts. Each entity has a bbox and optionally an OCR list. |
| `players[i].bbox` | yes per entity | `[x1, y1, x2, y2]` in pixel space, top-left + bottom-right. |
| `players[i].team` | optional per entity | Team-affiliation label from a small closed vocabulary, e.g. `"A"` or `"B"`. Omit (do not set to `null`) when affiliation can't be determined — referees, partial occlusion, unaffiliated bystanders. |
| `players[i].ocr` | optional per entity | List of `{text, polygon}` items. Polygon = 4 `[x, y]` points in pixel space. |

### The `scene_type` vocabulary

The closed scene-type vocabulary we used has nine classes — anything outside it collapses to `"Other"` (the ninth class). Pick a vocabulary of similar size for your domain (3-12 classes works; 30+ classes blurs the boundary between scene classification and free-form description and the model stops gaining from the structured label).

| Class | Definition |
|---|---|
| `In-Game` | Action on the field/court/ice during competition. |
| `Interview` | One-on-one or group interviews (on-site or studio). |
| `Press Conference` | Formal media session with podium/backdrop/mics. |
| `Warm-ups` | Pre-game athlete activity (stretching, drills, layups). |
| `Player Arrivals` | Athletes entering venue, tunnel walks, pre-game arrivals. |
| `Locker Room` | Locker room scenes (pre, post, halftime). |
| `Winning Ceremony` | Trophy lifts, medal presentations, confetti moments. |
| `Fan Atmosphere` | Crowd shots, chants, fan reactions and celebrations. |
| `Other` | Everything not covered above; ambiguous contexts; the safety net for typos and unknown values. |

### Splits and held-out benchmark

* **Default train / val / test = `[0.80, 0.10, 0.10]`** with a deterministic, seeded shuffle on a stable filename ordering. Pin the seed across runs or your eval numbers won't be comparable. We used `[0.85, 0.10, 0.05]` for the production curriculum because the dataset was large enough that 5% was a comfortable test slice.
* **Held-out benchmark set is excluded by perceptual hash.** Each annotation JSON carries an `image.p_hash` field — the 64-bit pHash of the source image, computed once during data prep (we used the `imagehash.phash` library; the canonical lowercase hex string is what lives in the JSON). The benchmark CSV is a flat column of these hex strings. At dataset-load time, every image whose `p_hash` matches a benchmark hash by **exact string equality** (case-normalised, whitespace-stripped — *no Hamming-distance radius*) is dropped from train and val. This is intentionally stricter than typical pHash dedup: a Hamming radius would also drop near-duplicates *inside* your training set, and for broadcast-style data with many similar frames of the same play you usually want to keep those.
* **Why not just split by source-file path?** Broadcast frames are typically extracted from contiguous video; consecutive frames are near-identical but live in different files. A path-based split puts those near-duplicates in both train and test and inflates your test accuracy by 10-30 points. Hash-based exclusion catches what the path-based split misses.

## The grammar your serialiser must produce

For one image, the serialised string must look exactly like this — same token order, same closing tags, no extra whitespace inside tags:

```
<MULTIMODAL_VISUAL_CAPTION><stype>{SCENE_TYPE}<nath>{N}
  <player_1><bbox><loc_*><loc_*><loc_*><loc_*></bbox><team>{TEAM}
    <ocr>{TEXT}<loc_*><loc_*><loc_*><loc_*><loc_*><loc_*><loc_*><loc_*>
    ... (more ocr items per player) ...
  </player_1>
  <player_2>...</player_2>
  ...
<gdesc>{FREE TEXT}</s>
```

> **What `<MULTIMODAL_VISUAL_CAPTION>` is** — a custom task-prompt token that you register alongside the structural tokens (see [`TOKENS.md`](TOKENS.md#the-task-prompt-token)). It's the *input* to the encoder, not part of the decoder's output. The template above shows the full conceptual sequence (input prompt + decoder output, concatenated) for readability; the model itself only generates from `<stype>` onwards. How you handle the prompt at training time (separate `labels` tensor vs. prompt-masked concatenated string) is up to your training recipe.

Rules the LLM you task with building the serialiser must respect:

1. **Header first.** `<stype>{class}` then `<nath>{count}` always come before any player block.
2. **One block per entity, indexed.** Use `<player_1>...</player_1>`, `<player_2>...</player_2>`, etc. Cap at 8 (or whatever `MAX_PLAYER_INDEX` you choose), drop the extras.
3. **Bbox immediately inside a player block.** `<bbox>` then exactly 4 `<loc_*>` then `</bbox>`.
4. **Team comes after the bbox, before any OCR.** `<team>` followed by exactly one short class label from a small closed vocabulary (we used `"A"` and `"B"` — two teams in frame is the common case for sports). Anything outside the vocabulary is dropped, not coerced to a default. The team label sits *between* `</bbox>` and the first `<ocr>` so that detection and affiliation are always emitted as one atomic block per entity.
5. **OCR is text + polygon, glued together.** Always `<ocr>` then the literal text (no spaces inside the token), then exactly 8 `<loc_*>` (the 4 polygon corners as x,y pairs). The polygon **must** come immediately after the text — that gluing is what suppresses hallucinated jersey numbers.
6. **Optional fields are skipped, not emitted as empty.** No `<ocr>` block at all for players with no readable text; no `<team>` block at all for players whose affiliation is unknown; no `<gdesc>` block at all for images without a caption.
7. **`<gdesc>` is last,** before `</s>`. (You can put it first instead, but pick one ordering and stay with it across the entire dataset — Stage 1 will only converge if the order is consistent.)

## Quantising coordinates

Florence-2 ships 1000 special location tokens `<loc_0>` … `<loc_999>` that discretise normalised `[0, 1]` coordinates. **Reuse them — do not add new location tokens.** Each pixel coordinate becomes one `<loc_*>` via this formula:

> `bin = floor(min(1.0, max(0.0, value / dim)) * 1000)` → emit `<loc_{bin}>`

`dim` is `width` for x-coordinates, `height` for y-coordinates. Boxes use 4 such bins, polygons use 8 (4 (x, y) pairs). The Medium post explains why creating new `<loc_*>` tokens is strictly worse than reusing the pretrained ones.

## Worked example

To make the contract concrete, here is one image, its annotation file, and the exact token string the serialiser must produce for it. The image is 1024 × 682 px and contains two hockey players, both with readable jersey numbers.

![Two hockey players battling for the puck along the boards — one in a green Forest Trykers #28 jersey, one in a maroon Iron Falcons #45 jersey](images/example_hockey.jpg)

**`example_hockey.json`** — the annotation in the contract format above:

```jsonc
{
  "image_size": [1024, 682],
  "scene_type": "In-Game",
  "general_description": "Two hockey players battle for the puck along the boards: a Forest Trykers forward wearing #28 attempts to shield it from the opposing Iron Falcons captain #45.",
  "players": [
    {
      "bbox": [220, 30, 560, 670],
      "team": "A",
      "ocr": [
        { "text": "28", "polygon": [[275, 150], [360, 150], [360, 215], [275, 215]] }
        // ... additional <ocr> items per player if more readable text is visible
        //     (jersey logo text, captain "C", sleeve numbers, ad boards behind, ...)
      ]
    },
    {
      "bbox": [510, 90, 820, 660],
      "team": "B",
      "ocr": [
        { "text": "45", "polygon": [[685, 265], [750, 265], [750, 320], [685, 320]] }
        // ... additional <ocr> items per player if more readable text is visible
      ]
    }
  ]
}
```

> **Note on the `// ...` lines** — those are illustrative comments showing where additional OCR items would go in a real annotation; they are **not** part of the contract. A strict JSON file has no comments. Each `ocr` array can hold **0..N** items per player; the worked example shows one each because that's all the imagery cleanly contains.

**Serialised token string** — the full conceptual sequence (encoder input prompt + decoder-generated output, concatenated). The decoder only emits the portion from `<stype>` onwards; the leading `<MULTIMODAL_VISUAL_CAPTION>` is the input passed through the processor (see [`TOKENS.md`](TOKENS.md#the-task-prompt-token)). The `<loc_*>` indices are computed from the pixel coordinates above using the quantisation formula in the previous section:

```
<MULTIMODAL_VISUAL_CAPTION><stype>In-Game<nath>2
  <player_1><bbox><loc_214><loc_43><loc_546><loc_982></bbox><team>A
    <ocr>28<loc_268><loc_219><loc_351><loc_219><loc_351><loc_315><loc_268><loc_315>
    ... (additional <ocr> items per player would continue here, one per readable region) ...
  </player_1>
  <player_2><bbox><loc_498><loc_131><loc_800><loc_967></bbox><team>B
    <ocr>45<loc_668><loc_388><loc_732><loc_388><loc_732><loc_469><loc_668><loc_469>
    ... (additional <ocr> items per player would continue here) ...
  </player_2>
<gdesc>Two hockey players battle for the puck along the boards: a Forest Trykers forward wearing #28 attempts to shield it from the opposing Iron Falcons captain #45.</s>
```

> **Note on the `... (additional <ocr> items …) ...` lines** — those are illustrative placeholders, not literal tokens the model should emit. They mirror the `... (more ocr items per player) ...` line from the grammar template at the top of this doc, and exist only to make clear that the per-player `<ocr>` block is variable-length. In the actual training target, only the concrete `<ocr>…` lines are present, joined together with no whitespace.

Things worth noticing in the string above:

* **`<nath>2`** — the explicit entity count must equal the number of `<player_N>` blocks. The model checks this against itself, and a mismatch is a strong "this generation went off the rails" signal at inference time.
* **Team affiliation is per-player and abstract** — `<team>A` for the Forest Trykers, `<team>B` for the Iron Falcons. The model never sees the real team names; it learns the *relative* concept "all players sharing the same jersey design are the same team", which generalises across leagues, sports, and unseen uniforms. Keeping the value vocabulary small and closed (`"A"`, `"B"`, optionally `"C"` for a referee) is what makes this work.
* **OCR text and its polygon are adjacent** — `<ocr>28<loc_...>` and `<ocr>45<loc_...>`. That gluing (rule 5 of the grammar) is what binds the recognised text to a region of the image and suppresses free-floating jersey-number hallucinations.
* **OCR is a list per player, not a single value.** Each player can have 0..N `<ocr>…` blocks back-to-back inside their `<player_N>` block — one per readable region. The example above shows one each because that's all the image cleanly contains; a real annotation of a player whose jersey shows `#28` on the back, `FOREST TRYKERS` on the chest, and a captain's `C` on the shoulder would have three `<ocr>…` blocks for that one player. The `// ...` and `... (additional <ocr> items …) ...` markers in the JSON and token string are illustrative; they are not tokens the model emits.
* **The indentation and newlines above are for human readability only.** The actual training target is a single contiguous string with no whitespace inserted.
* **A player whose number is occluded** would simply have no `ocr` field at all — never an empty list, never an empty `<ocr></ocr>` block. Same for `<team>` when the team can't be determined. Rule 6.
* **`<gdesc>` is last**, immediately before `</s>`. Pick one slot and stay with it across the entire dataset.

## Training-time image preprocessing

Once the annotation contract is settled, the per-image pipeline at training time is short but specific. Every choice here affects schema compliance, OCR accuracy, and how well the model holds up on unseen broadcasts.

### The full pipeline, in order

| Step | What it does | Notes |
|---|---|---|
| 1. `PIL.Image.open(path)` | Lazy-load. | |
| 2. Apply EXIF orientation | Rotate per the camera's EXIF `Orientation` flag. | Skipping this is the #1 silent-killer bug — phone uploads (portrait orientation) end up rotated 90° relative to their pixel-space annotations and every bbox is wrong. |
| 3. Force RGB mode | Convert from RGBA / L / CMYK / palette to RGB. | A single CMYK image in the loader silently crashes the processor halfway through training. |
| 4. Training-only augmentations | See the augmentation table below. | Val / test get **none** of these — only steps 1, 2, 3, and the final resize. |
| 5. Final resize to `(768, 768)` | Plain `image.resize(..., resample=BICUBIC)` — **stretch, not letterbox.** | Florence-2's processor expects square 768×768. Stretching distorts aspect ratio, but it's fine because all coordinates are normalised to `[0, 1]` *before* the resize, so they keep their correct relative positions. |
| 6. `processor(images=..., text=task_prompt, return_tensors="pt", padding=True)` | Florence-2's `CLIPImageProcessor`: another resize to 768×768 if step 5 wasn't done, then `rescale = 1/255` then ImageNet normalisation (`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`). Center crop is **off**. | Step 5's resize is technically redundant with the processor's resize, but doing it explicitly in the dataset makes the augmentations operate at the final scale, not the original resolution — more deterministic loss behaviour. |

### The augmentation policy (training split only)

The augmentation policy is **OCR-aware** — when the sample has at least one `<ocr>` item, the heavier transforms are dialled back or skipped entirely to protect text edges. The numbers below are the production defaults; the OCR-aware branches are the small details that lift OCR CER by 3-5 points.

| Augmentation | Probability (per sample) | Range | OCR-aware? |
|---|---|---|---|
| Photometric "block" (gates the next 4) | 80% | — | no |
| ↳ Brightness | 50% within block | ±15% | no |
| ↳ Contrast | 50% within block | ±15% | no |
| ↳ Saturation | 50% within block | ±20% | no |
| ↳ Sharpness | 30% within block | ±10% | no |
| Convert to grayscale (then back to 3-channel RGB) | 5% | — | no (same rate for OCR and non-OCR) |
| Gaussian blur | **3% if OCR present**, **20% otherwise** | radius `0.1-0.3` (OCR) or `0.5-1.0` (no OCR) | yes |
| Multi-scale downsample-then-upsample (off by default — `USE_MULTI_SCALE=False`) | 40% if enabled | scale `0.95-0.98` (OCR) or `0.7-0.95` (no OCR), random resample mode on the up-pass | yes |
| Gentle re-sharpening for OCR samples | 15% (OCR samples only) | factor `1.02-1.08` | yes |

**No geometric augmentations.** No random crops, no horizontal flips, no rotation — every one of those would invalidate the bbox / polygon coordinates that are already locked in by the annotation. If you add geometric augmentation later you must propagate the same transform through the `<loc_*>` quantiser, and a left-right flip is *especially* dangerous for sports because handedness, jersey-side numbers, and team-side conventions all break.

### Loss masking on padded regions

There is no image-side padding to mask — the resize is to a fixed square. The only mask is on the *text* side: padded label positions are set to `IGNORE_INDEX = -100`, and the weighted-CE loss zeroes their weight before the reduction (see the snippet in [`TWO_STAGE_TRAINING.md` → Loss](TWO_STAGE_TRAINING.md#loss--token-weighted-cross-entropy-with-two-boosts)).

## Three routes to a dataset

The annotation contract above is small, but actually producing thousands of annotations is the hard part. Pick whichever fits your budget:

1. **Hand-label everything** (small, very high quality). Use your annotation tool of choice for boxes + polygons, a spreadsheet for `scene_type` + `general_description`. Export, then have your LLM write a 30-line converter into the contract above.
2. **Semi-automated teacher pipeline** (recommended for 5–20K samples). One off-the-shelf model per sub-task gives you a noisy first pass; humans review only the disagreements:
   * Detector (YOLO, Grounded-SAM) → candidate bounding boxes
   * OCR (PaddleOCR, EasyOCR, or *cropped* Florence-2 on each box) → jersey-number text + polygon
   * VLM (Gemini, GPT-4o, Florence-2 cascade) → scene type + free-form description
   * Human-in-the-loop → review only the low-confidence cases
   * Serialise into the contract above
3. **Existing public datasets** (cheapest). Most public sports datasets cover only one sub-task; synthesise the rest with off-the-shelf models. Quality is bounded by the synthesised fields.

A single GPU and a long weekend gets you 5–20K annotations via route 2, which is enough.

## Before you train: smoke-test the serialiser

This is the single most useful thing you can do before launching any training run. For 5–10 randomly-sampled annotations, run your serialiser and **read the output string with your eyes**. If you can't parse it visually, the model won't learn it either. Look for:

* Consistent ordering (`<stype>` always before `<nath>`, `<gdesc>` always in the same slot).
* No stray spaces inside tag names.
* Every `<player_N>` has a matching `</player_N>`.
* Player indices are contiguous from 1, not random.
* OCR polygon has exactly 8 `<loc_*>` tokens, not 6 or 10.

This is the entire QA gate for the data-prep stage.

## Common pitfalls

* **Polygon point count ≠ 4.** Many OCR tools emit polygons with 6+ vertices. Resample to 4 corners (or the bounding rect) before serialising.
* **Pixel vs normalised coords.** All annotation coordinates are pixels. Normalisation happens inside the serialiser using `image_size`.
* **Image stem mismatch.** `0001.jpg` ↔ `0001.json`. Any prefix/suffix mismatch silently drops the sample.
* **`scene_type` typos.** Anything outside the closed vocabulary becomes `"Other"`. The smoke-test above catches this.
* **Mixing orderings.** If half the dataset has `<gdesc>` first and the other half has it last, Stage 1 plateaus around 70-80% compliance and you'll think you have a model problem when you have a data problem.
* **Too many entities.** With `MAX_PLAYER_INDEX = 8`, a 12-player annotation silently loses the last 4. Either raise the cap (and add tokens for it — see `TOKENS.md`) or accept the loss.
