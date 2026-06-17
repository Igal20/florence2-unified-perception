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

A held-out benchmark set should live outside this dataset (or be flagged by a hash list and filtered out of train/val) so you can evaluate without leakage.

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

**Serialised token string** — exactly what the decoder must learn to emit for this image (the `<loc_*>` indices are computed from the pixel coordinates above using the quantisation formula in the previous section):

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
