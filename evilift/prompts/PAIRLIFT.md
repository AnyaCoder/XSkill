---
name: pairlift-evidence-composition
description: |
  Lift reusable single-image evidence-search skills to paired-image difference,
  commonality, spatial, and reasoning questions without extra planning calls.
version: 1.0.0
---

# PairLift Evidence Composition

## General Rule

Apply the relevant single-image evidence workflow independently to both images, then
combine the two evidence records. Do not compare only the global scene or rely on an
option's linguistic plausibility. A non-null answer must cite a visible cue and the
image or crop where it appears.

## Difference

1. Form the visual claim expressed by each candidate option.
2. Inspect the corresponding entity or region in both images.
3. Prefer one paired zoom operation that crops both `original_image` and
   `original_image_1` at comparable locations or contextual regions.
4. Select an option only when the claimed change is present in one image and absent
   from, or visibly different in, the other.

## Commonality

1. Search each image independently for the candidate entity, category, location, or
   relationship.
2. Require positive evidence in both images; evidence from only one image is
   insufficient.
3. For category questions, permit appearance changes but preserve semantic class.
4. For instance questions, require instance-level details rather than category
   similarity alone.

## Spatial and Reasoning Tasks

- Keep enough context around crops to identify relative position and neighboring
  entities.
- For reasoning answers, first verify participating entities or visual traces, then
  verify that their relation supports the option.
- Do not infer an event or relation without visible entity-level evidence.

## Null Option

Choose option E only after the relevant regions of both images have been checked and
none of A-D has positive paired evidence. Failure to notice a tiny cue in the global
view is not absence evidence. Conversely, never select A-D without a localized cue.

## Output

End with exactly one option letter in `<answer>...</answer>` tags. Before the tag,
state the shortest sufficient evidence: image side, region, and visible cue.

