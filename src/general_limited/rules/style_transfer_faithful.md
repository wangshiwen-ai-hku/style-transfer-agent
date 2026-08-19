---
name: style_transfer_faithful
description: Style transfer under a strict preservation contract. Transfers appearance-level style (line, palette, texture, motifs, lighting) while treating facial geometry, expression, gaze and figure proportion as identity-critical and forbidding their alteration. Use when the user requires that the subject remain geometrically and expressively unchanged.
---

## Overview
(Critical Rule)
MUST KEEP the **position, orientation and rotation** of the content.
For you: **Tailor the analysis focus** according to the user provided **style image**.

This is the FAITHFUL variant of the style transfer task. It differs from the default
task in exactly one respect: which attributes are allowed to change. Everything else
(analysis depth, staged planning, reflection) is unchanged.

## Preservation Contract (STRICT)

The following are **identity-critical** and MUST be preserved. They may NOT appear in
the `change` list of any region, and no stage prompt may instruct their alteration:

- facial geometry: the position and shape of eyes, eyebrows, nose, mouth and jawline,
  and the proportions between them;
- **eyelid state and gaze direction**: if the subject's eyes are open in the content
  image, they MUST remain open, looking in the same direction; likewise if closed;
- facial expression: the emotion displayed by the subject is a property of the CONTENT,
  not of the style reference, and must not be replaced by the reference's expression;
- figure proportion: head-to-body ratio, limb length and body shape. Re-proportioning
  (chibi, caricature, elongation, geometric abstraction of the figure) is FORBIDDEN;
- pose, position and spatial arrangement of the subject and all objects;
- the number and identity of subjects and objects;
- any legible text.

The following remain **style-transferable** and SHOULD be transferred as fully as the
style reference warrants:

- line: weight, variation, continuity, drawn vs implied contours;
- colour palette: hues, accents, saturation and value range;
- texture and mark-making: brush or pen behaviour, granularity, edge quality;
- lighting, contrast and overall atmosphere;
- decorative motifs and ornaments belonging to the style rather than to its subject;
- background treatment and rendering, provided the background's spatial structure and
  the subject's placement within it are preserved.

## For Analyser
Style Attributes Analysis: analyse the style image from multiple aspects according to
its distinctive features. Grasp the main styles and describe them in enough detail that
another artist could repaint from the description. Identify the NON-style information in
the style image (i.e., its own subject matter) and do NOT transfer it to the content
image unless it is a recurring decorative motif.

Additionally, emit the `preservation_contract` field described in the default task file,
with every identity-critical attribute above listed under `keep` for the relevant region,
and NONE of them under `change`.

## For Planner
Attention to these aspects, IF ANY in this case:
- (Critical) Make use of a SketchDraft stage to strip the content image's original
  rendering style **without** displacing its geometry: the draft must be traced from the
  content image's own contours, not redrawn from a description.
- Care for similar-region transfer: if the content and style share regions (face, hair,
  cloth), transfer the style's *rendering* of that region while keeping the content's
  *geometry* of it.
- Apply the style's composition through palette, texture and background treatment rather
  than by moving, rescaling or re-cropping the subject.
- Add special decorations/motifs in regions that do not overlap facial features.
- Background style consistency: hold the original background structure and stylize it.

## Guard
If faithfully reproducing the style reference would require violating an identity-critical
attribute (for example, a style whose identity rests on severe facial distortion), do NOT
violate it. Transfer the style as far as the contract permits, and record the conflict in
the analysis output so that it is visible in the result rather than silently resolved.
