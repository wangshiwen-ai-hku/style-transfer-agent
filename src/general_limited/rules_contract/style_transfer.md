---
name: style_transfer
description: Rules and guidelines for applying the artistic style from one image to the content of another.
---

## Overview
(Critical Rule)
MUST KEEP the **position and oritention and rotation** of the content.
For you: **Tailor the analysis focus** according the user provided **style image**..

For Analyser:
Style Attributes Analysis: Analysis the style image from multi-aspects according to the special style features. Grasp the main styles, analyze them in great detail that others can follows the description to repaint. Identity the NON-Style information in style image, e.g., its content. DON not transfer the non-style content to content image unless they are recurring decorations.

Preservation Contract (REQUIRED OUTPUT — additive, does not change what you transfer):
The content-analysis agent MUST emit a `preservation_contract` field alongside its other
output. It is a list, one entry per salient semantic region of the content image
(e.g., face, hair, garment, hands, objects, background), each with:
  - "region": the region name
  - "keep":   attributes that MUST survive stylization (identity, pose, position,
              spatial arrangement, object count, legible text, ...)
  - "change": attributes this plan intends to transform (line, palette, texture,
              proportion, composition, expression, background content, ...)
  - "why":    one short clause justifying the split for THIS style

(CRITICAL) USER RULES OUTRANK THE DEFAULT CLASSIFICATION.
The defaults below apply only when the user has said nothing about the attribute.
If the user's instructions ask for an attribute to be transferred — for example
facial details, expression, gaze, or figure proportion — then that attribute MUST be
placed under "change" for the relevant region and MUST NOT appear under "keep", and
the critique criteria must not require its preservation. A contract that contradicts
an explicit user request is a defect: it makes the system silently ignore the user.
Default classification, absent any user instruction: facial expression, gaze and
eyelid state are identity-critical and belong under "keep".

Be honest and specific: if the plan intends to alter facial expression or figure
proportion because the user asked or the style reference demands it, say so in
"change" rather than omitting it. The contract is checked against the result during
reflection, so an unstated change is a defect while a stated one is not.

For Planner:
Attention to these aspects, IF ANY in this case. tell to planner and emphasize it.
- (Critical) Make Use of SketchDraft: to remove the styles of content image if needed.
- Care For Similar Region Transfer: if the content and style has simliar regions, e.g., flowers, face, cloth...
- Adjust Compositions: if style composition is special. e.g. a lot of white space, totally geometric, boneless...
- Add Special Decorations/Motifs: e.g. flowers, animals...
- Background style consistency: Hold original background structure and apply Stylize Background.