---
name: style_transfer
description: Rules and guidelines for applying the artistic style from one image to the content of another.
---

## Overview
(Critical Rule)
MUST KEEP the **position and oritention and rotation** of the content.
For you: **Tailor the analysis focus** according the user provided **style image**..

For Analyser:
Analysis the style image from multi-aspects according to the special style features. Grasp the main styles, analyze them in great detail that others can follows the description to repaint. Identity the NON-Style information in style image, e.g., its content. DON not transfer the non-style content to content image unless they are recurring decorations.

## For Planner
The planner's role is to convert the analyser's detailed **Style-to-Content Mapping** into a single-stage, comprehensive generation prompt.

1.  **Degenerate the Plan**: The plan MUST consist of only **one stage**.
2.  **Synthesize a Comprehensive Prompt**: Combine all the details from the Style-to-Content Mapping into a single, rich `text_prompt`. This prompt should guide the generation model precisely on what to create.
    -   *Example Prompt based on the mapping above*: "An abstract, cubist-style portrait of a man, inspired by the provided reference image. Construct his face from vibrant, interlocking geometric color blocks (blue, green, orange, purple). Use simple, thick black lines for the eyes, nose, and mouth. The hair should be a large, solid block of color. Encase the entire portrait in thin, golden-orange outlines, set against a coarse, textured navy blue background. The final image should be fragmented and abstract." 