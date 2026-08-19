---
name: style_fusion
description: Rules and guidelines for fusing the styles of two or more images to create a novel, coherent artistic style.
---

## Overview
(Critical Rule)
MUST KEEP the **balance, harmony, and structural logic** between the given styles. Avoid generating chaotic or messy textures.
For you: **Tailor the analysis focus** according to the user-provided **multiple style images** and their desired fusion strategy/ratio.

For Analyser:
1. **Independent Style Analysis**: Separately analyze the visual attributes of the provided styles. Identify their core elements like color palette, brush strokes, texture, spatial composition, geometric structure, and emotional tone. Discriminate between core styles and non-style content.
2. **Compatibility and Conflict Mapping**: Identify intersecting design choices where the styles can seamlessly merge, and identify clashes (e.g., highly detailed realism versus flat minimalism). 
3. **Fusion Strategy Design**: Define the weighting and allocation of the styles based on user requests (e.g., 70% Style A, 30% Style B). Determine exactly *what* aspects should be integrated from Style A (like lighting and color) and *what* aspects should be preserved from Style B (like motifs and texture details).

For Planner:
Attention to these aspects, IF ANY in this case. Tell the planner and emphasize it:
- (Critical) **Specify Style Roles Explicitly**: Make sure the `text_prompt` directly specifies how each style image is utilized (e.g., "Use the vibrant color palette and lighting of image 1, but adopt the intricate line-art of image 2").
- **Control Proportion via Prompt Weighting**: Advise using strong or supporting descriptive words for the elements of the blended styles based on the fusion ratio given by the user.
- **Resolve Structural Clashes**: Determine a spatial strategy if styles conflict fundamentally. Provide guidance to preserve Style A for the background style consistency, and apply Style B extensively to the foreground subjects.
- **Harmonize Semantic Motifs**: Explicitly instruct whether specific standalone decorations or motifs (flowers, icons) from the source styles should be discarded, heavily abstracted, or merged smoothly to maintain cohesion.
