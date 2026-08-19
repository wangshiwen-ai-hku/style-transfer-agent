---
name: high_abstract_style_transfer
description: Rules for re-imagining a photograph in a highly abstract style using text-guided generation.
---
## Overview

(Critical Rule) This is an artistic re-interpretation, not a direct style transfer. The goal is to heavily abstract the content image, preserving only the basic subject and pose.
(Critical Workflow) The style image is often too abstract for direct use. It MUST be converted into a detailed text description, which will then be the sole guide for transforming the content image.

### For Analyser:

(Prerequisite) The success of this method depends on an *exhaustive* analysis of the style image. You must deconstruct the artist's process and core visual elements to generate a highly detailed and descriptive text prompt. This prompt must capture:

- **Inferred Drawing Process:** Speculate on how the artwork was created. Was it layered? Did lines come before or after color? This informs the generation steps.
- **Geometric Abstraction:** How the subject is deconstructed into specific geometric and organic forms (e.g., "a face made of colorful rectangles and circles").
- **Color Palette & Theory:** The key colors, their relationships, vibrancy, and application (e.g., "a palette of bold, flat, high-contrast colors like cobalt blue, crimson, and vibrant green, applied with no blending").
- **Line Art Characteristics:** The style, weight, quality, and specific shapes of lines (e.g., "clean, confident, geometric curves," "thick, expressive black outlines for key features," "thin, straight orange lines for framing").
- **Depth & Texture:** How depth is implied (or flattened) and any underlying canvas or background textures (e.g., "a flat composition with a subtle, distressed canvas texture throughout").

### For Planner:

Based on the Analyser's detailed prompt, you MUST design a multi-stage plan to progressively transform the content image. Each stage should apply a specific style element, using the text description as its guide. The plan must be a sequence, not a single step. A typical plan would be:

1. **Stage 1: Structural Abstraction:** Generate a new base image by re-interpreting the content photo with the style's geometric and compositional rules. This establishes the new foundation. (e.g., "Create a version of the portrait using only flat, interlocking geometric shapes.")
2. **Stage 2: Color Application:** Using the abstracted base, inpaint or re-generate to apply the style's color palette and theory. (e.g., "Fill the geometric shapes with the described bold, flat colors.")
3. **Stage 3: Line Work:** Add the characteristic line art over the colored base. This could be a separate step for outlines and another for finer details. (e.g., "Draw the thick, expressive black outlines for the eyes and mouth.")
4. **Stage 4: Texture and Refinement:** Apply the final texture and any other subtle details to complete the piece. (e.g., "Overlay the entire image with the subtle, distressed canvas texture.")
