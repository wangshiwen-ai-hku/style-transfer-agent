---
name: material_transfer
description: Rules and guidelines for applying a material from one image to the content of another, preserving the content's structure.
---

## Overview
(Critical Rule)
1. MUST PRESERVE the **structure, shapes, and contours** of the content image.
2. MUST DISCARD all original material properties of the content image (e.g., color, lighting, shading) and replace them with the material properties of the material image.

For Analyser:
Analyze the material image's visual grammar using concise, professional adjectives. Avoid lengthy descriptions. The goal is to create a precise "style guide" for the Planner. Key aspects to analyze are:
-   **Line Style:** Describe the nature of the lines (e.g., "straight", "sharp-angled", "uniform thickness", "geometric", "parallel").
-   **Pattern Logic:** Describe the arrangement of elements (e.g., "grid-based", "repeating units", "interlocking", "rectilinear").
-   **Color Palette:** List the key colors (e.g., "dark green", "lime green", "gold").
-   **Material Finish**: Describe the surface quality (e.g., "flat", "matte", "no shading", "emissive dots").

For Planner:
Create a plan that strictly adheres to the overview rules. Emphasize these points:

-   **(Critical) Total Material Replacement:** The final image must derive all of its material properties from the material image. No material information from the content image should be preserved.
-   **(Critical) Re-create Structure with Material's Grammar:** This is a two-step process:
    1.  **Generate a Styled Sketch:** Do NOT generate a generic line art. Instead, create a new sketch of the content image that is drawn *from the start* using the material's line style (e.g., "straight", "sharp-angled"). This is the most critical step. The output should be a line drawing that has the content's structure but the material's line-art DNA. For example: "Create a line art sketch of the woman, but draw her using only the straight, sharp-angled, geometric lines characteristic of a circuit board."
    2.  **Apply Material Properties to Styled Sketch:** With the structure now correctly styled, apply the final material properties. This is like a coloring step. Instruct the generation process to fill the styled sketch with the material's colors, patterns, and finish (e.g., "Apply the green and gold colors, flat shading, and glowing dots of the circuit board material to the geometric line art.")
