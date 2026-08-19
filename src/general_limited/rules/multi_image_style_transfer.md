---
name: multi_image_style_transfer
description: Rules and guidelines for applying a consistent artistic style from one style image to multiple content images, ensuring consistency across them.
---

## Overview
(Critical Rule) 
MUST KEEP the **position and orientation and rotation** of the content in each image.
(Critical Rule for Multi-Image)
MUST IDENTIFY corresponding regions/objects across multiple content images and ensure **consistent style application** to these regions.

## For Analyser Agent(s):
1.  **Analyze the Style Image**: Analyze the style image from multi-aspects according to its special features.
    *   E.g., Color Palette, Texture, Brushstrokes, Line Art, Composition, Motifs/Decorations.
2.  **Analyze and Compare Content Images**:
    *   For each content image, identify main subjects and background.
    *   (Critical) Compare all content images to find corresponding elements, objects, or regions. For example, if two images feature the same person, identify the person as a corresponding element. If they are different scenes, look for similar types of regions (e.g., sky, trees, buildings).
    *   Describe the consistency relationship between the content images.

## For Planner Agent:
Based on the analysis, create a plan to apply the style consistently to ALL content images.
Attention to these aspects, IF ANY in this case. tell to planner and emphasize it.

1.  **(Critical) Ensure Consistency:** For the identified corresponding regions across content images, the generated style (e.g., color, texture, brushwork) MUST be consistent. The prompts for generating these regions should be very similar, if not identical.
2.  **Make Use of SketchDraft:** To remove the styles of content images and redraw the sketch as style image's line art and composition. Apply this step consistently if needed.
3.  **Care For Similar Region Transfer:** If the content and style have similar regions (e.g., flowers, faces, clothes), this applies to all content images.
4.  **Adjust Compositions:** If the style's composition is special (e.g., a lot of white space, totally geometric, boneless), apply composition adjustments consistently.
5.  **Add Special Decorations/Motifs:** If decorations/motifs (e.g., flowers, animals) are added, they should appear in a logically consistent manner across images.
6.  **Stylize Background:** Backgrounds (e.g., remove/stylize/substitute) should be stylized similarly to maintain overall coherence. 