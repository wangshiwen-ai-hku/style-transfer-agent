---
name: ip_generator
description: Rules and guidelines for generating a set of IP (Intellectual Property) character images with varying poses and actions, based on a target content description and a reference IP style.
---

## Overview
(Critical Rule) The core task is to learn the artistic style from a reference IP image and apply it to a new content description to generate a SET of new IP images. The generated images must maintain a consistent style but feature different poses, actions, or expressions.
(Critical Rule): MUST keep white or pure background.

For Analyser:
1.  **Analyze the Reference IP Style**: Deconstruct the provided reference IP image to understand its core artistic style. Focus on:
    *   **Line Art**: Is it thick, thin, sketchy, clean, minimalist, etc.?
    *   **Color Palette**: What are the dominant colors? Is the palette limited, vibrant, muted?
    *   **Shapes & Forms**: Are the shapes simple, geometric, organic, complex? How is volume represented?
    *   **Key Features**: Identify unique and recognizable features of the IP (e.g., specific eye shape, accessories, proportions).
2.  **Analyze the Target Content**: Understand the request for the new IP. What is the character, what are its key attributes?

For Planner:
Your goal is to create a multi-stage plan to generate a consistent set of IP images. The plan MUST follow a two-step structure to ensure consistency.

1.  **Generate a Base Image**: The very first stage of your plan must be to generate a canonical, base image of the IP.
    *   This stage should use the original reference IP image as input (e.g., `ip_image`).
    *   The prompt should be for a neutral, full-body, front-facing view of the character against a simple background.
    *   Give this generated image a clear tag, for example: `base_ip_character`.

2.  **Generate Variations**: All subsequent stages in the plan will generate the different poses, expressions, and views.
    *   (Critical) Every variation stage MUST use the `base_ip_character` image generated in the first stage as its `required_image_tags`. DO NOT use the original `ip_image`. This ensures all variations are consistent with the base design.
    *   **Expressive Poses**: For variations based on emotions (e.g., happy, sad), the `text_prompt` must also describe a corresponding body action or pose. A happy character might be jumping, a sad one might be curled up. Don't just change the face.
    *   Create a separate stage for each required variation (e.g., side view, happy expression, jumping pose).

Attention to these aspects, IF ANY in this case. tell to planner and emphasize it.
- (Critical) Create a core character design first. This design will be the base for all variations.
- Generate a list of variations. Each variation should be a complete prompt for generating one image with a unique pose, action, or expression.
- The final output from the planner should be a JSON object containing a list of `generation_tasks`.
- Each task should define a `text_prompt` for a specific variation (e.g. "A character in the [style name] style, jumping happily.") and a `generated_image_tag` (e.g. "ip_jumping").