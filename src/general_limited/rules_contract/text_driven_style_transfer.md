---
name: text_driven_style_transfer
description: Rules and guidelines for generating an image by directly combining content from a text prompt with the artistic style from a style image in a single step.
---

## Overview
(Critical Rule) 
The goal is to directly synthesize an image that faithfully represents the content from a user's text prompt, rendered in the specific artistic style of a provided style image. The process relies on a detailed analysis and a single, comprehensive generation plan.

## For Analyser
The analyser's primary role is to create a detailed **Style-to-Content Mapping**. This involves deconstructing both the style image and the text prompt and then creating a precise guide for how to render the text content in the given style.

1.  **Deconstruct the Style Image**: Analyze and list the core artistic elements of the style image. Be specific.
    -   **Composition**: How are elements arranged? (e.g., "A central, dominant portrait figure," "Asymmetrical balance").
    -   **Color Palette**: What are the key colors and their relationships? (e.g., "Vibrant, high-contrast blocks of blue, green, orange, and purple against a dark navy blue background").
    -   **Line Work & Shapes**: Describe the lines and shapes. (e.g., "Face and body constructed from hard-edged geometric shapes (squares, circles). Contours are defined by thin, golden-orange outlines. Facial features are rendered with simple, thick black lines.").
    -   **Texture & Brushwork**: Describe the surface quality. (e.g., "A coarse, canvas-like texture is visible throughout the image, especially in the background").
    -   **Overall Mood**: What is the feeling? (e.g., "Abstract, fragmented, modern cubist feel").

2.  **Deconstruct the Text Prompt**: Identify the key subjects, objects, and features described in the text.
    -   *Example Text*: "A portrait of a man."
    -   *Deconstruction*: Subject is "a man," which implies features like a face, eyes, nose, mouth, hair, and shoulders.

3.  **Create the Style-to-Content Mapping**: This is the most critical step. Create a detailed, explicit mapping of how each feature from the text prompt should be drawn using the elements from the style image.
    -   *Example Mapping for "a portrait of a man" in the provided style*:
        -   **Face Structure**: The man's face will not be realistic; it will be an abstract composition of interlocking geometric shapes and bold color blocks, mirroring the construction in the style image.
        -   **Eyes**: Render the eyes as simple, stylized black ovals or arcs, not as detailed eyeballs.
        -   **Nose**: Depict the nose with a single, minimalist black line or a simple geometric shape.
        -   **Mouth**: Represent the mouth as a single, bold-colored shape, not as realistic lips.
        -   **Hair**: The hair should be a large, solid block of color (e.g., blue or green), defined by the golden-orange outline, without individual strands.
        -   **Overall Form**: The entire portrait of the man should be outlined with the same thin, golden-orange lines against the textured, dark navy blue background.

## For Planner
The planner's role is to convert the analyser's detailed **Style-to-Content Mapping** into a single-stage, comprehensive generation prompt.

1.  **Degenerate the Plan**: The plan MUST consist of only **one stage**.
2.  **Synthesize a Comprehensive Prompt**: Combine all the details from the Style-to-Content Mapping into a single, rich `text_prompt`. This prompt should guide the generation model precisely on what to create.
    -   *Example Prompt based on the mapping above*: "An abstract, cubist-style portrait of a man, inspired by the provided reference image. Construct his face from vibrant, interlocking geometric color blocks (blue, green, orange, purple). Use simple, thick black lines for the eyes, nose, and mouth. The hair should be a large, solid block of color. Encase the entire portrait in thin, golden-orange outlines, set against a coarse, textured navy blue background. The final image should be fragmented and abstract." 