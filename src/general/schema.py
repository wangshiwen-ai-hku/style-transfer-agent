from webbrowser import BackgroundBrowser
from mcp.types import TextResourceContents
from pydantic import BaseModel, Field
from typing import List, Dict, NotRequired, TypedDict, Optional
from enum import Enum


class TaskIdentifier(BaseModel):
    """Identifies the primary task from the user's request."""
    task_type: str = Field(..., description="The identified task type, e.g., 'style_transfer', 'style_fusion', 'potrait_repaint'.")


class SkillSelector(BaseModel):
    """Selects the most appropriate skill file to use for the task."""
    skill_file_to_read: str = Field(..., description="The full path of the skill file that should be read to complete the task, e.g., 'src/general/rules/style_transfer.md'.")


class MODALITY(str, Enum):
    TEXT = "text"
    IMAGE = "image"


class Message(BaseModel):
    """
    A message part, which can be text or an image.
    type: text or image
    content: prompt for text, or a label for an image.
    image_tag: The tag of the image to be used. Required if type is IMAGE.
    """
    type: MODALITY
    content: Optional[str] = None
    image_tag: Optional[str] = None


class ToolCall(BaseModel):
    """
    Represents a call to a specific tool.
    """
    tool_name: str = Field(..., description="The name of the tool to be called, e.g., 'canny_edge_detector'.")
    tool_input: Dict = Field(..., description="The dictionary of arguments for the tool.")


class Stage(BaseModel):
    """
    Represents a single stage in the style transfer process.
    Each stage generates one image, either by calling a tool or by using a generative model.
    """
    stage_name: str = Field(..., description="A short name for this stage.")
    generated_image_tag: str = Field(..., description="The tag for the image generated in this stage. This tag can be referenced in subsequent stages.")
    required_image_tags: List[str] = Field(..., description="The tags for the images used in this stage. This tags can be referenced in subsequent stages.")
    text_prompt: str = Field(..., description="The text prompt for the stage.")
    gen_temperature: Optional[float] = Field(None, description="The temperature for the multi-modal imagen model.")


class FunctionAgentConfig(BaseModel):
    """
    Configuration for a single, specialized function agent.
    """
    agent_name: str = Field(..., description="The name of the agent")
    dependencies: List[str] = Field([], description="A list of agent names that this agent depends on.")
    temperature: float = Field(0.7, description="The temperature for the agent's LLM.")
    prompt: str = Field(..., description="The system prompt that defines the agent's task. It should instruct the agent to provide its output in a structured JSON format.")
    required_image_tags: List[str] = Field(..., description="A list of image tags that this agent requires as input, e.g., ['image_1'].")


class SystemOrchestration(BaseModel):
    """
    Defines the dynamically generated multi-agent system for image analysis based on user intent.
    """
    task_type: str = Field(..., description="The type of the task.")
    agent_graph: List[FunctionAgentConfig] = Field(..., description="A directed acyclic graph of specialized agents to perform analysis and planning.")
    result_critique_criteria: str = Field(..., description="The critique criteria for the task, tailored to the specific task.")

class StyleAnalysis(BaseModel):
    """ 
    A multi-dimensional analysis of the style image.
    """
    color_palette: str = Field(..., description="Description of the color palette, including dominant colors, accent colors, and overall saturation level (e.g., muted, vibrant, pastel).")
    brushwork_and_texture: str = Field(..., description="Analysis of the brushwork, texture, and line art style.")
    shapes_and_geometry: str = Field(..., description="Analysis of the shapes, geometry, and other special geometric patterns of the image.")
    lighting_and_shadow: str = Field(..., description="Description of the lighting, shadows, and overall contrast (e.g., high-contrast, soft).")
    artistic_style_and_genre: str = Field(..., description="Identification of the artistic style or genre, if recognizable (e.g., 'Impressionism', 'Art Nouveau', 'Japanese Woodblock Print'). If not applicable or recognizable, return an empty string.")
    key_elements_and_motifs: str = Field(..., description="Description of any recurring or key elements, symbols, or motifs that define the style. If not applicable, return an empty string.")
  
class StyleAuthorAnalysis(BaseModel):
    """
    A short definition of the possible author context of the work, such as ,childlike, experienced ink type, etc.
    """
    author: str = Field(..., description="Use words to describe the possible author tone of the work, such as ,childlike, ink drawer, etc.")
    draw_techniques: str = Field(..., description="Draw techniques used ")

class StylizeStagesHints(BaseModel):
    """
    Important stages in style transfer process, based on the `create_process` of style author and the distinct style features.
    may include sketchize, blur, ink wash, geomtric, .. actions, 
    At MOST 3 distinct actions.
    """
    actions_sequences: List[str] = Field(..., description="The sequences of the actions to be performed in the style transfer process., such as, blur, ink, sketch, ...")
    actions_objects: List[str] = Field(..., description="The objects to take this action in st process.")


class ImageAnalysis(BaseModel):
    style_image_analysis: StyleAnalysis
    style_author_analysis: StyleAuthorAnalysis
    content_image_description: str
    stylize_stages_hints: StylizeStagesHints
    similiar_regions_transfer_detail: str = Field(..., description="If the style image and content image have similar content region, MUST transfer this region one to one detaily, e.g. , eyebrow, hair, etc.")

class GeneralStyleAttributes(BaseModel):
    """
    General style attributes, such as, color palette, brushwork, texture, lighting, etc.
    Use terms to describe the style attributes, not sequence.
    """
    line_type: str = Field(..., description="The type of the line, such as, thick line, thin line, etc.")
    color_palettes: str = Field(..., description="The color palette of the style image, such as, muted, vibrant, pastel, etc.")
    texture: str = Field(..., description="The texture of the style image, such as, smooth, rough, etc.")
    mood: str = Field(..., description="The mood of the style image, such as, happy, sad, etc.")

class PersonalStyleAttributes(BaseModel):
    """
    Personal style attributes, such as, special stylized motifs, brush texture, composition feature, geometry feature, highly contrast ratio... summarize as terms, not sequence.
    """
    special_stylized_motifs: str = Field(..., description="Special stylized motifs, such as, bold lines, flat colors, etc.")
    brush: str = Field(..., description="Brush texture, such as, thick lines, thin lines, etc.")
    composition: str = Field(..., description="Composition feature, such as, balanced composition, asymmetric composition, etc.")
    background: str = Field(..., description="Background style, motifs, If Any")
    geometry: str = Field(..., description="Geometry feature, such as, round shapes, square shapes, etc.")
    styled_decorations: str = Field(..., description="Styled Motifs, such as, flowers, stars, etc, that irrevant to the style image content,")
    highly_specialized_features: str = Field(..., description="Highly specialized features, such as, extreme big head and small body")

class GuessAsAnAuthor(BaseModel):
    """
    Guess the author of the style image. and the create techniques and process. Be detail in create_draft_features and create_process.
    """
    author_type: str = Field(..., description="The type of the author, such as, childlike, experienced ink type, etc.")
    techniques: str = Field(..., description="Draw techniques used ")
    draft_features: str = Field(..., description="Draft features of the style image, like lines, shapes, abstraction, composition. Be detail in the features.")
    create_process: str = Field(..., description="Draw process, such as, sketch, color, etc. Be detail in the process.")


class ComprehensiveStyleAnalysis(BaseModel):
    """
    A comprehensive analysis of the style image. Be concise.
    guess_as_an_author include author type, draw techniques, draw process
    """
    style_image_type: str = Field(..., description="The type of the style image, such as, painting, photo, digital_art, etc.")
    general_styles: GeneralStyleAttributes
    personal_styles: PersonalStyleAttributes
    guess_as_an_author: GuessAsAnAuthor
    style_image_content_to_remove: str = Field(..., description="The content of the style image, such as, a portrait of a woman, a landscape, a city, that must be removed in the style")


class StyleTransferAnalysis(BaseModel):
    """
    An analysis of how to transfer the style to the content image, respecting the content's structure.
    """
    content_image_description: str = Field(..., description="A brief description of the content image, focusing on key elements to preserve.")
    stylization_hints: StylizeStagesHints
    similar_regions_transfer_detail: str = Field(..., description="Provide ordered descriptions of `1. object1: transfer1 xxx. 2. object2: transfer2 xxx. ...`. If there are similar objects or regions (e.g., face, hair, eyes) between the content and style images, describe in detail how to specifically transfer the style for those regions, preserving the content's pose and position.")

 

class StyleTransferPlan(BaseModel):    
    stages: List[Stage]


class Reflection(BaseModel):
    """
    A reflection on the style transfer result for a single stage.
    """
    critique: str = Field(..., description="A detailed critique of the last generated image, comparing it against the style image and the content image. Mention what was done well and what needs improvement.")
    # content_hold_score: float = Field(..., description="A score from 0.0 to 10.0 indicating how well the content is preserved. 10.0 is a perfect match.")
    # style_transfer_score: float = Field(..., description="A score from 0.0 to 10.0 indicating how well the style was transferred. 10.0 is a perfect match.")
    is_satisfied: bool = Field(..., description="Set to true if the result is satisfactory and no more stages are needed.")


class State(TypedDict):
    project_dir: str

    image_paths: List[str]
    provided_images: List[str]  # for multi-image tasks
    system_orchestration: NotRequired[SystemOrchestration]
    analysis_context: NotRequired[Dict[str, str]]

    style_transfer_plan: NotRequired[StyleTransferPlan]
    generated_images_map: Dict[str, str]
    user_prompt: str
    directly: Optional[bool]
    reflection_count: int
    reflections: List[Reflection]




    