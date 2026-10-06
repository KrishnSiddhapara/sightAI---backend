from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict

class InstanceAttributes(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    clothing: Optional[str] = Field(default="not clearly visible", description="Clothing type/description for person (e.g. 'red t-shirt and jeans'). Use 'not clearly visible' if unknown.")
    clothing_color: Optional[str] = Field(default="not clearly visible", description="Dominant clothing colors belonging ONLY to this specific person instance.")
    pose: Optional[str] = Field(default="unknown", description="Visible pose (e.g. 'standing', 'sitting', 'running', 'unknown').")
    action: Optional[str] = Field(default="unknown", description="Visible action/activity performed by this entity.")
    accessories: Optional[str] = Field(default="none visible", description="Visible accessories (e.g. 'glasses, backpack', 'none visible').")
    object_color: Optional[str] = Field(default="not clearly visible", description="Dominant color of this non-person object instance.")
    type_or_subtype: Optional[str] = Field(default="unknown", description="Specific type or visual characteristic (e.g. 'sedan', 'mountain bike', 'coniferous fir').")
    visible_details: Optional[str] = Field(default="none noted", description="Notable visible characteristics belonging strictly to this single physical instance.")

class ImageMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    original_width: int = Field(default=0, description="Original image width in pixels after EXIF normalization.")
    original_height: int = Field(default=0, description="Original image height in pixels after EXIF normalization.")
    analysis_width: int = Field(default=0, description="Width of the preprocessed image in pixels sent to VLM.")
    analysis_height: int = Field(default=0, description="Height of the preprocessed image in pixels sent to VLM.")
    scale_x: float = Field(default=1.0, description="Horizontal scale factor from original to analysis image.")
    scale_y: float = Field(default=1.0, description="Vertical scale factor from original to analysis image.")
    offset_x: int = Field(default=0, description="Horizontal crop/padding offset.")
    offset_y: int = Field(default=0, description="Vertical crop/padding offset.")
    rotation: int = Field(default=0, description="EXIF rotation applied in degrees.")

class BoundingBox(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    x_min: float = Field(default=0.0, description="Normalized X minimum coordinate (0 to 1000 scale, left edge).")
    y_min: float = Field(default=0.0, description="Normalized Y minimum coordinate (0 to 1000 scale, top edge).")
    x_max: float = Field(default=1000.0, description="Normalized X maximum coordinate (0 to 1000 scale, right edge).")
    y_max: float = Field(default=1000.0, description="Normalized Y maximum coordinate (0 to 1000 scale, bottom edge).")

    @field_validator('x_min', 'y_min', 'x_max', 'y_max', mode='before')
    @classmethod
    def parse_coordinate(cls, v):
        if v is None:
            return 0.0
        try:
            fv = float(v)
            return max(0.0, min(1000.0, round(fv, 2)))
        except (ValueError, TypeError):
            return 0.0

class ObjectInstance(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str = Field(description="Unique instance identifier (e.g. 'person_1', 'person_2', 'car_1').")
    attributes: InstanceAttributes = Field(default_factory=InstanceAttributes, description="Independent attributes strictly belonging to this single visual instance.")
    bounding_box: Optional[BoundingBox] = Field(default=None, description="Normalized 0-1000 bounding box coordinates [x_min, y_min, x_max, y_max] surrounding this specific instance. Return null if localization is uncertain.")
    uncertainty_reason: Optional[str] = Field(default=None, description="Explicit reason if identification or count of this instance is uncertain (e.g. 'Partially occluded behind tree').")

class DetectedObjectCategory(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    name: str = Field(description="Normalized category name in lowercase (e.g. 'person', 'car', 'dog', 'bottle', 'bicycle'). Standardize synonyms like 'automobile' to 'car'.")
    confirmed_count: int = Field(default=1, description="Exact count of clearly visible physical instances in this category. Do not double count or count reflections/posters/shadows.")
    uncertain_count: int = Field(default=0, description="Count of partially visible or unconfirmed instances that cannot be conclusively verified.")
    instances: List[ObjectInstance] = Field(default_factory=list, description="Independent instance objects for each detected physical entity.")

class SceneDescription(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    environment: str = Field(default="Environment observed", description="Specific environment description (e.g., 'Indoor office space with wooden desks'). Do not invent specific named venues unless visually proven.")
    primary_activity: str = Field(default="Primary activity observed", description="Primary activity occurring in the scene.")
    summary: str = Field(default="Visual summary of the scene", description="Concise visual summary of the scene.")

class GroundedAnalysisResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    image_metadata: Optional[ImageMetadata] = Field(default=None, description="Canonical source image dimension & transformation metadata.")
    objects: List[DetectedObjectCategory] = Field(default_factory=list, description="List of detected object categories with verified physical counts and independent instance attributes.")
    scene: SceneDescription = Field(default_factory=SceneDescription, description="Scene-level understanding grounded strictly in visible evidence.")
    overall_summary: str = Field(default="Executive summary of the image.", description="Executive summary built strictly from the verified structured objects and scene analysis.")

class VersionRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    version_id: str = Field(description="Unique version identifier e.g. 'v0', 'v1'")
    version_number: int = Field(description="Numeric version sequence index")
    created_at: str = Field(description="ISO 8601 server timestamp")
    updated_at: str = Field(description="ISO 8601 server timestamp")
    formatted_time: str = Field(description="Human-readable formatted timestamp")
    edit_prompt: str = Field(description="Edit prompt or creation description")
    source_version_number: int = Field(default=0, description="Parent version number this edit was based on")
    parent_version_id: Optional[str] = Field(default="v0", description="Parent version identifier")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded image string")

class SourceItem(BaseModel):
    title: str = Field(description="Title of the web source or page.")
    url: str = Field(description="Validated external URL.")
    domain: str = Field(description="Domain name (e.g. 'amazon.com', 'jamesclear.com').")
    source_type: str = Field(default="general", description="Source category: official, retailer, publisher, reference, general.")
    snippet: Optional[str] = Field(default=None, description="Short summary snippet from the source.")

class FactItem(BaseModel):
    label: str = Field(description="Fact title e.g. 'Listed Price', 'Official Publisher', 'Processor', 'Availability'.")
    value: str = Field(description="Extracted value e.g. '₹49,999', 'Avery (Penguin)', 'Intel Core i7-13700H', 'In Stock'.")
    source_title: Optional[str] = Field(default=None, description="Supporting source name.")

class ProductEntity(BaseModel):
    name: Optional[str] = Field(default=None, description="Product or entity title.")
    brand: Optional[str] = Field(default=None, description="Brand or manufacturer.")
    model: Optional[str] = Field(default=None, description="Model designation/number.")
    variant: Optional[str] = Field(default=None, description="Variant details e.g. 256GB, Red, Edition.")
    category: Optional[str] = Field(default=None, description="Category e.g. smartphone, book, laptop, vehicle, camera.")
    confidence: float = Field(default=1.0, description="Confidence score 0.0 - 1.0.")

class AgentResearchRequest(BaseModel):
    question: str = Field(description="User natural language question.")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded image data for visual Q&A.")
    image_context: Optional[dict] = Field(default=None, description="Existing visual analysis context (objects, scene, summary).")
    conversation_history: Optional[List[dict]] = Field(default=[], description="Previous conversation turn history.")
    user_region: Optional[str] = Field(default=None, description="Optional country/region preference.")

class AgentResearchResponse(BaseModel):
    success: bool = True
    answer: str = Field(description="Grounded response synthesized by the Research Agent.")
    intent: str = Field(default="GENERAL_KNOWLEDGE", description="Detected query intent.")
    requires_research: bool = Field(description="Whether external web research tools were executed.")
    entity: Optional[ProductEntity] = Field(default=None, description="Extracted product/entity details.")
    facts: List[FactItem] = Field(default=[], description="Structured key facts extracted from research.")
    used_tools: List[str] = Field(default=[], description="List of executed tool names.")
    sources: List[SourceItem] = Field(default=[], description="External sources and reference links.")
    confidence: str = Field(default="high", description="Confidence level: high, medium, low.")
    research_summary: Optional[str] = Field(default=None, description="User-facing concise research summary.")
    error: Optional[str] = Field(default=None, description="Error message if research failed.")
