from typing import List, Optional
from pydantic import BaseModel, Field

class InstanceAttributes(BaseModel):
    clothing: Optional[str] = Field(default="not clearly visible", description="Clothing type/description for person (e.g. 'red t-shirt and jeans'). Use 'not clearly visible' if unknown.")
    clothing_color: Optional[str] = Field(default="not clearly visible", description="Dominant clothing colors belonging ONLY to this specific person instance.")
    pose: Optional[str] = Field(default="unknown", description="Visible pose (e.g. 'standing', 'sitting', 'running', 'unknown').")
    action: Optional[str] = Field(default="unknown", description="Visible action/activity performed by this entity.")
    accessories: Optional[str] = Field(default="none visible", description="Visible accessories (e.g. 'glasses, backpack', 'none visible').")
    object_color: Optional[str] = Field(default="not clearly visible", description="Dominant color of this non-person object instance.")
    type_or_subtype: Optional[str] = Field(default="unknown", description="Specific type or visual characteristic (e.g. 'sedan', 'mountain bike', 'coniferous fir').")
    visible_details: Optional[str] = Field(default="none noted", description="Notable visible characteristics belonging strictly to this single physical instance.")

class BoundingBox(BaseModel):
    x_min: int = Field(ge=0, le=1000, description="Normalized X minimum coordinate (0 to 1000 scale, left edge).")
    y_min: int = Field(ge=0, le=1000, description="Normalized Y minimum coordinate (0 to 1000 scale, top edge).")
    x_max: int = Field(ge=0, le=1000, description="Normalized X maximum coordinate (0 to 1000 scale, right edge).")
    y_max: int = Field(ge=0, le=1000, description="Normalized Y maximum coordinate (0 to 1000 scale, bottom edge).")

class ObjectInstance(BaseModel):
    id: str = Field(description="Unique instance identifier (e.g. 'person_1', 'person_2', 'car_1').")
    attributes: InstanceAttributes = Field(description="Independent attributes strictly belonging to this single visual instance.")
    bounding_box: Optional[BoundingBox] = Field(default=None, description="Normalized 0-1000 bounding box coordinates [x_min, y_min, x_max, y_max] surrounding this specific instance. Return null if localization is uncertain.")
    uncertainty_reason: Optional[str] = Field(default=None, description="Explicit reason if identification or count of this instance is uncertain (e.g. 'Partially occluded behind tree').")

class DetectedObjectCategory(BaseModel):
    name: str = Field(description="Normalized category name in lowercase (e.g. 'person', 'car', 'dog', 'bottle', 'bicycle'). Standardize synonyms like 'automobile' to 'car'.")
    confirmed_count: int = Field(description="Exact count of clearly visible physical instances in this category. Do not double count or count reflections/posters/shadows.")
    uncertain_count: int = Field(default=0, description="Count of partially visible or unconfirmed instances that cannot be conclusively verified.")
    instances: List[ObjectInstance] = Field(default=[], description="Independent instance objects for each detected physical entity.")

class SceneDescription(BaseModel):
    category: str = Field(description="Primary scene category (e.g. Indoor, Outdoor, Street, Office, Classroom, Home, Restaurant, Sports, Nature, Beach, Document, Other).")
    environment: str = Field(description="Specific environment description (e.g., 'Outdoor grassy park field with paved walkway'). Do not invent specific named venues unless visually proven.")
    primary_activity: str = Field(description="Primary activity occurring in the scene.")
    summary: str = Field(description="Concise visual summary of the scene.")

class GroundedAnalysisResult(BaseModel):
    objects: List[DetectedObjectCategory] = Field(description="List of detected object categories with verified physical counts and independent instance attributes.")
    scene: SceneDescription = Field(description="Scene-level understanding grounded strictly in visible evidence.")
    overall_summary: str = Field(description="Executive summary built strictly from the verified structured objects and scene analysis.")
