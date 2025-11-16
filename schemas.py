"""
Database Schemas for Crop to Nutrition to Patients

Each Pydantic model below maps to a MongoDB collection (lowercased class name).
We use these for validation and persistence through helper functions in database.py
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Literal, Dict, Any
from datetime import datetime

Role = Literal["farmer", "organization", "mediator", "delivery", "admin"]

class GeoPoint(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)

class User(BaseModel):
    name: str
    email: EmailStr
    password_hash: str
    role: Role
    phone: Optional[str] = None
    organization_name: Optional[str] = None
    location: GeoPoint
    address: Optional[str] = None
    is_active: bool = True

class Crop(BaseModel):
    name: str
    nutrients: Dict[str, float] = Field(default_factory=dict, description="macro/micro per 100g")

class Surplus(BaseModel):
    farmer_id: str
    crop_name: str
    quantity_kg: float = Field(..., gt=0)
    harvest_date: datetime
    location: GeoPoint
    image_url: Optional[str] = None
    status: Literal["open", "locked", "fulfilled"] = "open"

class Demand(BaseModel):
    organization_id: str
    crop_preferences: List[str] = Field(default_factory=list)
    nutrition_requirements: Dict[str, float] = Field(default_factory=dict)
    quantity_kg: float = Field(..., gt=0)
    required_by: datetime
    location: GeoPoint
    status: Literal["open", "locked", "fulfilled"] = "open"

class Match(BaseModel):
    surplus_id: str
    demand_id: str
    mediator_id: str
    distance_km: float
    score: float
    status: Literal["locked", "fulfilled", "cancelled"] = "locked"

class Delivery(BaseModel):
    match_id: str
    delivery_agent_id: str
    pickup_time: Optional[datetime] = None
    drop_time: Optional[datetime] = None
    pod_image_url: Optional[str] = None
    status: Literal["assigned", "picked", "delivered"] = "assigned"

class Notification(BaseModel):
    user_id: str
    type: str
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    read: bool = False
