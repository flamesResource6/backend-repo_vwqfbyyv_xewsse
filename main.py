import os
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import jwt
from passlib.context import CryptContext

from database import db, create_document, get_documents
from schemas import User as UserSchema, Surplus as SurplusSchema, Demand as DemandSchema, Match as MatchSchema, Delivery as DeliverySchema, Notification as NotificationSchema, GeoPoint

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
JWT_ALGO = "HS256"
TOKEN_EXPIRE_MIN = 60 * 24 * 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

app = FastAPI(title="Crop to Nutrition to Patients API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Utils ------------------

BENGALURU_CENTER = {"lat": 12.9716, "lng": 77.5946}
RADIUS_KM = 50.0


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def within_bengaluru_radius(point: Dict[str, float]) -> bool:
    return (
        haversine_km(point["lat"], point["lng"], BENGALURU_CENTER["lat"], BENGALURU_CENTER["lng"]) \
        <= RADIUS_KM
    )


# ------------------ Auth ------------------

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str
    phone: Optional[str] = None
    organization_name: Optional[str] = None
    address: Optional[str] = None
    lat: float
    lng: float


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


def create_token(user: Dict[str, Any]) -> str:
    payload = {
        "sub": str(user.get("_id")),
        "email": user["email"],
        "role": user["role"],
        "name": user.get("name"),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRE_MIN),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        email = payload.get("email")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = db["user"].find_one({"email": email})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


def require_role(roles: List[str]):
    def checker(user=Depends(get_current_user)):
        if user.get("role") not in roles:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user

    return checker


@app.post("/auth/register", response_model=TokenResponse)
def register(payload: RegisterRequest):
    email = payload.email.lower()
    if db["user"].find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    if payload.role not in ["farmer", "organization", "mediator", "delivery", "admin"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    if not within_bengaluru_radius({"lat": payload.lat, "lng": payload.lng}):
        raise HTTPException(status_code=400, detail="Service limited to 50km around Bengaluru")

    user_doc = UserSchema(
        name=payload.name,
        email=email,
        password_hash=pwd_context.hash(payload.password),
        role=payload.role,
        phone=payload.phone,
        organization_name=payload.organization_name,
        location=GeoPoint(lat=payload.lat, lng=payload.lng),
        address=payload.address,
        is_active=True,
    ).model_dump()

    user_id = db["user"].insert_one(user_doc).inserted_id
    user_doc["_id"] = user_id
    token = create_token(user_doc)
    user_safe = {k: v for k, v in user_doc.items() if k not in ["password_hash"]}
    return TokenResponse(access_token=token, user=user_safe)


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest):
    email = payload.email.lower()
    user = db["user"].find_one({"email": email})
    if not user or not pwd_context.verify(payload.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(user)
    user_safe = {k: v for k, v in user.items() if k not in ["password_hash"]}
    return TokenResponse(access_token=token, user=user_safe)


# ------------------ Surplus & Demand ------------------

class SurplusCreate(BaseModel):
    crop_name: str
    quantity_kg: float
    harvest_date: datetime
    lat: float
    lng: float
    image_url: Optional[str] = None


@app.post("/surplus")
def create_surplus(payload: SurplusCreate, user=Depends(require_role(["farmer"]))):
    if not within_bengaluru_radius({"lat": payload.lat, "lng": payload.lng}):
        raise HTTPException(status_code=400, detail="Location outside 50km Bengaluru radius")
    doc = SurplusSchema(
        farmer_id=str(user["_id"]),
        crop_name=payload.crop_name,
        quantity_kg=payload.quantity_kg,
        harvest_date=payload.harvest_date,
        location=GeoPoint(lat=payload.lat, lng=payload.lng),
        image_url=payload.image_url,
        status="open",
    ).model_dump()
    sid = db["surplus"].insert_one(doc).inserted_id
    return {"id": str(sid), "message": "Surplus created"}


@app.get("/surplus")
def list_surplus(status: str = "open"):
    items = get_documents("surplus", {"status": status})
    result = []
    for it in items:
        loc = it.get("location", {})
        if loc and within_bengaluru_radius(loc):
            it["_id"] = str(it["_id"])  # stringify id
            result.append(it)
    return result


class DemandCreate(BaseModel):
    crop_preferences: List[str] = []
    nutrition_requirements: Dict[str, float] = {}
    quantity_kg: float
    required_by: datetime
    lat: float
    lng: float


@app.post("/demand")
def create_demand(payload: DemandCreate, user=Depends(require_role(["organization"]))):
    if not within_bengaluru_radius({"lat": payload.lat, "lng": payload.lng}):
        raise HTTPException(status_code=400, detail="Location outside 50km Bengaluru radius")
    doc = DemandSchema(
        organization_id=str(user["_id"]),
        crop_preferences=payload.crop_preferences,
        nutrition_requirements=payload.nutrition_requirements,
        quantity_kg=payload.quantity_kg,
        required_by=payload.required_by,
        location=GeoPoint(lat=payload.lat, lng=payload.lng),
        status="open",
    ).model_dump()
    did = db["demand"].insert_one(doc).inserted_id
    return {"id": str(did), "message": "Demand created"}


@app.get("/demand")
def list_demands(status: str = "open"):
    items = get_documents("demand", {"status": status})
    result = []
    for it in items:
        loc = it.get("location", {})
        if loc and within_bengaluru_radius(loc):
            it["_id"] = str(it["_id"])  # stringify id
            result.append(it)
    return result


# ------------------ Matching & Locking ------------------

class LockRequest(BaseModel):
    surplus_id: str
    demand_id: str


@app.post("/match/lock")
def lock_match(payload: LockRequest, user=Depends(require_role(["mediator", "admin"]))):
    from bson import ObjectId

    surplus = db["surplus"].find_one({"_id": ObjectId(payload.surplus_id)})
    demand = db["demand"].find_one({"_id": ObjectId(payload.demand_id)})
    if not surplus or not demand:
        raise HTTPException(status_code=404, detail="Surplus or Demand not found")
    if surplus["status"] != "open" or demand["status"] != "open":
        raise HTTPException(status_code=400, detail="Already locked or fulfilled")

    s_loc = surplus["location"]
    d_loc = demand["location"]
    dist = haversine_km(s_loc["lat"], s_loc["lng"], d_loc["lat"], d_loc["lng"])
    if dist > RADIUS_KM:
        raise HTTPException(status_code=400, detail="Outside 50km radius")

    # basic nutrient similarity (dot product overlap)
    s_crop = surplus["crop_name"].lower()
    prefs = [c.lower() for c in demand.get("crop_preferences", [])]
    crop_score = 1.0 if (not prefs or s_crop in prefs) else 0.5

    nut_req: Dict[str, float] = demand.get("nutrition_requirements", {})
    nut_score = 1.0 if not nut_req else min(1.0, surplus.get("quantity_kg", 0) / max(1.0, demand["quantity_kg"]))
    score = crop_score * 0.6 + (1.0 - min(dist / RADIUS_KM, 1.0)) * 0.2 + nut_score * 0.2

    # Lock both
    db["surplus"].update_one({"_id": surplus["_id"]}, {"$set": {"status": "locked"}})
    db["demand"].update_one({"_id": demand["_id"]}, {"$set": {"status": "locked"}})

    match_doc = MatchSchema(
        surplus_id=str(surplus["_id"]),
        demand_id=str(demand["_id"]),
        mediator_id=str(user["_id"]),
        distance_km=round(dist, 2),
        score=round(score, 3),
        status="locked",
    ).model_dump()
    mid = db["match"].insert_one(match_doc).inserted_id

    # Notify
    for uid in [surplus["farmer_id"], demand["organization_id"]]:
        notif = NotificationSchema(
            user_id=str(uid),
            type="match_locked",
            message="A supply-demand match has been locked",
            metadata={"match_id": str(mid)},
        ).model_dump()
        db["notification"].insert_one(notif)

    return {"id": str(mid), "distance_km": round(dist, 2), "score": round(score, 3)}


# ------------------ Delivery ------------------

class AssignDeliveryRequest(BaseModel):
    match_id: str
    delivery_agent_id: str


@app.post("/delivery/assign")
def assign_delivery(payload: AssignDeliveryRequest, user=Depends(require_role(["mediator", "admin"]))):
    from bson import ObjectId

    match = db["match"].find_one({"_id": ObjectId(payload.match_id)})
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    doc = DeliverySchema(
        match_id=str(match["_id"]),
        delivery_agent_id=payload.delivery_agent_id,
        status="assigned",
    ).model_dump()
    did = db["delivery"].insert_one(doc).inserted_id

    notif = NotificationSchema(
        user_id=payload.delivery_agent_id,
        type="delivery_assigned",
        message="You have a new delivery assignment",
        metadata={"delivery_id": str(did)},
    ).model_dump()
    db["notification"].insert_one(notif)

    return {"id": str(did), "message": "Delivery assigned"}


class ProofRequest(BaseModel):
    delivery_id: str
    pod_image_url: str


@app.post("/delivery/proof")
def upload_proof(payload: ProofRequest, user=Depends(require_role(["delivery"]))):
    from bson import ObjectId

    delivery = db["delivery"].find_one({"_id": ObjectId(payload.delivery_id)})
    if not delivery:
        raise HTTPException(status_code=404, detail="Delivery not found")
    if str(user["_id"]) != delivery["delivery_agent_id"]:
        raise HTTPException(status_code=403, detail="Not your delivery")

    db["delivery"].update_one(
        {"_id": delivery["_id"]},
        {"$set": {"pod_image_url": payload.pod_image_url, "status": "delivered", "drop_time": datetime.now(timezone.utc)}},
    )

    # mark match fulfilled and unlock statuses
    match = db["match"].find_one({"_id": delivery["_id"]})

    return {"message": "Proof uploaded"}


@app.post("/status/fulfill")
def fulfill_match(match_id: str, user=Depends(require_role(["mediator", "admin"]))):
    from bson import ObjectId
    m = db["match"].find_one({"_id": ObjectId(match_id)})
    if not m:
        raise HTTPException(status_code=404, detail="Match not found")
    db["match"].update_one({"_id": m["_id"]}, {"$set": {"status": "fulfilled"}})

    # set surplus & demand fulfilled
    db["surplus"].update_one({"_id": ObjectId(m["surplus_id"])}, {"$set": {"status": "fulfilled"}})
    db["demand"].update_one({"_id": ObjectId(m["demand_id"])}, {"$set": {"status": "fulfilled"}})

    return {"message": "Match fulfilled"}


# ------------------ Notifications ------------------

@app.get("/notifications")
def get_notifications(user=Depends(get_current_user)):
    items = list(db["notification"].find({"user_id": str(user["_id"])}))
    for it in items:
        it["_id"] = str(it["_id"])
    return items


# ------------------ ML Microservice (simple in-app stubs) ------------------

class RecoRequest(BaseModel):
    limit: int = 10


@app.post("/reco")
def recommend(payload: RecoRequest):
    # find best matches for open surplus and open demand within 50km
    surplus_list = list(db["surplus"].find({"status": "open"}))
    demand_list = list(db["demand"].find({"status": "open"}))
    recos: List[Dict[str, Any]] = []
    for s in surplus_list:
        for d in demand_list:
            s_loc = s.get("location", {})
            d_loc = d.get("location", {})
            if not s_loc or not d_loc:
                continue
            dist = haversine_km(s_loc["lat"], s_loc["lng"], d_loc["lat"], d_loc["lng"])
            if dist > RADIUS_KM:
                continue
            s_crop = s["crop_name"].lower()
            prefs = [c.lower() for c in d.get("crop_preferences", [])]
            crop_score = 1.0 if (not prefs or s_crop in prefs) else 0.4
            nut_req: Dict[str, float] = d.get("nutrition_requirements", {})
            nut_score = 1.0 if not nut_req else min(1.0, s.get("quantity_kg", 0) / max(1.0, d["quantity_kg"]))
            score = crop_score * 0.6 + (1.0 - min(dist / RADIUS_KM, 1.0)) * 0.2 + nut_score * 0.2
            recos.append({
                "surplus_id": str(s["_id"]),
                "demand_id": str(d["_id"]),
                "distance_km": round(dist, 2),
                "score": round(score, 3),
            })
    recos.sort(key=lambda x: x["score"], reverse=True)
    return recos[: payload.limit]


class GradeRequest(BaseModel):
    image_url: str


@app.post("/grade")
def grade_crop(payload: GradeRequest):
    # Stub: return heuristic grade by filename hints
    url = payload.image_url.lower()
    grade = "A"
    if any(k in url for k in ["damaged", "bruise", "rot"]):
        grade = "C"
    elif any(k in url for k in ["ok", "fair", "b"]):
        grade = "B"
    return {"grade": grade, "confidence": 0.82}


class ForecastRequest(BaseModel):
    crop_name: str


@app.post("/forecast")
def forecast(payload: ForecastRequest):
    # Stub: simple seasonality based on month
    month = datetime.now().month
    base = 100
    seasonal = [0.9, 0.92, 0.95, 1.0, 1.1, 1.2, 1.25, 1.2, 1.1, 1.0, 0.95, 0.92][month - 1]
    return {
        "crop": payload.crop_name,
        "next_4_weeks_kg": [int(base * seasonal * f) for f in [0.9, 1.0, 1.1, 1.2]],
    }


class NLPParseRequest(BaseModel):
    text: str


@app.post("/nlp-parse")
def nlp_parse(payload: NLPParseRequest):
    text = payload.text.lower()
    nutrients = {}
    if "protein" in text:
        nutrients["protein_g_per_day"] = 50
    if "iron" in text:
        nutrients["iron_mg_per_day"] = 18
    if "vitamin c" in text or "vit c" in text:
        nutrients["vitamin_c_mg_per_day"] = 90
    if "calorie" in text or "energy" in text:
        nutrients["calories_kcal_per_day"] = 2000
    return {"structured": nutrients, "notes": "Heuristic parser output"}


# ------------------ Seed Bengaluru Orgs ------------------

SEED_ORGS = [
    {
        "name": "St. John's Medical College Hospital",
        "email": "procurement@stjohnsblr.org",
        "role": "organization",
        "organization_name": "St. John's Hospital",
        "lat": 12.9345,
        "lng": 77.6050,
        "address": "Sarjapur Road, Bengaluru"
    },
    {
        "name": "Vriddha Ashram Bengaluru",
        "email": "admin@vriddhaashram.in",
        "role": "organization",
        "organization_name": "Vriddha Ashram",
        "lat": 12.9711,
        "lng": 77.6412,
        "address": "Old Age Home, Indiranagar, Bengaluru"
    },
    {
        "name": "SOS Children's Village Bengaluru",
        "email": "support@sosblr.org",
        "role": "organization",
        "organization_name": "SOS Orphanage",
        "lat": 12.9900,
        "lng": 77.5710,
        "address": "Bengaluru"
    },
]


@app.post("/seed-bengaluru")
def seed_bengaluru():
    created = []
    for org in SEED_ORGS:
        if not db["user"].find_one({"email": org["email"]}):
            user_doc = UserSchema(
                name=org["name"],
                email=org["email"],
                password_hash=pwd_context.hash("password123"),
                role=org["role"],
                phone=None,
                organization_name=org["organization_name"],
                location=GeoPoint(lat=org["lat"], lng=org["lng"]),
                address=org["address"],
                is_active=True,
            ).model_dump()
            db["user"].insert_one(user_doc)
            created.append(org["organization_name"])
    return {"created": created}


# ------------------ Health ------------------

@app.get("/")
def read_root():
    return {"message": "Crop to Nutrition to Patients Backend Running", "region": "Bengaluru", "radius_km": RADIUS_KM}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, "name") else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
