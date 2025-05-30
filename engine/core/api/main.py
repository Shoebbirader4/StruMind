from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional
from engine.core.api.engine_api import EngineAPI

app = FastAPI()

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine_api = EngineAPI()

# --- Pydantic Schemas ---
class Node(BaseModel):
    id: Any
    x: float
    y: float
    z: float
    restraints: Optional[Dict[str, bool]] = None

class Element(BaseModel):
    id: Any
    startNode: Any
    endNode: Any
    section: Optional[Dict[str, Any]] = None
    material: Optional[Dict[str, Any]] = None

class Load(BaseModel):
    id: Any
    node: Any
    fx: float = 0
    fy: float = 0
    fz: float = 0
    mx: float = 0
    my: float = 0
    mz: float = 0

class ModelData(BaseModel):
    nodes: list[Node]
    elements: list[Element]
    loads: list[Load]
    materials: Optional[list[Dict[str, Any]]] = None
    sections: Optional[list[Dict[str, Any]]] = None

# --- API Endpoints ---
@app.post("/model/upload")
def upload_model(model: ModelData):
    # Convert to AnalyticalModel (implement conversion as needed)
    try:
        analytical_model = AnalyticalModel.from_dict(model.dict())
        engine_api.load_model(analytical_model)
        return {"status": "Model loaded"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analysis/run")
def run_analysis(loads: list[Load] = Body(...)):
    try:
        results = engine_api.run_analysis([l.dict() for l in loads])
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/results")
def get_results():
    results = engine_api.get_results()
    if results is None:
        raise HTTPException(status_code=404, detail="No results available")
    return results

@app.get("/sections")
def get_sections():
    return engine_api.get_section_names()

@app.get("/materials")
def get_materials():
    return engine_api.get_material_names() 