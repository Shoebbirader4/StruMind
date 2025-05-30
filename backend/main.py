from fastapi import FastAPI, HTTPException, UploadFile, File, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from typing import List, Tuple, Optional, Literal
import sys
import os
from datetime import datetime

# Add engine to sys.path for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../engine')))
from core.api.engine_api import EngineAPI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance (for demo)
engine = EngineAPI()

# In-memory storage for dashboard
recent_models = []
recent_analyses = []

class BeamInput(BaseModel):
    start_node: Tuple[float, float, float]
    end_node: Tuple[float, float, float]
    section: str
    material: str

class ColumnInput(BaseModel):
    base_node: Tuple[float, float, float]
    top_node: Tuple[float, float, float]
    section: str
    material: str

class LoadInput(BaseModel):
    load_type: Literal['point', 'distributed', 'area', 'temperature']
    location: int  # node or element index
    direction: Optional[int] = 1  # 0=x, 1=y, 2=z
    magnitude: float

class ModelInput(BaseModel):
    beams: Optional[List[BeamInput]] = []
    columns: Optional[List[ColumnInput]] = []
    loads: Optional[List[LoadInput]] = []

class DesignRequest(BaseModel):
    code_name: str

class ExportRequest(BaseModel):
    filename: str = "exported_model.ifc"

@app.get("/")
def read_root():
    return {"message": "Structural Engine API is running!"}

@app.post("/model")
def create_model(model: ModelInput):
    global engine
    engine = EngineAPI()
    # Add beams
    for beam in model.beams:
        engine.create_beam(
            start_node=beam.start_node,
            end_node=beam.end_node,
            section=beam.section,
            material=beam.material
        )
    # Add columns
    for col in model.columns:
        engine.create_column(
            base_node=col.base_node,
            top_node=col.top_node,
            section=col.section,
            material=col.material
        )
    # Store loads in engine for analysis
    engine._frontend_loads = model.loads or []
    # Add to recent models
    recent_models.append({
        'timestamp': datetime.now().isoformat(),
        'beams': len(model.beams),
        'columns': len(model.columns),
        'loads': len(model.loads or [])
    })
    return {"status": "Model created", "beams": len(model.beams), "columns": len(model.columns), "loads": len(model.loads or [])}

@app.post("/analyze")
def analyze():
    try:
        loads = getattr(engine, '_frontend_loads', [])
        engine.run_analysis(loads)
        # Add to recent analyses
        recent_analyses.append({
            'timestamp': datetime.now().isoformat(),
            'result_keys': list(engine.get_results().keys()) if engine.get_results() else []
        })
        return {"status": "Analysis complete", "results": engine.get_results()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/design")
def design(request: DesignRequest):
    try:
        # For demo, use empty forces_by_element (should be filled with real analysis results)
        forces_by_element = {}
        results = engine.run_code_checks(request.code_name, forces_by_element)
        return {"status": "Design checks complete", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export")
def export_model(request: ExportRequest):
    try:
        filename = request.filename
        # Export IFC file to a temp location
        export_path = os.path.abspath(filename)
        msg = engine.export_ifc(export_path)
        if not os.path.exists(export_path):
            raise Exception(f"Export failed: {msg}")
        return FileResponse(export_path, media_type='application/octet-stream', filename=filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/beam-forces")
def get_beam_forces():
    try:
        diagrams = engine.get_beam_internal_force_diagrams()
        # Convert numpy arrays to lists for JSON serialization
        for d in diagrams:
            d['x'] = d['x'].tolist() if hasattr(d['x'], 'tolist') else d['x']
            d['M'] = d['M'].tolist() if hasattr(d['M'], 'tolist') else d['M']
            d['V'] = d['V'].tolist() if hasattr(d['V'], 'tolist') else d['V']
            d['N'] = d['N'].tolist() if hasattr(d['N'], 'tolist') else d['N']
        return JSONResponse(content={"diagrams": diagrams})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drawings/svg")
def get_svg_drawing():
    try:
        # Generate SVG drawing as a string
        svg = engine.generate_drawings(filename=None, pdf=False)
        return Response(content=svg, media_type="image/svg+xml")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/displacement-contours")
def get_displacement_contours():
    try:
        contours = engine.get_shell_displacement_contours()
        # Convert numpy arrays to lists for JSON serialization
        for c in contours:
            c['disp'] = [float(d) for d in c['disp']]
        return JSONResponse(content={"contours": contours})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/nonlinear")
def nonlinear_analysis(options: dict = {}):
    try:
        result = engine.run_nonlinear_analysis(options)
        return {"status": "Nonlinear analysis complete", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/dynamic")
def dynamic_analysis():
    try:
        loads = getattr(engine, '_frontend_loads', [])
        result = engine.run_dynamic_analysis(loads)
        return {"status": "Dynamic analysis complete", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/buckling")
def buckling_analysis():
    try:
        result = engine.run_buckling_analysis()
        return {"status": "Buckling analysis complete", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/modal")
def modal_analysis(num_modes: int = 5, lumped: bool = True):
    try:
        result = engine.run_modal_analysis(num_modes=num_modes, lumped=lumped)
        return {"status": "Modal analysis complete", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/time-history")
def time_history_analysis(params: dict):
    try:
        result = engine.run_time_history(**params)
        return {"status": "Time history analysis complete", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/response-spectrum")
def response_spectrum_analysis(params: dict):
    try:
        result = engine.run_response_spectrum(**params)
        return {"status": "Response spectrum analysis complete", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/import/ifc")
def import_ifc(file: UploadFile = File(...)):
    try:
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as f:
            f.write(file.file.read())
        msg = engine.import_ifc(file_location)
        os.remove(file_location)
        return {"status": "IFC import complete", "message": msg}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sections")
def get_sections():
    try:
        return {"sections": engine.get_section_names()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/materials")
def get_materials():
    try:
        return {"materials": engine.get_material_names()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export/report")
def export_report(format: str = 'csv'):
    try:
        filename = f"report.{format}"
        msg = engine.generate_report(filename=filename, pdf=(format=='pdf'), excel=(format=='excel'))
        if not os.path.exists(filename):
            raise Exception(f"Report export failed: {msg}")
        media_type = 'application/pdf' if format=='pdf' else 'application/vnd.ms-excel' if format=='excel' else 'text/csv'
        return FileResponse(filename, media_type=media_type, filename=filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard")
def dashboard():
    return {"recent_models": recent_models[-10:], "recent_analyses": recent_analyses[-10:]}

@app.get("/geometry")
def geometry():
    try:
        nodes = getattr(engine.model, 'nodes', [])
        elements = []
        for elem in getattr(engine.model, 'elements', []):
            elements.append({
                'type': type(elem).__name__,
                'nodes': getattr(elem, 'nodes', [])
            })
        return {"nodes": nodes, "elements": elements}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/undo")
def undo():
    try:
        result = engine.undo() or "Not implemented"
        return {"status": "Undo", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/redo")
def redo():
    try:
        result = engine.redo() or "Not implemented"
        return {"status": "Redo", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class BeamDirectInput(BaseModel):
    start_node: Tuple[float, float, float]
    end_node: Tuple[float, float, float]
    section: str
    material: str

@app.post("/element/beam")
def create_beam(beam: BeamDirectInput):
    try:
        result = engine.create_beam(
            start_node=beam.start_node,
            end_node=beam.end_node,
            section=beam.section,
            material=beam.material
        ) or "Not implemented"
        return {"status": "Beam created", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ColumnDirectInput(BaseModel):
    base_node: Tuple[float, float, float]
    top_node: Tuple[float, float, float]
    section: str
    material: str

@app.post("/element/column")
def create_column(col: ColumnDirectInput):
    try:
        result = engine.create_column(
            base_node=col.base_node,
            top_node=col.top_node,
            section=col.section,
            material=col.material
        ) or "Not implemented"
        return {"status": "Column created", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/section/{name}")
def get_section_detail(name: str = Path(...)):
    try:
        result = engine.get_section(name)
        return {"section": result}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/material/{name}")
def get_material_detail(name: str = Path(...)):
    try:
        result = engine.get_material(name)
        return {"material": result}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

class LoadCaseInput(BaseModel):
    name: str
    loads: list

@app.post("/load/case")
def create_load_case(case: LoadCaseInput):
    try:
        result = engine.create_load_case(case.name, case.loads)
        return {"status": "Load case created", "result": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class LoadComboInput(BaseModel):
    name: str
    cases_and_factors: list

@app.post("/load/combination")
def create_load_combination(combo: LoadComboInput):
    try:
        result = engine.create_load_combination(combo.name, combo.cases_and_factors)
        return {"status": "Load combination created", "result": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CodeComboInput(BaseModel):
    load_cases: list
    code: str = "ASCE"

@app.post("/load/code-combinations")
def generate_code_combinations(data: CodeComboInput):
    try:
        result = engine.generate_code_combinations(data.load_cases, data.code)
        return {"status": "Code combinations generated", "result": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/sections")
def optimize_sections():
    try:
        result = engine.optimize_sections() or "Not implemented"
        return {"status": "Optimization", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class StagedConstructionInput(BaseModel):
    stages: list

@app.post("/analyze/staged-construction")
def staged_construction(data: StagedConstructionInput):
    try:
        result = engine.run_staged_construction(data.stages)
        return {"status": "Staged construction analysis complete", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PushoverInput(BaseModel):
    control_node: int
    direction: int = 1
    steps: int = 10
    max_disp: float = 0.1

@app.post("/analyze/pushover")
def pushover_analysis(data: PushoverInput):
    try:
        result = engine.run_pushover_analysis(
            data.control_node, data.direction, data.steps, data.max_disp
        )
        return {"status": "Pushover analysis complete", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/matrix/mass")
def get_mass_matrix(lumped: bool = Query(True)):
    try:
        result = engine.get_mass_matrix(lumped=lumped)
        return {"mass_matrix": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/matrix/damping")
def get_damping_matrix(alpha: float = Query(0.0), beta: float = Query(0.01), lumped: bool = Query(True)):
    try:
        result = engine.get_damping_matrix(alpha=alpha, beta=beta, lumped=lumped)
        return {"damping_matrix": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drawings/pdf")
def get_pdf_drawing(view: str = Query('plan'), overlay_forces: bool = Query(True)):
    try:
        filename = "drawing.pdf"
        msg = engine.generate_drawings(filename=filename, view=view, overlay_forces=overlay_forces, pdf=True)
        if not os.path.exists(filename):
            raise Exception(f"PDF drawing export failed: {msg}")
        return FileResponse(filename, media_type='application/pdf', filename=filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# TODO: Add endpoints for model creation, analysis, design, export, etc. 