from core.api.engine_api import EngineAPI

# Initialize the engine API
engine = EngineAPI()

# Example: Create a beam and a column (details would be filled in real implementation)
engine.create_beam(start_node=(0,0,0), end_node=(5,0,0), section='IPE200', material='Steel')
engine.create_column(base_node=(0,0,0), top_node=(0,0,3), section='HEA200', material='Steel')

# Run analysis
engine.run_analysis()

# Run design checks
engine.run_design()

# Export to IFC
engine.export_ifc('output_model.ifc') 