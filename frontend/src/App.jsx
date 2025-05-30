import React, { useEffect, useState } from 'react';
import { AppBar, Toolbar, Typography, Container, Button, Box, Paper, CircularProgress, TextField, Grid, Snackbar, Alert, MenuItem, Select, FormControl, InputLabel, Tabs, Tab, Drawer, List, ListItem, ListItemIcon, ListItemText, Divider, Stepper, Step, StepLabel, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, IconButton, Dialog, DialogTitle, DialogContent, DialogActions, Card, CardContent, CardHeader, Accordion, AccordionSummary, AccordionDetails, TableFooter } from '@mui/material';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import DashboardIcon from '@mui/icons-material/Dashboard';
import BuildIcon from '@mui/icons-material/Build';
import AssessmentIcon from '@mui/icons-material/Assessment';
import BarChartIcon from '@mui/icons-material/BarChart';
import LayersIcon from '@mui/icons-material/Layers';
import ThreeDRotationIcon from '@mui/icons-material/ThreeDRotation';
import ScienceIcon from '@mui/icons-material/Science';
import SettingsIcon from '@mui/icons-material/Settings';
import AddIcon from '@mui/icons-material/Add';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import DownloadIcon from '@mui/icons-material/Download';
import Popover from '@mui/material/Popover';
import Switch from '@mui/material/Switch';
import FormControlLabel from '@mui/material/FormControlLabel';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import CardActions from '@mui/material/CardActions';
import Avatar from '@mui/material/Avatar';

function Beam3D({ nodes }) {
  if (!nodes || nodes.length < 2) return null;
  const [start, end] = nodes;
  return (
    <line>
      <bufferGeometry attach="geometry">
        <bufferAttribute
          attach="attributes-position"
          count={2}
          array={new Float32Array([...start, ...end])}
          itemSize={3}
        />
      </bufferGeometry>
      <lineBasicMaterial attach="material" color="orange" linewidth={2} />
    </line>
  );
}

function Model3D({ nodes, elements }) {
  return (
    <Canvas camera={{ position: [5, 5, 5], fov: 50 }} style={{ height: 400, background: '#222' }}>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <OrbitControls />
      {elements.map((el, i) => (
        <Beam3D key={i} nodes={el.nodes} />
      ))}
    </Canvas>
  );
}

const drawerWidth = 220;
const navItems = [
  { label: 'Dashboard', icon: <DashboardIcon /> },
  { label: 'Model Builder', icon: <BuildIcon /> },
  { label: 'Results', icon: <BarChartIcon /> },
  { label: 'Advanced Analysis', icon: <ScienceIcon /> },
  { label: '2D View', icon: <LayersIcon /> },
  { label: '3D Viewer', icon: <ThreeDRotationIcon /> },
  { label: 'Expert', icon: <SettingsIcon /> },
];

const modelSteps = ['Nodes', 'Elements', 'Loads', 'Materials/Sections'];

function App() {
  const [apiStatus, setApiStatus] = useState('Connecting to backend...');
  const [tab, setTab] = useState(0);
  const [showDashboard, setShowDashboard] = useState(true);
  const [showModelBuilder, setShowModelBuilder] = useState(false);
  const [beams, setBeams] = useState([]);
  const [columns, setColumns] = useState([]);
  const [loads, setLoads] = useState([]);
  const [beamForm, setBeamForm] = useState({ start_node: '', end_node: '', section: '', material: '' });
  const [columnForm, setColumnForm] = useState({ base_node: '', top_node: '', section: '', material: '' });
  const [loadForm, setLoadForm] = useState({ load_type: 'point', location: '', direction: '1', magnitude: '' });
  const [loading, setLoading] = useState(false);
  const [analyzeResult, setAnalyzeResult] = useState(null);
  const [designCode, setDesignCode] = useState('AISC');
  const [designResult, setDesignResult] = useState(null);
  const [beamDiagrams, setBeamDiagrams] = useState([]);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const [svgDrawing, setSvgDrawing] = useState(null);
  const [displacementContours, setDisplacementContours] = useState([]);
  const [sections, setSections] = useState([]);
  const [materials, setMaterials] = useState([]);
  const [nonlinearResult, setNonlinearResult] = useState(null);
  const [dynamicResult, setDynamicResult] = useState(null);
  const [bucklingResult, setBucklingResult] = useState(null);
  const [modalResult, setModalResult] = useState(null);
  const [timeHistoryResult, setTimeHistoryResult] = useState(null);
  const [responseSpectrumResult, setResponseSpectrumResult] = useState(null);
  const [importIFCMessage, setImportIFCMessage] = useState(null);
  const [reportExportMessage, setReportExportMessage] = useState(null);
  const [dashboardData, setDashboardData] = useState({ recent_models: [], recent_analyses: [] });
  const [geometry, setGeometry] = useState({ nodes: [], elements: [] });
  const [modelStep, setModelStep] = useState(1);
  const [editBeamIdx, setEditBeamIdx] = useState(null);
  const [editColumnIdx, setEditColumnIdx] = useState(null);
  const [editLoadIdx, setEditLoadIdx] = useState(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [modalType, setModalType] = useState('');
  const [modalData, setModalData] = useState({});
  const [svgZoom, setSvgZoom] = useState(1);
  const [svgPan, setSvgPan] = useState({ x: 0, y: 0 });
  const [svgDragging, setSvgDragging] = useState(false);
  const [svgDragStart, setSvgDragStart] = useState({ x: 0, y: 0 });
  const [selected2DElement, setSelected2DElement] = useState(null);
  const [showLabels, setShowLabels] = useState(true);
  const [showNodeNumbers, setShowNodeNumbers] = useState(true);
  const [selected3DElement, setSelected3DElement] = useState(null);
  const [anchorEl, setAnchorEl] = useState(null);
  const [wireframeMode, setWireframeMode] = useState(true);

  useEffect(() => {
    axios.get('http://localhost:8000/sections').then(res => setSections(res.data.sections));
    axios.get('http://localhost:8000/materials').then(res => setMaterials(res.data.materials));
    axios.get('http://localhost:8000/').then(res => setApiStatus(res.data.message)).catch(() => setApiStatus('Backend not reachable'));
    axios.get('http://localhost:8000/dashboard').then(res => setDashboardData(res.data));
  }, []);

  const handleTabChange = (event, newValue) => {
    setTab(newValue);
  };

  const handleNav = (page) => {
    setShowDashboard(page === 'dashboard');
    setShowModelBuilder(page === 'model');
    setAnalyzeResult(null);
    setDesignResult(null);
  };

  const handleBeamChange = (e) => {
    setBeamForm({ ...beamForm, [e.target.name]: e.target.value });
  };

  const handleColumnChange = (e) => {
    setColumnForm({ ...columnForm, [e.target.name]: e.target.value });
  };

  const handleLoadChange = (e) => {
    setLoadForm({ ...loadForm, [e.target.name]: e.target.value });
  };

  const addBeam = () => {
    if (!beamForm.start_node || !beamForm.end_node || !beamForm.section || !beamForm.material) return;
    setBeams([...beams, { ...beamForm }]);
    setBeamForm({ start_node: '', end_node: '', section: '', material: '' });
  };

  const addColumn = () => {
    if (!columnForm.base_node || !columnForm.top_node || !columnForm.section || !columnForm.material) return;
    setColumns([...columns, { ...columnForm }]);
    setColumnForm({ base_node: '', top_node: '', section: '', material: '' });
  };

  const addLoad = () => {
    if (!loadForm.load_type || loadForm.location === '' || loadForm.magnitude === '') return;
    setLoads([...loads, { ...loadForm }]);
    setLoadForm({ load_type: 'point', location: '', direction: '1', magnitude: '' });
  };

  const handleSubmitModel = async () => {
    setLoading(true);
    try {
      // Convert string tuples to arrays
      const parseTuple = (str) => str.split(',').map(Number);
      const beamsPayload = beams.map(b => ({ ...b, start_node: parseTuple(b.start_node), end_node: parseTuple(b.end_node) }));
      const columnsPayload = columns.map(c => ({ ...c, base_node: parseTuple(c.base_node), top_node: parseTuple(c.top_node) }));
      const loadsPayload = loads.map(l => ({ ...l, location: parseInt(l.location, 10), direction: parseInt(l.direction, 10), magnitude: parseFloat(l.magnitude) }));
      await axios.post('http://localhost:8000/model', { beams: beamsPayload, columns: columnsPayload, loads: loadsPayload });
      setSnackbar({ open: true, message: 'Model created successfully!', severity: 'success' });
    } catch (e) {
      setSnackbar({ open: true, message: 'Model creation failed', severity: 'error' });
    }
    setLoading(false);
  };

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/analyze');
      setAnalyzeResult(res.data);
      setSnackbar({ open: true, message: 'Analysis complete!', severity: 'success' });
    } catch (e) {
      setSnackbar({ open: true, message: 'Analysis failed', severity: 'error' });
    }
    setLoading(false);
  };

  const handleDesign = async () => {
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/design', { code_name: designCode });
      setDesignResult(res.data);
      setSnackbar({ open: true, message: 'Design checks complete!', severity: 'success' });
    } catch (e) {
      setSnackbar({ open: true, message: 'Design checks failed', severity: 'error' });
    }
    setLoading(false);
  };

  const handleExport = async () => {
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/export', { filename: 'exported_model.ifc' }, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'exported_model.ifc');
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
      setSnackbar({ open: true, message: 'Export successful!', severity: 'success' });
    } catch (e) {
      setSnackbar({ open: true, message: 'Export failed', severity: 'error' });
    }
    setLoading(false);
  };

  const fetchBeamDiagrams = async () => {
    setLoading(true);
    try {
      const res = await axios.get('http://localhost:8000/results/beam-forces');
      setBeamDiagrams(res.data.diagrams);
      setSnackbar({ open: true, message: 'Beam force diagrams loaded!', severity: 'success' });
    } catch (e) {
      setSnackbar({ open: true, message: 'Failed to load diagrams', severity: 'error' });
    }
    setLoading(false);
  };

  const fetchSvgDrawing = async () => {
    setLoading(true);
    try {
      const res = await axios.get('http://localhost:8000/drawings/svg');
      setSvgDrawing(res.data);
      setSnackbar({ open: true, message: 'SVG drawing loaded!', severity: 'success' });
    } catch (e) {
      setSnackbar({ open: true, message: 'Failed to load SVG drawing', severity: 'error' });
    }
    setLoading(false);
  };

  const fetchDisplacementContours = async () => {
    setLoading(true);
    try {
      const res = await axios.get('http://localhost:8000/results/displacement-contours');
      setDisplacementContours(res.data.contours);
      setSnackbar({ open: true, message: 'Displacement contours loaded!', severity: 'success' });
    } catch (e) {
      setSnackbar({ open: true, message: 'Failed to load displacement contours', severity: 'error' });
    }
    setLoading(false);
  };

  // Advanced analysis handlers
  const handleNonlinear = async () => {
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/analyze/nonlinear', { steps: 10, max_iter: 20, tol: 1e-4 });
      setNonlinearResult(res.data.result);
      setSnackbar({ open: true, message: 'Nonlinear analysis complete!', severity: 'success' });
    } catch (e) {
      setSnackbar({ open: true, message: 'Nonlinear analysis failed', severity: 'error' });
    }
    setLoading(false);
  };
  const handleDynamic = async () => {
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/analyze/dynamic');
      setDynamicResult(res.data.result);
      setSnackbar({ open: true, message: 'Dynamic analysis complete!', severity: 'success' });
    } catch (e) {
      setSnackbar({ open: true, message: 'Dynamic analysis failed', severity: 'error' });
    }
    setLoading(false);
  };
  const handleBuckling = async () => {
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/analyze/buckling');
      setBucklingResult(res.data.result);
      setSnackbar({ open: true, message: 'Buckling analysis complete!', severity: 'success' });
    } catch (e) {
      setSnackbar({ open: true, message: 'Buckling analysis failed', severity: 'error' });
    }
    setLoading(false);
  };
  const handleModal = async () => {
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/analyze/modal', { num_modes: 5, lumped: true });
      setModalResult(res.data.result);
      setSnackbar({ open: true, message: 'Modal analysis complete!', severity: 'success' });
    } catch (e) {
      setSnackbar({ open: true, message: 'Modal analysis failed', severity: 'error' });
    }
    setLoading(false);
  };
  const handleTimeHistory = async () => {
    setLoading(true);
    try {
      const params = { /* user params here */ };
      const res = await axios.post('http://localhost:8000/analyze/time-history', params);
      setTimeHistoryResult(res.data.result);
      setSnackbar({ open: true, message: 'Time history analysis complete!', severity: 'success' });
    } catch (e) {
      setSnackbar({ open: true, message: 'Time history analysis failed', severity: 'error' });
    }
    setLoading(false);
  };
  const handleResponseSpectrum = async () => {
    setLoading(true);
    try {
      const params = { /* user params here */ };
      const res = await axios.post('http://localhost:8000/analyze/response-spectrum', params);
      setResponseSpectrumResult(res.data.result);
      setSnackbar({ open: true, message: 'Response spectrum analysis complete!', severity: 'success' });
    } catch (e) {
      setSnackbar({ open: true, message: 'Response spectrum analysis failed', severity: 'error' });
    }
    setLoading(false);
  };
  const handleImportIFC = async (file) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await axios.post('http://localhost:8000/import/ifc', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      setImportIFCMessage(res.data.message);
      setSnackbar({ open: true, message: res.data.message, severity: 'success' });
    } catch (e) {
      setSnackbar({ open: true, message: 'IFC import failed', severity: 'error' });
    }
    setLoading(false);
  };
  const handleExportReport = async (format) => {
    setLoading(true);
    try {
      const res = await axios.post(`http://localhost:8000/export/report?format=${format}`, {}, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `report.${format}`);
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
      setReportExportMessage('Report exported!');
      setSnackbar({ open: true, message: 'Report exported!', severity: 'success' });
    } catch (e) {
      setSnackbar({ open: true, message: 'Report export failed', severity: 'error' });
    }
    setLoading(false);
  };

  const fetchGeometry = async () => {
    setLoading(true);
    try {
      const res = await axios.get('http://localhost:8000/geometry');
      setGeometry(res.data);
      setSnackbar({ open: true, message: '3D geometry loaded!', severity: 'success' });
    } catch (e) {
      setSnackbar({ open: true, message: 'Failed to load geometry', severity: 'error' });
    }
    setLoading(false);
  };

  // Modal handlers
  const openModal = (type, idx = null, data = {}) => {
    setModalType(type);
    setModalOpen(true);
    setModalData(data);
    if (type === 'beam') setEditBeamIdx(idx);
    if (type === 'column') setEditColumnIdx(idx);
    if (type === 'load') setEditLoadIdx(idx);
  };
  const closeModal = () => {
    setModalOpen(false);
    setModalData({});
    setEditBeamIdx(null);
    setEditColumnIdx(null);
    setEditLoadIdx(null);
  };
  // Table add/edit/delete logic
  const handleSaveModal = () => {
    if (modalType === 'beam') {
      if (editBeamIdx !== null) {
        const newBeams = [...beams];
        newBeams[editBeamIdx] = modalData;
        setBeams(newBeams);
      } else {
        setBeams([...beams, modalData]);
      }
    }
    if (modalType === 'column') {
      if (editColumnIdx !== null) {
        const newColumns = [...columns];
        newColumns[editColumnIdx] = modalData;
        setColumns(newColumns);
      } else {
        setColumns([...columns, modalData]);
      }
    }
    if (modalType === 'load') {
      if (editLoadIdx !== null) {
        const newLoads = [...loads];
        newLoads[editLoadIdx] = modalData;
        setLoads(newLoads);
      } else {
        setLoads([...loads, modalData]);
      }
    }
    closeModal();
  };
  const handleDelete = (type, idx) => {
    if (type === 'beam') setBeams(beams.filter((_, i) => i !== idx));
    if (type === 'column') setColumns(columns.filter((_, i) => i !== idx));
    if (type === 'load') setLoads(loads.filter((_, i) => i !== idx));
  };

  // 2D View event handlers
  const handleSvgWheel = (e) => {
    e.preventDefault();
    setSvgZoom(z => Math.max(0.2, Math.min(5, z - e.deltaY * 0.001)));
  };
  const handleSvgMouseDown = (e) => {
    setSvgDragging(true);
    setSvgDragStart({ x: e.clientX - svgPan.x, y: e.clientY - svgPan.y });
  };
  const handleSvgMouseMove = (e) => {
    if (svgDragging) {
      setSvgPan({ x: e.clientX - svgDragStart.x, y: e.clientY - svgDragStart.y });
    }
  };
  const handleSvgMouseUp = () => setSvgDragging(false);
  const handleSvgElementClick = (type, idx) => setSelected2DElement({ type, idx });

  // 3D View element click/hover
  const handle3DElementClick = (el, event) => {
    setSelected3DElement(el);
    setAnchorEl(event.target);
  };
  const handle3DPopoverClose = () => setAnchorEl(null);

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: '#f5f6fa' }}>
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: { width: drawerWidth, boxSizing: 'border-box', bgcolor: '#1a237e', color: 'white' },
        }}
      >
        <MuiToolbar sx={{ minHeight: 64 }}>
          <Typography variant="h6" sx={{ fontWeight: 700, letterSpacing: 1 }}>
            StruMind
          </Typography>
        </MuiToolbar>
        <Divider sx={{ bgcolor: 'rgba(255,255,255,0.12)' }} />
        <List>
          {navItems.map((item, idx) => (
            <ListItem button key={item.label} selected={tab === idx} onClick={() => setTab(idx)}>
              <ListItemIcon sx={{ color: 'white' }}>{item.icon}</ListItemIcon>
              <ListItemText primary={item.label} />
            </ListItem>
          ))}
        </List>
      </Drawer>
      <Box sx={{ flexGrow: 1, ml: `${drawerWidth}px` }}>
        <AppBar position="fixed" color="primary" sx={{ zIndex: 1201, ml: `${drawerWidth}px` }}>
          <MuiToolbar>
            <Typography variant="h6" sx={{ flexGrow: 1 }}>
              {navItems[tab].label} {tab === 0 && dashboardData.recent_models.length > 0 ? `- ${dashboardData.recent_models[dashboardData.recent_models.length-1].timestamp}` : ''}
            </Typography>
            <Typography variant="body2" sx={{ mr: 2 }}>
              {apiStatus}
            </Typography>
            {/* Add quick actions here if needed */}
          </MuiToolbar>
        </AppBar>
        <MuiToolbar />
        <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
          {/* Replace tab === N with tab index from navItems */}
          {tab === 0 && (
            <Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
                <Card sx={{ minWidth: 180, mr: 2, bgcolor: '#e3f2fd' }}>
                  <CardContent>
                    <Typography variant="h6">Total Models</Typography>
                    <Typography variant="h4">{dashboardData.recent_models.length}</Typography>
                  </CardContent>
                </Card>
                <Card sx={{ minWidth: 180, mr: 2, bgcolor: '#e8f5e9' }}>
                  <CardContent>
                    <Typography variant="h6">Total Analyses</Typography>
                    <Typography variant="h4">{dashboardData.recent_analyses.length}</Typography>
                  </CardContent>
                </Card>
                <Card sx={{ minWidth: 220, bgcolor: '#fff3e0' }}>
                  <CardContent>
                    <Typography variant="h6">Last Analysis</Typography>
                    <Typography variant="body2">{dashboardData.recent_analyses.length > 0 ? dashboardData.recent_analyses[dashboardData.recent_analyses.length-1].timestamp : 'N/A'}</Typography>
                  </CardContent>
                </Card>
              </Box>
              <Typography variant="h5" gutterBottom color="primary">Recent Models</Typography>
              <Grid container spacing={2} sx={{ mb: 3 }}>
                {dashboardData.recent_models.length === 0 ? (
                  <Grid item xs={12}><Typography>No models yet.</Typography></Grid>
                ) : dashboardData.recent_models.map((m, i) => (
                  <Grid item xs={12} sm={6} md={4} key={i}>
                    <Card elevation={2} sx={{ p: 1 }}>
                      <CardContent>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                          <Avatar sx={{ bgcolor: '#1976d2', mr: 1 }}>M</Avatar>
                          <Typography variant="subtitle1">Model {i + 1}</Typography>
                        </Box>
                        <Typography variant="body2">Beams: {m.beams}, Columns: {m.columns}, Loads: {m.loads}</Typography>
                        <Typography variant="caption" color="text.secondary">{m.timestamp}</Typography>
                      </CardContent>
                      <CardActions>
                        <Button size="small">View</Button>
                        <Button size="small">Duplicate</Button>
                        <Button size="small" color="error">Delete</Button>
                      </CardActions>
                    </Card>
                  </Grid>
                ))}
              </Grid>
              <Typography variant="h5" gutterBottom color="primary">Recent Analyses</Typography>
              <Grid container spacing={2}>
                {dashboardData.recent_analyses.length === 0 ? (
                  <Grid item xs={12}><Typography>No analyses yet.</Typography></Grid>
                ) : dashboardData.recent_analyses.map((a, i) => (
                  <Grid item xs={12} sm={6} md={4} key={i}>
                    <Card elevation={2} sx={{ p: 1 }}>
                      <CardContent>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                          <Avatar sx={{ bgcolor: '#388e3c', mr: 1 }}>A</Avatar>
                          <Typography variant="subtitle1">Analysis {i + 1}</Typography>
                        </Box>
                        <Typography variant="body2">Result keys: {a.result_keys.join(', ')}</Typography>
                        <Typography variant="caption" color="text.secondary">{a.timestamp}</Typography>
                      </CardContent>
                      <CardActions>
                        <Button size="small">View</Button>
                        <Button size="small" color="error">Delete</Button>
                      </CardActions>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Box>
          )}
          {tab === 1 && (
            <Paper elevation={3} sx={{ p: 4 }}>
              <Typography variant="h5" gutterBottom color="primary">Model Builder</Typography>
              <Stepper activeStep={modelStep} alternativeLabel sx={{ mb: 4 }}>
                {modelSteps.map((label, idx) => (
                  <Step key={label} completed={modelStep > idx}>
                    <StepLabel>{label}</StepLabel>
                  </Step>
                ))}
              </Stepper>
              {/* Elements Step: Beams/Columns */}
              <Box sx={{ mb: 4 }}>
                <Typography variant="h6">Beams</Typography>
                <TableContainer component={Paper} sx={{ mb: 2 }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Start Node</TableCell>
                        <TableCell>End Node</TableCell>
                        <TableCell>Section</TableCell>
                        <TableCell>Material</TableCell>
                        <TableCell align="right">Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {beams.map((b, i) => (
                        <TableRow key={i}>
                          <TableCell>{b.start_node}</TableCell>
                          <TableCell>{b.end_node}</TableCell>
                          <TableCell>{b.section}</TableCell>
                          <TableCell>{b.material}</TableCell>
                          <TableCell align="right">
                            <IconButton size="small" onClick={() => openModal('beam', i, b)}><EditIcon /></IconButton>
                            <IconButton size="small" onClick={() => handleDelete('beam', i)}><DeleteIcon /></IconButton>
                          </TableCell>
                        </TableRow>
                      ))}
                      <TableRow>
                        <TableCell colSpan={5} align="center">
                          <Button startIcon={<AddIcon />} onClick={() => openModal('beam')}>Add Beam</Button>
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
                <Typography variant="h6">Columns</Typography>
                <TableContainer component={Paper} sx={{ mb: 2 }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Base Node</TableCell>
                        <TableCell>Top Node</TableCell>
                        <TableCell>Section</TableCell>
                        <TableCell>Material</TableCell>
                        <TableCell align="right">Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {columns.map((c, i) => (
                        <TableRow key={i}>
                          <TableCell>{c.base_node}</TableCell>
                          <TableCell>{c.top_node}</TableCell>
                          <TableCell>{c.section}</TableCell>
                          <TableCell>{c.material}</TableCell>
                          <TableCell align="right">
                            <IconButton size="small" onClick={() => openModal('column', i, c)}><EditIcon /></IconButton>
                            <IconButton size="small" onClick={() => handleDelete('column', i)}><DeleteIcon /></IconButton>
                          </TableCell>
                        </TableRow>
                      ))}
                      <TableRow>
                        <TableCell colSpan={5} align="center">
                          <Button startIcon={<AddIcon />} onClick={() => openModal('column')}>Add Column</Button>
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
              {/* Loads Step */}
              <Box sx={{ mb: 4 }}>
                <Typography variant="h6">Loads</Typography>
                <TableContainer component={Paper} sx={{ mb: 2 }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Type</TableCell>
                        <TableCell>Location</TableCell>
                        <TableCell>Direction</TableCell>
                        <TableCell>Magnitude</TableCell>
                        <TableCell align="right">Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {loads.map((l, i) => (
                        <TableRow key={i}>
                          <TableCell>{l.load_type}</TableCell>
                          <TableCell>{l.location}</TableCell>
                          <TableCell>{l.direction}</TableCell>
                          <TableCell>{l.magnitude}</TableCell>
                          <TableCell align="right">
                            <IconButton size="small" onClick={() => openModal('load', i, l)}><EditIcon /></IconButton>
                            <IconButton size="small" onClick={() => handleDelete('load', i)}><DeleteIcon /></IconButton>
                          </TableCell>
                        </TableRow>
                      ))}
                      <TableRow>
                        <TableCell colSpan={5} align="center">
                          <Button startIcon={<AddIcon />} onClick={() => openModal('load')}>Add Load</Button>
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
              {/* Live 2D SVG Preview */}
              <Box sx={{ mb: 4 }}>
                <Typography variant="h6">2D Schematic Preview</Typography>
                <Box sx={{ border: '1px solid #ccc', borderRadius: 2, bgcolor: '#fff', p: 2, minHeight: 220, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <svg width="400" height="200" style={{ background: '#fafafa' }}>
                    {/* Draw beams */}
                    {beams.map((b, i) => {
                      const s = b.start_node.split(',').map(Number);
                      const e = b.end_node.split(',').map(Number);
                      return <line key={i} x1={s[0]*40+200} y1={100-s[1]*40} x2={e[0]*40+200} y2={100-e[1]*40} stroke="#1976d2" strokeWidth="3" />;
                    })}
                    {/* Draw columns */}
                    {columns.map((c, i) => {
                      const s = c.base_node.split(',').map(Number);
                      const e = c.top_node.split(',').map(Number);
                      return <line key={i} x1={s[0]*40+200} y1={100-s[1]*40} x2={e[0]*40+200} y2={100-e[1]*40} stroke="#388e3c" strokeWidth="3" />;
                    })}
                    {/* Draw nodes as circles */}
                    {Array.from(new Set([
                      ...beams.flatMap(b => [b.start_node, b.end_node]),
                      ...columns.flatMap(c => [c.base_node, c.top_node])
                    ])).map((n, i) => {
                      const p = n.split(',').map(Number);
                      return <circle key={i} cx={p[0]*40+200} cy={100-p[1]*40} r={6} fill="#ff9800" stroke="#333" />;
                    })}
                  </svg>
                </Box>
              </Box>
              {/* Submit/Analyze Buttons */}
              <Box sx={{ mt: 2 }}>
                <Button variant="contained" color="primary" sx={{ mr: 2 }} onClick={handleSubmitModel} disabled={loading}>
                  {loading ? <CircularProgress size={20} /> : 'Submit Model'}
                </Button>
                <Button variant="contained" color="secondary" onClick={handleAnalyze} disabled={loading}>
                  {loading ? <CircularProgress size={20} /> : 'Run Analysis'}
                </Button>
              </Box>
              {/* Modal Dialog for Add/Edit */}
              <Dialog open={modalOpen} onClose={closeModal}>
                <DialogTitle>{editBeamIdx !== null || editColumnIdx !== null || editLoadIdx !== null ? 'Edit' : 'Add'} {modalType.charAt(0).toUpperCase() + modalType.slice(1)}</DialogTitle>
                <DialogContent>
                  {modalType === 'beam' && (
                    <>
                      <TextField label="Start Node" value={modalData.start_node || ''} onChange={e => setModalData({ ...modalData, start_node: e.target.value })} fullWidth sx={{ mb: 2 }} />
                      <TextField label="End Node" value={modalData.end_node || ''} onChange={e => setModalData({ ...modalData, end_node: e.target.value })} fullWidth sx={{ mb: 2 }} />
                      <TextField label="Section" value={modalData.section || ''} onChange={e => setModalData({ ...modalData, section: e.target.value })} fullWidth sx={{ mb: 2 }} />
                      <TextField label="Material" value={modalData.material || ''} onChange={e => setModalData({ ...modalData, material: e.target.value })} fullWidth />
                    </>
                  )}
                  {modalType === 'column' && (
                    <>
                      <TextField label="Base Node" value={modalData.base_node || ''} onChange={e => setModalData({ ...modalData, base_node: e.target.value })} fullWidth sx={{ mb: 2 }} />
                      <TextField label="Top Node" value={modalData.top_node || ''} onChange={e => setModalData({ ...modalData, top_node: e.target.value })} fullWidth sx={{ mb: 2 }} />
                      <TextField label="Section" value={modalData.section || ''} onChange={e => setModalData({ ...modalData, section: e.target.value })} fullWidth sx={{ mb: 2 }} />
                      <TextField label="Material" value={modalData.material || ''} onChange={e => setModalData({ ...modalData, material: e.target.value })} fullWidth />
                    </>
                  )}
                  {modalType === 'load' && (
                    <>
                      <TextField label="Type" value={modalData.load_type || ''} onChange={e => setModalData({ ...modalData, load_type: e.target.value })} fullWidth sx={{ mb: 2 }} />
                      <TextField label="Location" value={modalData.location || ''} onChange={e => setModalData({ ...modalData, location: e.target.value })} fullWidth sx={{ mb: 2 }} />
                      <TextField label="Direction" value={modalData.direction || ''} onChange={e => setModalData({ ...modalData, direction: e.target.value })} fullWidth sx={{ mb: 2 }} />
                      <TextField label="Magnitude" value={modalData.magnitude || ''} onChange={e => setModalData({ ...modalData, magnitude: e.target.value })} fullWidth />
                    </>
                  )}
                </DialogContent>
                <DialogActions>
                  <Button onClick={closeModal}>Cancel</Button>
                  <Button onClick={handleSaveModal} variant="contained">Save</Button>
                </DialogActions>
              </Dialog>
              {/* Analysis Results (unchanged) */}
              {analyzeResult && (
                <Box sx={{ mt: 4 }}>
                  <Typography variant="h6" color="primary">Analysis Results</Typography>
                  <pre style={{ textAlign: 'left', background: '#f4f4f4', padding: 16, borderRadius: 8, overflowX: 'auto' }}>{JSON.stringify(analyzeResult, null, 2)}</pre>
                </Box>
              )}
              {/* Design Checks (unchanged) */}
              <Box sx={{ mt: 4 }}>
                <FormControl sx={{ minWidth: 200, mr: 2 }} size="small">
                  <InputLabel id="design-code-label">Design Code</InputLabel>
                  <Select
                    labelId="design-code-label"
                    id="design-code-select"
                    value={designCode}
                    label="Design Code"
                    onChange={e => setDesignCode(e.target.value)}
                  >
                    <MenuItem value="AISC">AISC</MenuItem>
                    <MenuItem value="Eurocode">Eurocode</MenuItem>
                    <MenuItem value="IS456">IS456</MenuItem>
                  </Select>
                </FormControl>
                <Button variant="contained" color="success" onClick={handleDesign} disabled={loading}>
                  {loading ? <CircularProgress size={20} /> : 'Run Design Checks'}
                </Button>
              </Box>
              {designResult && (
                <Box sx={{ mt: 4 }}>
                  <Typography variant="h6" color="success.main">Design Check Results</Typography>
                  <pre style={{ textAlign: 'left', background: '#f4f4f4', padding: 16, borderRadius: 8, overflowX: 'auto' }}>{JSON.stringify(designResult, null, 2)}</pre>
                </Box>
              )}
              <Box sx={{ mt: 4 }}>
                <Button variant="contained" color="info" onClick={handleExport} disabled={loading}>
                  {loading ? <CircularProgress size={20} /> : 'Export IFC'}
                </Button>
              </Box>
            </Paper>
          )}
          {tab === 2 && (
            <Box>
              <Card elevation={3} sx={{ mb: 3 }}>
                <CardHeader title="Beam Force Diagrams" action={<Button variant="outlined" onClick={fetchBeamDiagrams} disabled={loading}>{loading ? <CircularProgress size={20} /> : 'Reload'}</Button>} />
                <CardContent>
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography>All Beams</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      {beamDiagrams.length > 0 ? beamDiagrams.map((d, i) => (
                        <Box key={i} sx={{ mb: 4 }}>
                          <Typography variant="subtitle1" color="text.secondary">Beam {i + 1}</Typography>
                          <ResponsiveContainer width="100%" height={200}>
                            <LineChart data={d.x.map((x, idx) => ({ x, M: d.M[idx], V: d.V[idx], N: d.N[idx] }))}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="x" label={{ value: 'x (m)', position: 'insideBottomRight', offset: 0 }} />
                              <YAxis label={{ value: 'Value', angle: -90, position: 'insideLeft' }} />
                              <Tooltip />
                              <Legend />
                              <Line type="monotone" dataKey="M" stroke="#1976d2" name="Moment (M)" />
                              <Line type="monotone" dataKey="V" stroke="#d32f2f" name="Shear (V)" />
                              <Line type="monotone" dataKey="N" stroke="#388e3c" name="Axial (N)" />
                            </LineChart>
                          </ResponsiveContainer>
                        </Box>
                      )) : <Typography color="text.secondary">No beam force diagrams available.</Typography>}
                    </AccordionDetails>
                  </Accordion>
                </CardContent>
              </Card>
              <Card elevation={3} sx={{ mb: 3 }}>
                <CardHeader title="SVG Drawing" action={<Button variant="outlined" onClick={fetchSvgDrawing} disabled={loading}>{loading ? <CircularProgress size={20} /> : 'Reload'}</Button>} />
                <CardContent>
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography>SVG Plan/Elevation</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      {svgDrawing ? (
                        <Box sx={{ border: '1px solid #ccc', borderRadius: 2, overflow: 'auto', background: '#fff', p: 2 }}>
                          <div dangerouslySetInnerHTML={{ __html: svgDrawing }} />
                          <Button startIcon={<DownloadIcon />} sx={{ mt: 2 }} variant="outlined" onClick={async () => {
                            const res = await axios.get('http://localhost:8000/drawings/svg');
                            const blob = new Blob([res.data], { type: 'image/svg+xml' });
                            const url = window.URL.createObjectURL(blob);
                            const link = document.createElement('a');
                            link.href = url;
                            link.setAttribute('download', 'drawing.svg');
                            document.body.appendChild(link);
                            link.click();
                            link.parentNode.removeChild(link);
                          }}>Download SVG</Button>
                        </Box>
                      ) : <Typography color="text.secondary">No SVG drawing available.</Typography>}
                    </AccordionDetails>
                  </Accordion>
                </CardContent>
              </Card>
              <Card elevation={3} sx={{ mb: 3 }}>
                <CardHeader title="Displacement Contours" action={<Button variant="outlined" onClick={fetchDisplacementContours} disabled={loading}>{loading ? <CircularProgress size={20} /> : 'Reload'}</Button>} />
                <CardContent>
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography>Contours Table</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      {displacementContours.length > 0 ? (
                        <TableContainer component={Paper}>
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell>Element</TableCell>
                                <TableCell>Displacements</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {displacementContours.map((c, i) => (
                                <TableRow key={i}>
                                  <TableCell>{i + 1}</TableCell>
                                  <TableCell>{c.disp.map((d, j) => `Node ${j + 1}: ${d.toFixed(4)}`).join(', ')}</TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                            <TableFooter>
                              <TableRow>
                                <TableCell colSpan={2} align="right">
                                  <Button startIcon={<DownloadIcon />} variant="outlined" onClick={async () => {
                                    // Download as CSV
                                    let csv = 'Element,Displacements\n';
                                    displacementContours.forEach((c, i) => {
                                      csv += `${i + 1},${c.disp.map((d, j) => `Node ${j + 1}: ${d.toFixed(4)}`).join(' | ')}\n`;
                                    });
                                    const blob = new Blob([csv], { type: 'text/csv' });
                                    const url = window.URL.createObjectURL(blob);
                                    const link = document.createElement('a');
                                    link.href = url;
                                    link.setAttribute('download', 'displacement_contours.csv');
                                    document.body.appendChild(link);
                                    link.click();
                                    link.parentNode.removeChild(link);
                                  }}>Download CSV</Button>
                                </TableCell>
                              </TableRow>
                            </TableFooter>
                          </Table>
                        </TableContainer>
                      ) : <Typography color="text.secondary">No displacement contours available.</Typography>}
                    </AccordionDetails>
                  </Accordion>
                </CardContent>
              </Card>
            </Box>
          )}
          {tab === 3 && (
            <Paper elevation={3} sx={{ p: 4 }}>
              <Typography variant="h5" gutterBottom color="primary">Advanced Analysis</Typography>
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Nonlinear Analysis</Typography>
                  <Tooltip title="Run advanced nonlinear analysis with custom options."><HelpOutlineIcon sx={{ ml: 1 }} /></Tooltip>
                </AccordionSummary>
                <AccordionDetails>
                  <Button onClick={handleNonlinear} variant="contained" color="warning" sx={{ mr: 2 }} disabled={loading}>
                    {loading ? <CircularProgress size={20} /> : 'Run Nonlinear Analysis'}
                  </Button>
                  {nonlinearResult && <Box sx={{ mt: 2 }}><Typography variant="subtitle1">Nonlinear Result</Typography><pre>{JSON.stringify(nonlinearResult, null, 2)}</pre></Box>}
                </AccordionDetails>
              </Accordion>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Dynamic Analysis</Typography>
                  <Tooltip title="Run dynamic time-dependent analysis."><HelpOutlineIcon sx={{ ml: 1 }} /></Tooltip>
                </AccordionSummary>
                <AccordionDetails>
                  <Button onClick={handleDynamic} variant="contained" color="info" sx={{ mr: 2 }} disabled={loading}>
                    {loading ? <CircularProgress size={20} /> : 'Run Dynamic Analysis'}
                  </Button>
                  {dynamicResult && <Box sx={{ mt: 2 }}><Typography variant="subtitle1">Dynamic Result</Typography><pre>{JSON.stringify(dynamicResult, null, 2)}</pre></Box>}
                </AccordionDetails>
              </Accordion>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Buckling Analysis</Typography>
                  <Tooltip title="Compute critical buckling loads and modes."><HelpOutlineIcon sx={{ ml: 1 }} /></Tooltip>
                </AccordionSummary>
                <AccordionDetails>
                  <Button onClick={handleBuckling} variant="contained" color="secondary" sx={{ mr: 2 }} disabled={loading}>
                    {loading ? <CircularProgress size={20} /> : 'Run Buckling Analysis'}
                  </Button>
                  {bucklingResult && <Box sx={{ mt: 2 }}><Typography variant="subtitle1">Buckling Result</Typography><pre>{JSON.stringify(bucklingResult, null, 2)}</pre></Box>}
                </AccordionDetails>
              </Accordion>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Modal Analysis</Typography>
                  <Tooltip title="Extract natural frequencies and mode shapes."><HelpOutlineIcon sx={{ ml: 1 }} /></Tooltip>
                </AccordionSummary>
                <AccordionDetails>
                  <Button onClick={handleModal} variant="contained" color="primary" sx={{ mr: 2 }} disabled={loading}>
                    {loading ? <CircularProgress size={20} /> : 'Run Modal Analysis'}
                  </Button>
                  {modalResult && <Box sx={{ mt: 2 }}><Typography variant="subtitle1">Modal Result</Typography><pre>{JSON.stringify(modalResult, null, 2)}</pre></Box>}
                </AccordionDetails>
              </Accordion>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Time History Analysis</Typography>
                  <Tooltip title="Run time history (transient) analysis. Provide F_time and parameters as JSON."><HelpOutlineIcon sx={{ ml: 1 }} /></Tooltip>
                </AccordionSummary>
                <AccordionDetails>
                  <TextField label="Parameters (JSON)" multiline minRows={3} maxRows={8} fullWidth sx={{ mb: 2 }} defaultValue="{}" />
                  <Button onClick={handleTimeHistory} variant="contained" color="info" sx={{ mr: 2 }} disabled={loading}>
                    {loading ? <CircularProgress size={20} /> : 'Run Time History'}
                  </Button>
                  {timeHistoryResult && <Box sx={{ mt: 2 }}><Typography variant="subtitle1">Time History Result</Typography><pre>{JSON.stringify(timeHistoryResult, null, 2)}</pre></Box>}
                </AccordionDetails>
              </Accordion>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Response Spectrum Analysis</Typography>
                  <Tooltip title="Run response spectrum analysis. Provide spectrum and parameters as JSON."><HelpOutlineIcon sx={{ ml: 1 }} /></Tooltip>
                </AccordionSummary>
                <AccordionDetails>
                  <TextField label="Parameters (JSON)" multiline minRows={3} maxRows={8} fullWidth sx={{ mb: 2 }} defaultValue="{}" />
                  <Button onClick={handleResponseSpectrum} variant="contained" color="success" sx={{ mr: 2 }} disabled={loading}>
                    {loading ? <CircularProgress size={20} /> : 'Run Response Spectrum'}
                  </Button>
                  {responseSpectrumResult && <Box sx={{ mt: 2 }}><Typography variant="subtitle1">Response Spectrum Result</Typography><pre>{JSON.stringify(responseSpectrumResult, null, 2)}</pre></Box>}
                </AccordionDetails>
              </Accordion>
              <Divider sx={{ my: 3 }} />
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Import IFC</Typography>
                  <Tooltip title="Import a model from an IFC file."><HelpOutlineIcon sx={{ ml: 1 }} /></Tooltip>
                </AccordionSummary>
                <AccordionDetails>
                  <input type="file" accept=".ifc" onChange={e => handleImportIFC(e.target.files[0])} />
                  {importIFCMessage && <Typography variant="body2" color="success.main">{importIFCMessage}</Typography>}
                </AccordionDetails>
              </Accordion>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Export Report</Typography>
                  <Tooltip title="Export a report in CSV, PDF, or Excel format."><HelpOutlineIcon sx={{ ml: 1 }} /></Tooltip>
                </AccordionSummary>
                <AccordionDetails>
                  <Button onClick={() => handleExportReport('csv')} variant="outlined" sx={{ mr: 1 }}>Export CSV</Button>
                  <Button onClick={() => handleExportReport('pdf')} variant="outlined" sx={{ mr: 1 }}>Export PDF</Button>
                  <Button onClick={() => handleExportReport('excel')} variant="outlined">Export Excel</Button>
                  {reportExportMessage && <Typography variant="body2" color="success.main">{reportExportMessage}</Typography>}
                </AccordionDetails>
              </Accordion>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Available Sections & Materials</Typography>
                  <Tooltip title="Browse available sections and materials."><HelpOutlineIcon sx={{ ml: 1 }} /></Tooltip>
                </AccordionSummary>
                <AccordionDetails>
                  <Typography variant="h6">Sections</Typography>
                  <Box sx={{ bgcolor: '#f4f4f4', p: 2, borderRadius: 2, mb: 2 }}>{sections.join(', ')}</Box>
                  <Typography variant="h6">Materials</Typography>
                  <Box sx={{ bgcolor: '#f4f4f4', p: 2, borderRadius: 2 }}>{materials.join(', ')}</Box>
                </AccordionDetails>
              </Accordion>
            </Paper>
          )}
          {tab === 4 && (
            <Paper elevation={3} sx={{ p: 4 }}>
              <Typography variant="h5" gutterBottom color="primary">2D View</Typography>
              <Box sx={{ mb: 2 }}>
                <FormControlLabel control={<Switch checked={showLabels} onChange={e => setShowLabels(e.target.checked)} />} label="Show Element Labels" />
                <FormControlLabel control={<Switch checked={showNodeNumbers} onChange={e => setShowNodeNumbers(e.target.checked)} />} label="Show Node Numbers" />
              </Box>
              <Box sx={{ border: '1px solid #ccc', borderRadius: 2, bgcolor: '#fff', p: 2, minHeight: 400, position: 'relative', overflow: 'hidden' }}>
                <svg
                  width={600}
                  height={400}
                  style={{ background: '#fafafa', cursor: svgDragging ? 'grabbing' : 'grab' }}
                  onWheel={handleSvgWheel}
                  onMouseDown={handleSvgMouseDown}
                  onMouseMove={handleSvgMouseMove}
                  onMouseUp={handleSvgMouseUp}
                  onMouseLeave={handleSvgMouseUp}
                >
                  <g transform={`translate(${svgPan.x},${svgPan.y}) scale(${svgZoom})`}>
                    {/* Draw beams */}
                    {beams.map((b, i) => {
                      const s = b.start_node.split(',').map(Number);
                      const e = b.end_node.split(',').map(Number);
                      const isSelected = selected2DElement && selected2DElement.type === 'beam' && selected2DElement.idx === i;
                      return (
                        <Tooltip key={i} title={`Beam ${i + 1}`}><line x1={s[0]*40+200} y1={100-s[1]*40} x2={e[0]*40+200} y2={100-e[1]*40} stroke={isSelected ? '#ff9800' : '#1976d2'} strokeWidth={isSelected ? 6 : 3} onClick={() => handleSvgElementClick('beam', i)} style={{ cursor: 'pointer' }} /></Tooltip>
                      );
                    })}
                    {/* Draw columns */}
                    {columns.map((c, i) => {
                      const s = c.base_node.split(',').map(Number);
                      const e = c.top_node.split(',').map(Number);
                      const isSelected = selected2DElement && selected2DElement.type === 'column' && selected2DElement.idx === i;
                      return (
                        <Tooltip key={i} title={`Column ${i + 1}`}><line x1={s[0]*40+200} y1={100-s[1]*40} x2={e[0]*40+200} y2={100-e[1]*40} stroke={isSelected ? '#ff9800' : '#388e3c'} strokeWidth={isSelected ? 6 : 3} onClick={() => handleSvgElementClick('column', i)} style={{ cursor: 'pointer' }} /></Tooltip>
                      );
                    })}
                    {/* Draw nodes as circles */}
                    {Array.from(new Set([
                      ...beams.flatMap(b => [b.start_node, b.end_node]),
                      ...columns.flatMap(c => [c.base_node, c.top_node])
                    ])).map((n, i) => {
                      const p = n.split(',').map(Number);
                      return <Tooltip key={i} title={`Node ${i + 1}`}><circle cx={p[0]*40+200} cy={100-p[1]*40} r={8} fill="#ff9800" stroke="#333" onClick={() => handleSvgElementClick('node', i)} style={{ cursor: 'pointer' }} />
                        {showNodeNumbers && <text x={p[0]*40+210} y={100-p[1]*40} fontSize="14" fill="#333">{i + 1}</text>}
                      </Tooltip>;
                    })}
                    {/* Draw labels */}
                    {showLabels && beams.map((b, i) => {
                      const s = b.start_node.split(',').map(Number);
                      const e = b.end_node.split(',').map(Number);
                      const mx = (s[0]+e[0])/2*40+200;
                      const my = (s[1]+e[1])/2*-40+100;
                      return <text key={i} x={mx} y={my-10} fontSize="14" fill="#1976d2">B{i+1}</text>;
                    })}
                    {showLabels && columns.map((c, i) => {
                      const s = c.base_node.split(',').map(Number);
                      const e = c.top_node.split(',').map(Number);
                      const mx = (s[0]+e[0])/2*40+200;
                      const my = (s[1]+e[1])/2*-40+100;
                      return <text key={i} x={mx} y={my-10} fontSize="14" fill="#388e3c">C{i+1}</text>;
                    })}
                  </g>
                </svg>
                {/* Side panel for selected element info */}
                {selected2DElement && (
                  <Box sx={{ position: 'absolute', top: 20, right: 20, bgcolor: '#fff', border: '1px solid #1976d2', borderRadius: 2, p: 2, minWidth: 180, zIndex: 2 }}>
                    <Typography variant="subtitle1">{selected2DElement.type.charAt(0).toUpperCase() + selected2DElement.type.slice(1)} Info</Typography>
                    <Typography variant="body2">Index: {selected2DElement.idx + 1}</Typography>
                    {/* Add more info as needed */}
                    <Button size="small" sx={{ mt: 1 }} onClick={() => setSelected2DElement(null)}>Close</Button>
                  </Box>
                )}
              </Box>
            </Paper>
          )}
          {tab === 5 && (
            <Paper elevation={3} sx={{ p: 4 }}>
              <Typography variant="h5" gutterBottom color="primary">3D Viewer</Typography>
              <Box sx={{ mb: 2 }}>
                <FormControlLabel control={<Switch checked={wireframeMode} onChange={e => setWireframeMode(e.target.checked)} />} label="Wireframe Mode" />
              </Box>
              <Button variant="contained" color="primary" sx={{ mb: 2 }} onClick={fetchGeometry} disabled={loading}>
                {loading ? <CircularProgress size={20} /> : 'Load 3D Geometry'}
              </Button>
              {geometry.elements.length > 0 && (
                <Box sx={{ position: 'relative' }}>
                  <Canvas camera={{ position: [5, 5, 5], fov: 50 }} style={{ height: 400, background: '#222' }}>
                    <ambientLight intensity={0.5} />
                    <pointLight position={[10, 10, 10]} />
                    <OrbitControls />
                    {geometry.elements.map((el, i) => (
                      <mesh
                        key={i}
                        position={[0, 0, 0]}
                        onClick={e => handle3DElementClick(el, e)}
                        onPointerOver={e => e.target.material.color.set('#ff9800')}
                        onPointerOut={e => e.target.material.color.set(el.type === 'BeamElement' ? 'orange' : '#1976d2')}
                      >
                        <bufferGeometry attach="geometry">
                          <bufferAttribute
                            attach="attributes-position"
                            count={el.nodes.length}
                            array={new Float32Array(el.nodes.flat())}
                            itemSize={3}
                          />
                        </bufferGeometry>
                        <lineBasicMaterial attach="material" color={el.type === 'BeamElement' ? 'orange' : '#1976d2'} linewidth={wireframeMode ? 2 : 6} />
                      </mesh>
                    ))}
                  </Canvas>
                  {/* Info popover for selected element */}
                  <Popover
                    open={Boolean(anchorEl)}
                    anchorEl={anchorEl}
                    onClose={handle3DPopoverClose}
                    anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
                    transformOrigin={{ vertical: 'top', horizontal: 'left' }}
                  >
                    <Box sx={{ p: 2, minWidth: 200 }}>
                      <Typography variant="subtitle1">Element Info</Typography>
                      <Typography variant="body2">Type: {selected3DElement?.type}</Typography>
                      <Typography variant="body2">Nodes: {selected3DElement?.nodes?.map(n => `(${n.join(',')})`).join(', ')}</Typography>
                      <Button size="small" sx={{ mt: 1 }} onClick={handle3DPopoverClose}>Close</Button>
                    </Box>
                  </Popover>
                  {/* Legend */}
                  <Box sx={{ position: 'absolute', bottom: 10, right: 10, bgcolor: '#fff', border: '1px solid #1976d2', borderRadius: 2, p: 1, zIndex: 2 }}>
                    <Typography variant="caption" color="#1976d2">Beam: Orange</Typography><br />
                    <Typography variant="caption" color="#388e3c">Column: Green</Typography>
                  </Box>
                </Box>
              )}
            </Paper>
          )}
          {tab === 6 && (
            <Paper elevation={3} sx={{ p: 4 }}>
              <Typography variant="h5" gutterBottom color="primary">Expert / All Features</Typography>
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Undo / Redo</Typography>
                  <Tooltip title="Undo or redo the last model edit."><HelpOutlineIcon sx={{ ml: 1 }} /></Tooltip>
                </AccordionSummary>
                <AccordionDetails>
                  <Button variant="outlined" sx={{ mr: 1 }} onClick={async () => {
                    setLoading(true);
                    try {
                      const res = await axios.post('http://localhost:8000/model/undo');
                      setSnackbar({ open: true, message: res.data.status + ': ' + res.data.result, severity: 'success' });
                    } catch (e) {
                      setSnackbar({ open: true, message: 'Undo failed', severity: 'error' });
                    }
                    setLoading(false);
                  }}>Undo</Button>
                  <Button variant="outlined" onClick={async () => {
                    setLoading(true);
                    try {
                      const res = await axios.post('http://localhost:8000/model/redo');
                      setSnackbar({ open: true, message: res.data.status + ': ' + res.data.result, severity: 'success' });
                    } catch (e) {
                      setSnackbar({ open: true, message: 'Redo failed', severity: 'error' });
                    }
                    setLoading(false);
                  }}>Redo</Button>
                </AccordionDetails>
              </Accordion>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Direct Beam/Column Creation</Typography>
                  <Tooltip title="Add beams or columns directly to the model."><HelpOutlineIcon sx={{ ml: 1 }} /></Tooltip>
                </AccordionSummary>
                <AccordionDetails>
                  {/* ... existing direct beam/column creation UI ... */}
                </AccordionDetails>
              </Accordion>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Section / Material Details</Typography>
                  <Tooltip title="Query details for a section or material."><HelpOutlineIcon sx={{ ml: 1 }} /></Tooltip>
                </AccordionSummary>
                <AccordionDetails>
                  {/* ... existing section/material details UI ... */}
                </AccordionDetails>
              </Accordion>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Load Case / Combination / Code Combos</Typography>
                  <Tooltip title="Create load cases, combinations, or code-based combos."><HelpOutlineIcon sx={{ ml: 1 }} /></Tooltip>
                </AccordionSummary>
                <AccordionDetails>
                  {/* ... existing load case/combo UI ... */}
                </AccordionDetails>
              </Accordion>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Section Optimization</Typography>
                  <Tooltip title="Optimize element sections for efficiency."><HelpOutlineIcon sx={{ ml: 1 }} /></Tooltip>
                </AccordionSummary>
                <AccordionDetails>
                  {/* ... existing optimization UI ... */}
                </AccordionDetails>
              </Accordion>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Staged Construction Analysis</Typography>
                  <Tooltip title="Analyze staged construction scenarios. Provide stages as JSON."><HelpOutlineIcon sx={{ ml: 1 }} /></Tooltip>
                </AccordionSummary>
                <AccordionDetails>
                  <TextField label="Stages (JSON)" multiline minRows={3} maxRows={8} fullWidth sx={{ mr: 2, width: 400 }} id="staged-construction-stages" />
                  {/* ... existing staged construction UI ... */}
                </AccordionDetails>
              </Accordion>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Pushover Analysis</Typography>
                  <Tooltip title="Run pushover analysis. Set control node, direction, steps, and max displacement."><HelpOutlineIcon sx={{ ml: 1 }} /></Tooltip>
                </AccordionSummary>
                <AccordionDetails>
                  {/* ... existing pushover UI ... */}
                </AccordionDetails>
              </Accordion>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Mass / Damping Matrices</Typography>
                  <Tooltip title="Query the global mass and damping matrices."><HelpOutlineIcon sx={{ ml: 1 }} /></Tooltip>
                </AccordionSummary>
                <AccordionDetails>
                  {/* ... existing mass/damping matrix UI ... */}
                </AccordionDetails>
              </Accordion>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Export PDF Drawing</Typography>
                  <Tooltip title="Export a PDF drawing of the model."><HelpOutlineIcon sx={{ ml: 1 }} /></Tooltip>
                </AccordionSummary>
                <AccordionDetails>
                  {/* ... existing PDF drawing export UI ... */}
                </AccordionDetails>
              </Accordion>
            </Paper>
          )}
        </Container>
        <Snackbar open={snackbar.open} autoHideDuration={4000} onClose={() => setSnackbar({ ...snackbar, open: false })}>
          <Alert onClose={() => setSnackbar({ ...snackbar, open: false })} severity={snackbar.severity} sx={{ width: '100%' }}>
            {snackbar.message}
          </Alert>
        </Snackbar>
        <Box sx={{ mt: 8, py: 4, bgcolor: '#1976d2', color: 'white', textAlign: 'center' }}>
          <Typography variant="body2">
            &copy; {new Date().getFullYear()} Structural Insight Nexus. All rights reserved.
          </Typography>
        </Box>
      </Box>
    </Box>
  );
}

export default App;
