import React, { useState, useEffect, useRef } from 'react';
import './TrainTrafficControl.css';
// Updated track sections with Indian station names (without STN_ prefix in display)
const TRACK_SECTIONS = [
  // Main line (horizontal) - Major stations
  { id: 'STN_A',   x: 50,  y: 300, width: 60, height: 8,  type: 'station', station: 'A', platforms: 4, name: 'New Delhi - STN_A', category: 'major' },
  { id: 'BLOCK_A1',x: 120, y: 300, width: 60, height: 8,  type: 'block',   name: 'Block A1' },
  { id: 'BLOCK_A2',x: 190, y: 300, width: 60, height: 8,  type: 'block',   name: 'Block A2' },
  { id: 'STN_B',   x: 260, y: 300, width: 60, height: 8,  type: 'junction', station: 'B', platforms: 3, name: 'Ghaziabad Jn - STN_B', category: 'junction' },
  { id: 'BLOCK_B1',x: 330, y: 300, width: 60, height: 8,  type: 'block',   name: 'Block B1' },
  { id: 'BLOCK_B2',x: 400, y: 300, width: 60, height: 8,  type: 'block',   name: 'Block B2' },
  { id: 'STN_C',   x: 470, y: 300, width: 60, height: 8,  type: 'junction', station: 'C', platforms: 3, name: 'Kanpur Jn - STN_C', category: 'junction' },
  { id: 'BLOCK_C1',x: 540, y: 300, width: 60, height: 8,  type: 'block',   name: 'Block C1' },
  { id: 'BLOCK_C2',x: 610, y: 300, width: 60, height: 8,  type: 'block',   name: 'Block C2' },
  { id: 'STN_D',   x: 680, y: 300, width: 60, height: 8,  type: 'station', station: 'D', platforms: 2, name: 'Howrah - STN_D', category: 'terminal' },
  
  // Northern branch
  { id: 'STN_E',   x: 50,  y: 180, width: 60, height: 8,  type: 'station', station: 'E', platforms: 3, name: 'Chandigarh - STN_E', category: 'major' },
  { id: 'BLOCK_E1',x: 120, y: 180, width: 60, height: 8,  type: 'block',   name: 'Block E1' },
  { id: 'BLOCK_E2',x: 190, y: 180, width: 60, height: 8,  type: 'block',   name: 'Block E2' },
  { id: 'STN_F',   x: 260, y: 180, width: 60, height: 8,  type: 'junction', station: 'F', platforms: 2, name: 'Ambala Jn - STN_F', category: 'junction' },
  { id: 'BLOCK_F1',x: 330, y: 180, width: 60, height: 8,  type: 'block',   name: 'Block F1' },
  { id: 'BLOCK_F2',x: 400, y: 180, width: 60, height: 8,  type: 'block',   name: 'Block F2' },
  { id: 'STN_G',   x: 470, y: 180, width: 60, height: 8,  type: 'station', station: 'G', platforms: 2, name: 'Amritsar - STN_G', category: 'major' },
  
  // Upper branch
  { id: 'STN_H',   x: 190, y: 60,  width: 60, height: 8,  type: 'station', station: 'H', platforms: 2, name: 'Shimla - STN_H', category: 'hill' },
  { id: 'BLOCK_H1',x: 260, y: 60,  width: 60, height: 8,  type: 'block',   name: 'Block H1' },
  { id: 'BLOCK_H2',x: 330, y: 60,  width: 60, height: 8,  type: 'block',   name: 'Block H2' },
  { id: 'STN_I',   x: 400, y: 60,  width: 60, height: 8,  type: 'station', station: 'I', platforms: 2, name: 'Manali - STN_I', category: 'hill' },
  
  // Southern branch
  { id: 'STN_J',   x: 120, y: 420, width: 60, height: 8,  type: 'station', station: 'J', platforms: 3, name: 'Chennai - STN_J', category: 'major' },
  { id: 'BLOCK_J1',x: 190, y: 420, width: 60, height: 8,  type: 'block',   name: 'Block J1' },
  { id: 'BLOCK_J2',x: 260, y: 420, width: 60, height: 8,  type: 'block',   name: 'Block J2' },
  { id: 'STN_K',   x: 330, y: 420, width: 60, height: 8,  type: 'junction', station: 'K', platforms: 2, name: 'Vijayawada Jn - STN_K', category: 'junction' },
  { id: 'BLOCK_K1',x: 400, y: 420, width: 60, height: 8,  type: 'block',   name: 'Block K1' },
  { id: 'STN_L',   x: 470, y: 420, width: 60, height: 8,  type: 'station', station: 'L', platforms: 2, name: 'Kochi - STN_L', category: 'coastal' },
  
  // Connecting blocks for junctions
  { id: 'BLOCK_V_A_E', x: 50,  y: 240, width: 60, height: 8, type: 'block', name: 'V-Block (A-E)' },
  { id: 'BLOCK_V_A_J', x: 85,  y: 360, width: 60, height: 8, type: 'block', name: 'V-Block (A-J)' },
  { id: 'BLOCK_V_B_F', x: 260, y: 240, width: 60, height: 8, type: 'block', name: 'V-Block (B-F)' },
  { id: 'BLOCK_V_F_H', x: 225, y: 120, width: 60, height: 8, type: 'block', name: 'V-Block (F-H)' },
  { id: 'BLOCK_V_B_K', x: 295, y: 360, width: 60, height: 8, type: 'block', name: 'V-Block (B-K)' },
  { id: 'BLOCK_V_C_G', x: 470, y: 240, width: 60, height: 8, type: 'block', name: 'V-Block (C-G)' },
];

// Indian train names
const INDIAN_TRAIN_NAMES = [
  { number: '12951', name: 'Mumbai Rajdhani' },
  { number: '12301', name: 'Howrah Rajdhani' },
  { number: '12009', name: 'Shatabdi Express' },
  { number: '12267', name: 'Mumbai Duronto' },
  { number: '22691', name: 'Rajdhani Express' },
  { number: '12002', name: 'Bhopal Shatabdi' },
  { number: '12875', name: 'Neelachal Express' },
  { number: '12626', name: 'Kerala Express' },
  { number: '12841', name: 'Coromandel Express' },
  { number: '12460', name: 'Kanyakumari Express' },
];

// Expanded connections (unchanged)
const CONNECTIONS = [
  // Main line connections
  { from: 'STN_A',    to: 'BLOCK_A1',  type: 'main', path: `M80,304 L150,304` }, 
  { from: 'BLOCK_A1', to: 'BLOCK_A2',  type: 'main', path: `M150,304 L220,304` },
  { from: 'BLOCK_A2', to: 'STN_B',     type: 'main', path: `M220,304 L290,304` }, 
  { from: 'STN_B',    to: 'BLOCK_B1',  type: 'main', path: `M290,304 L360,304` },
  { from: 'BLOCK_B1', to: 'BLOCK_B2',  type: 'main', path: `M360,304 L430,304` }, 
  { from: 'BLOCK_B2', to: 'STN_C',     type: 'main', path: `M430,304 L500,304` },
  { from: 'STN_C',    to: 'BLOCK_C1',  type: 'main', path: `M500,304 L570,304` },
  { from: 'BLOCK_C1', to: 'BLOCK_C2',  type: 'main', path: `M570,304 L640,304` },
  { from: 'BLOCK_C2', to: 'STN_D',     type: 'main', path: `M640,304 L710,304` },
  
  // Northern branch connections
  { from: 'STN_E',    to: 'BLOCK_E1',  type: 'branch', path: `M80,184 L150,184` }, 
  { from: 'BLOCK_E1', to: 'BLOCK_E2',  type: 'branch', path: `M150,184 L220,184` },
  { from: 'BLOCK_E2', to: 'STN_F',     type: 'branch', path: `M220,184 L290,184` }, 
  { from: 'STN_F',    to: 'BLOCK_F1',  type: 'branch', path: `M290,184 L360,184` },
  { from: 'BLOCK_F1', to: 'BLOCK_F2',  type: 'branch', path: `M360,184 L430,184` }, 
  { from: 'BLOCK_F2', to: 'STN_G',     type: 'branch', path: `M430,184 L500,184` },
  
  // Upper branch connections
  { from: 'STN_H',    to: 'BLOCK_H1',  type: 'branch', path: `M220,64 L290,64` }, 
  { from: 'BLOCK_H1', to: 'BLOCK_H2',  type: 'branch', path: `M290,64 L360,64` },
  { from: 'BLOCK_H2', to: 'STN_I',     type: 'branch', path: `M360,64 L430,64` },
  
  // Southern branch connections
  { from: 'STN_J',    to: 'BLOCK_J1',  type: 'branch', path: `M150,424 L220,424` }, 
  { from: 'BLOCK_J1', to: 'BLOCK_J2',  type: 'branch', path: `M220,424 L290,424` },
  { from: 'BLOCK_J2', to: 'STN_K',     type: 'branch', path: `M290,424 L360,424` }, 
  { from: 'STN_K',    to: 'BLOCK_K1',  type: 'branch', path: `M360,424 L430,424` },
  { from: 'BLOCK_K1', to: 'STN_L',     type: 'branch', path: `M430,424 L500,424` },
  
  // Junction connections (vertical)
  { from: 'STN_A',    to: 'BLOCK_V_A_E', type: 'junction', path: `M80,300 L80,244` },
  { from: 'BLOCK_V_A_E', to: 'STN_E',    type: 'junction', path: `M80,244 L80,188` },
  { from: 'STN_A',    to: 'BLOCK_V_A_J', type: 'junction', path: `M80,308 L115,364` },
  { from: 'BLOCK_V_A_J', to: 'STN_J',    type: 'junction', path: `M115,364 L150,424` },
  { from: 'STN_B',    to: 'BLOCK_V_B_F', type: 'junction', path: `M290,300 L290,244` },
  { from: 'BLOCK_V_B_F', to: 'STN_F',    type: 'junction', path: `M290,244 L290,188` },
  { from: 'STN_F',    to: 'BLOCK_V_F_H', type: 'junction', path: `M255,180 L255,124` },
  { from: 'BLOCK_V_F_H', to: 'STN_H',    type: 'junction', path: `M255,124 L220,68` },
  { from: 'STN_B',    to: 'BLOCK_V_B_K', type: 'junction', path: `M290,308 L325,364` },
  { from: 'BLOCK_V_B_K', to: 'STN_K',    type: 'junction', path: `M325,364 L360,424` },
  { from: 'STN_C',    to: 'BLOCK_V_C_G', type: 'junction', path: `M500,300 L500,244` },
  { from: 'BLOCK_V_C_G', to: 'STN_G',    type: 'junction', path: `M500,244 L500,188` },

];

const TrainTrafficControl = () => {
  // Existing state variables...
  const [trains, setTrains] = useState([]);
  const [blockOccupancy, setBlockOccupancy] = useState({});
  const [stationPlatforms, setStationPlatforms] = useState({});
  const [simulationTime, setSimulationTime] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [trainProgress, setTrainProgress] = useState({});
  const [hoveredTrain, setHoveredTrain] = useState(null);
  const [selectedTrain, setSelectedTrain] = useState(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [currentTime, setCurrentTime] = useState(new Date());
  const [activeMenuItem, setActiveMenuItem] = useState('live-monitoring');
  const [activeButtons, setActiveButtons] = useState({
    overview: true, signals: false, speed: false, alerts: false
  });
  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  
  // Enhanced state for ML and optimization features
  const [metrics, setMetrics] = useState({ throughput: 0, avgDelay: 0, utilization: 0, avgSpeed: 0 });
  const [enhancedMetrics, setEnhancedMetrics] = useState({ 
    on_time_percentage: 100, ml_accuracy: 0, recommendations_accepted: 0, total_recommendations: 0 
  });
  const [mlPredictions, setMlPredictions] = useState({});
  const [optimizationRecommendations, setOptimizationRecommendations] = useState([]);
  const [notifications, setNotifications] = useState([]);
  const [showDelayInjector, setShowDelayInjector] = useState(false);
  const [selectedTrainForDelay, setSelectedTrainForDelay] = useState('');
  const [delayMinutes, setDelayMinutes] = useState(5);


  const API_BASE_URL = 'http://localhost:8000';
  const WS_URL = 'ws://localhost:8000/ws';

  const menuItems = [
    { id: 'live-monitoring', label: 'Live Monitoring', icon: 'standard', category: 'operations' },
    { id: 'schedule-view', label: 'Schedule View', icon: 'schedule', category: 'operations' },
    { id: 'station-status', label: 'Station Status', icon: 'standard', category: 'operations' },
    { id: 'ml-predictions', label: 'ML Predictions', icon: 'ai', category: 'ai' },
    { id: 'optimization', label: 'Optimization', icon: 'optimization', category: 'optimization' },
    { id: 'audit-trail', label: 'Audit Trail', icon: 'standard', category: 'operations' },
    { id: 'performance-dashboard', label: 'Performance Dashboard', icon: 'analysis', category: 'analysis' },
  ];

  // Helper functions to format time and get station name
  const formatTime = (ticks) => {
    const hours = Math.floor(ticks / 60);
    const minutes = ticks % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
  };

  // Modified helper function to get station display with both code and name
  const getStationDisplayName = (stationId) => {
    const section = TRACK_SECTIONS.find(s => s.id === stationId);
    if (!section) return stationId;
    return `${section.name} - ${stationId}`;
  };

  const getStationName = (stationId) => {
    const section = TRACK_SECTIONS.find(s => s.id === stationId);
    return section ? section.name : stationId.replace('STN_', '');
  };

  const getStatusColor = (statusType, waitingForBlock) => {
    if (waitingForBlock) return '#ff6b6b'; // Red for waiting
    switch (statusType) {
      case 'running': return '#51cf66'; // Green for running
      case 'scheduled': return '#74c0fc'; // Blue for scheduled
      case 'completed': return '#868e96'; // Gray for completed
      default: return '#ffd43b'; // Yellow for others
    }
  };

  // WebSocket connection logic
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket(WS_URL);

        ws.onopen = () => { 
          console.log('WebSocket connected'); 
          setConnected(true); 
          setError(null); 
          setLoading(false); 
        };
        ws.onmessage = (event) => {
          try { 
            const data = JSON.parse(event.data); 
            updateSystemState(data); 
          }
          catch (err) { 
            console.error('Error parsing WebSocket message:', err); 
          }
        };
        ws.onclose = () => {
          console.log('WebSocket disconnected'); 
          setConnected(false);
          reconnectTimeoutRef.current = setTimeout(() => { 
            console.log('Attempting to reconnect...'); 
            connectWebSocket(); 
          }, 3000);
        };
        ws.onerror = (error) => { 
          console.error('WebSocket error:', error); 
          setError('Connection failed'); 
          setLoading(false); 
        };
        wsRef.current = ws;
      } catch (err) { 
        console.error('Failed to create WebSocket connection:', err); 
        setError('Failed to connect to backend'); 
        setLoading(false); 
      }

    };


    connectWebSocket();

    return () => {
      if (wsRef.current) { wsRef.current.close(); }
      if (reconnectTimeoutRef.current) { clearTimeout(reconnectTimeoutRef.current); }
    };
  }, []);

  // Mock data generation with Indian train names
  useEffect(() => {
    if (!connected) {
      const mockTrains = INDIAN_TRAIN_NAMES.slice(0, 8).map((trainData, index) => ({
        id: `train_${index + 1}`,
        number: trainData.number,
        name: trainData.name,
        section: ['STN_A', 'STN_B', 'STN_E', 'STN_F'][index % 4],
        speed: 45 + Math.random() * 30,
        statusType: Math.random() > 0.2 ? 'running' : 'waiting',
        status: Math.random() > 0.2 ? 'On Time' : 'Delayed',
        waitingForBlock: Math.random() > 0.7,
        route: ['STN_A', 'BLOCK_A1', 'STN_B', 'STN_C'],
        departureTime: index * 5,
        destination: 'D'
      }));
      setTrains(mockTrains);
      setLoading(false);
    }
  }, [connected]);

  const updateSystemState = (data) => {
    const updatedTrains = (data.trains || []).map((train, index) => ({
      ...train,
      number: train.number || INDIAN_TRAIN_NAMES[index % INDIAN_TRAIN_NAMES.length]?.number || `1234${index}`,
      name: train.name || INDIAN_TRAIN_NAMES[index % INDIAN_TRAIN_NAMES.length]?.name || `Express ${index + 1}`
    }));
    
    setTrains(updatedTrains);
    setBlockOccupancy(data.blockOccupancy || {});
    setStationPlatforms(data.stationPlatforms || {});
    setSimulationTime(data.simulationTime || 0);
    setIsRunning(data.isRunning || false);
    setTrainProgress(data.trainProgress || {});
    setMetrics(data.metrics || { throughput: 0, avgDelay: 0, utilization: 0, avgSpeed: 0 });



    setEnhancedMetrics(data.enhancedMetrics || { on_time_percentage: 100, ml_accuracy: 0, recommendations_accepted: 0, total_recommendations: 0 });
    setMlPredictions(data.mlPredictions || {});
    setOptimizationRecommendations(data.optimizationRecommendations || []);

    if (data.events && data.events.length > 0) {
      const newNotifications = data.events.map(eventText => ({
        id: Date.now() + Math.random(),
        text: eventText,
      }));
      setNotifications(prev => [...newNotifications, ...prev].slice(0, 20)); 
    }
  };

  // Control functions
  const controlSimulation = async (action) => {
    try {
      const response = await fetch(`${API_BASE_URL}/simulation-control`, {
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' }, 
        body: JSON.stringify({ action }),
      });
      if (!response.ok) { 
        throw new Error(`HTTP error! status: ${response.status}`); 
      }
      await response.json();
      console.log(`Simulation ${action} successful`);
    } catch (err) { 
      console.error(`Failed to ${action} simulation:`, err);
      setError(`Failed to ${action} simulation: ${err.message}`); 
      setTimeout(() => setError(null), 5000);
    }
  };

  const injectDelay = async () => {
    if (!selectedTrainForDelay) return;
    
    try {
      const response = await fetch(`${API_BASE_URL}/inject-delay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          train_id: selectedTrainForDelay, 
          delay_minutes: delayMinutes 
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to inject delay');
      }
      
      const result = await response.json();
      console.log('Delay injection successful:', result);
      
      setShowDelayInjector(false);
      setSelectedTrainForDelay('');
      
      setNotifications(prev => [{
        id: Date.now(),
        text: `Delay injected: ${delayMinutes}min delay added to selected train`,
      }, ...prev].slice(0, 20));
      
    } catch (err) {
      console.error('Failed to inject delay:', err);
      setError(`Failed to inject delay: ${err.message}`);
      setTimeout(() => setError(null), 5000);
    }
  };

  const applyOptimization = async (recommendationId, accept) => {
    try {
      const response = await fetch(`${API_BASE_URL}/apply-optimization`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          recommendation_id: recommendationId, 
          accept 
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to ${accept ? 'apply' : 'reject'} optimization`);
      }
      
      const result = await response.json();
      console.log(`Optimization ${accept ? 'applied' : 'rejected'} successfully:`, result);
      
      setNotifications(prev => [{
        id: Date.now(),
        text: `Optimization ${accept ? 'applied' : 'rejected'}: ${result.message}`,
      }, ...prev].slice(0, 20));
      
    } catch (err) {
      console.error(`Failed to ${accept ? 'apply' : 'reject'} optimization:`, err);
      setError(`Failed to ${accept ? 'apply' : 'reject'} optimization: ${err.message}`);
      setTimeout(() => setError(null), 5000);
    }
  };

  useEffect(() => {
    const clock = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(clock);
  }, []);

  // Helper functions
  const getSectionState = (sectionId) => {
    const section = TRACK_SECTIONS.find(s => s.id === sectionId);
    if (!section) return 'free';

    if (section.type === 'block') { 
      return blockOccupancy[sectionId] ? 'occupied' : 'free'; 
    }
    if (section.type === 'station' || section.type === 'junction') {
      const platforms = stationPlatforms[sectionId] || {};
      const occupied = Object.values(platforms).filter(Boolean).length;
      if (occupied === 0) return 'free';
      if (occupied < (section.platforms || 1)) return 'partial';
      return 'occupied';
    }
    return 'free';
  };
  
  const getTrainsInSection = (sectionId) => trains.filter(train => train.section === sectionId);
  const getSectionCenter = (section) => ({ x: section.x + section.width / 2, y: section.y + section.height / 2 });
  const handleMouseMove = (e) => setMousePos({ x: e.clientX, y: e.clientY });
  const handleTrainClick = (train, event) => { 
    event.stopPropagation(); 
    setSelectedTrain(selectedTrain?.id === train.id ? null : train); 
  };

  const [hoverTimeout, setHoverTimeout] = useState(null);
  const handleTrainHover = (train, event) => { 
    event.stopPropagation(); 
    if (hoverTimeout) clearTimeout(hoverTimeout);
    setHoveredTrain(train); 
  };

  const handleTrainLeave = (event) => { 
    event.stopPropagation(); 
    const timeout = setTimeout(() => setHoveredTrain(null), 150);
    setHoverTimeout(timeout);
  };

  const handleButtonClick = (buttonName) => setActiveButtons(prev => ({ ...prev, [buttonName]: !prev[buttonName] }));
  const handleMenuItemClick = (itemId) => setActiveMenuItem(itemId);
  const handleSimulationControl = (action) => controlSimulation(action);
  const getRouteIndex = (trainId) => (trainProgress[trainId]?.currentRouteIndex || 0);
  
  const freeBlocksCount = () => {
    let totalSlots = 0;
    let occupiedSlots = 0;
    totalSlots += Object.keys(blockOccupancy).length;
    occupiedSlots += Object.keys(blockOccupancy).filter(id => blockOccupancy[id] !== null).length;
    Object.values(stationPlatforms).forEach(platformMap => {
      totalSlots += Object.keys(platformMap).length;
      occupiedSlots += Object.values(platformMap).filter(occupant => occupant !== null).length;
    });
    return totalSlots - occupiedSlots;
  };

  const getStationOccupancyStatus = (stationId) => {
    const platforms = stationPlatforms[stationId] || {};
    const occupied = Object.values(platforms).filter(Boolean).length;
    const total = Object.keys(platforms).length;
    return { occupied, total, percentage: total > 0 ? (occupied / total) * 100 : 0 };
  };

  const getStationIcon = (section) => {
    if (section.type === 'junction') {
      return '⊕';
    }
    switch (section.category) {
      case 'major': return '◼';
      case 'terminal': return '◆';
      case 'hill': return '▲';
      case 'coastal': return '⬟';
      default: return '●';
    }
  };

  // Modified Schedule View Component
  const renderScheduleView = () => (
    <div className="panel-section">
      <div className="panel-header">TRAIN SCHEDULE</div>
      <div className="schedule-controls">
        <div className="time-display">Current Time: {formatTime(simulationTime)}</div>
      </div>
      <div className="schedule-table">
        <div className="schedule-header">
          <div className="col-train">Train</div>
          <div className="col-route">Route</div>
          <div className="col-departure">Departure</div>
          <div className="col-status">Status</div>
          <div className="col-delay">Delay</div>
        </div>
        {trains.map(train => {
          const departureTime = formatTime(train.departureTime || 0);
          const startStation = getStationDisplayName(train.route?.[0] || '');
          const endStation = getStationDisplayName(train.route?.[train.route?.length - 1] || `STN_${train.destination}`);
          const prediction = mlPredictions[train.id];
          const actualDelay = prediction ? prediction.predicted_delay : 0;
          
          return (
            <div key={train.id} className={`schedule-row ${train.statusType} ${selectedTrain?.id === train.id ? 'selected' : ''}`}
                 onClick={() => setSelectedTrain(selectedTrain?.id === train.id ? null : train)}>
              <div className="col-train">
                <div className="train-number">{train.number}</div>
                <div className="train-name">{train.name}</div>
              </div>
              <div className="col-route">
                <span className="route-start">{startStation}</span>
                <span className="route-arrow">→</span>
                <span className="route-end">{endStation}</span>
              </div>
              <div className="col-departure">
                <div className="scheduled-time">{departureTime}</div>
                <div className="priority">P{train.priority || 99}</div>
              </div>
              <div className="col-status">
                <div className={`status-indicator ${train.statusType}`} 
                     style={{ backgroundColor: getStatusColor(train.statusType, train.waitingForBlock) }}></div>
                <span className={`status-text ${train.waitingForBlock ? 'waiting' : ''}`}>
                  {train.waitingForBlock ? 'WAITING' : train.status}
                </span>
              </div>
              <div className="col-delay">
                <span className={`delay-value ${actualDelay > 3 ? 'high-delay' : actualDelay > 0 ? 'moderate-delay' : 'on-time'}`}>
                  {actualDelay > 0 ? `+${actualDelay}` : '0'}
                </span>
                {train.injected_delay > 0 && (
                  <span className="injected-delay-indicator">
                    (Inj: {train.injected_delay}m)
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );

  // Modified Station Status Component with dual display
  const renderStationStatus = () => {
    const stations = TRACK_SECTIONS.filter(s => s.type === 'station' || s.type === 'junction');
    
    return (
      <div className="panel-section">
        <div className="panel-header">STATION STATUS</div>
        <div className="station-status-grid">
          {stations.map(station => {
            const status = getStationOccupancyStatus(station.id);
            const platforms = stationPlatforms[station.id] || {};
            
            return (
              <div key={station.id} className={`station-status-card ${station.type}`}>
                <div className="station-header">
                  <div className="station-name">
                    <span className="station-icon">{getStationIcon(station)}</span>
                    <div className="station-name-group">
                      <span className="station-code">{station.station}</span>
                      <span className="station-name-text">{station.name}</span>
                    </div>
                    {station.type === 'junction' && <span className="junction-label">JN</span>}
                  </div>
                  <div className={`occupancy-badge ${status.percentage > 80 ? 'high' : status.percentage > 50 ? 'medium' : 'low'}`}>
                    {status.occupied}/{status.total}
                  </div>
                </div>
                <div className="platform-details">
                  {Object.entries(platforms).map(([platformNum, occupant]) => (
                    <div key={platformNum} className={`platform-status ${occupant ? 'occupied' : 'free'}`}>
                      <span className="platform-label">P{platformNum}</span>
                      <span className="platform-occupant">{occupant || 'FREE'}</span>
                    </div>
                  ))}
                </div>
                <div className="station-utilization">
                  <div className="utilization-bar">
                    <div 
                      className="utilization-fill" 
                      style={{ width: `${status.percentage}%` }}
                    ></div>
                  </div>
                  <span className="utilization-text">{status.percentage.toFixed(0)}% utilized</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderMLPredictions = () => (
    <div className="panel-section">
      <div className="panel-header">ML ETA PREDICTIONS</div>
      <div className="ml-accuracy-display">
        Model Accuracy: {(enhancedMetrics.ml_accuracy * 100).toFixed(1)}%
      </div>
      {Object.entries(mlPredictions).length === 0 ? (
        <div className="no-predictions">No active predictions available</div>
      ) : (
        Object.entries(mlPredictions).map(([trainId, prediction]) => {
          const train = trains.find(t => t.id === trainId);
          if (!train) return null;
          
          return (
            <div key={trainId} className="prediction-item">
              <div className="prediction-header">
                <span className="train-name">{train.number}</span>
                <span className={`delay-indicator ${prediction.predicted_delay > 3 ? 'high-delay' : 'normal'}`}>
                  {prediction.predicted_delay > 0 ? `+${prediction.predicted_delay}` : prediction.predicted_delay}
                </span>
              </div>
              <div className="prediction-details">
                <div>Predicted ETA: {prediction.predicted_eta} ticks</div>
                <div>Confidence: {(prediction.confidence * 100).toFixed(0)}%</div>
              </div>
            </div>
          );
        })
      )}
    </div>
  );

  const renderOptimizationPanel = () => (
    <div className="panel-section">
      <div className="panel-header">OPTIMIZATION RECOMMENDATIONS</div>
      <div className="optimization-controls">
        <button 
          className="control-btn optimization-btn"
          onClick={() => setShowDelayInjector(true)}
          disabled={trains.filter(t => t.statusType !== 'completed').length === 0}
        >
          INJECT DELAY
        </button>
        <div className="optimization-stats">
          <div className="stat-item">
            <span className="stat-label">Accepted:</span>
            <span className="stat-value">{enhancedMetrics.recommendations_accepted}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Total:</span>
            <span className="stat-value">{enhancedMetrics.total_recommendations}</span>
          </div>
        </div>
      </div>
      
      {optimizationRecommendations.length === 0 ? (
        <div className="no-recommendations">
          <div className="no-rec-icon">🎯</div>
          <div className="no-rec-text">No optimization recommendations at this time.</div>
          <div className="no-rec-subtext">System is running optimally</div>
        </div>
      ) : (
        <div className="recommendations-list">
          {optimizationRecommendations.map((rec, index) => (
            <div key={rec.id || index} className="recommendation-item">
              <div className="rec-header">
                <span className={`rec-type ${rec.type}`}>
                  {rec.type === 'speed_adjustment' ? '⚡ SPEED' : 
                   rec.type === 'priority_adjustment' ? '🏆 PRIORITY' : 
                   '🛤️ ROUTE'}
                </span>
                <span className="train-number">{rec.train_number}</span>
                <span className="confidence-badge">{(rec.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="rec-reason">{rec.reason}</div>
              {rec.type === 'speed_adjustment' && (
                <div className="rec-change">
                  Speed: {rec.current_value} → {rec.recommended_value} km/h
                </div>
              )}
              {rec.type === 'priority_adjustment' && (
                <div className="rec-change">
                  Priority: P{rec.current_value} → P{rec.recommended_value}
                </div>
              )}
              <div className="rec-actions">
                <button 
                  className="rec-btn accept"
                  onClick={() => applyOptimization(rec.id, true)}
                >
                  ✓ ACCEPT
                </button>
                <button 
                  className="rec-btn reject"
                  onClick={() => applyOptimization(rec.id, false)}
                >
                  ✗ REJECT
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const renderPerformanceMetrics = () => (
    <div className="panel-section">
      <div className="panel-header">ENHANCED METRICS</div>
      <div className="enhanced-metrics-grid">
        <div className="metric-card">
          <div className="metric-icon">⏱️</div>
          <div className="metric-label">On-Time %</div>
          <div className="metric-value green">{enhancedMetrics.on_time_percentage.toFixed(1)}%</div>
        </div>
        <div className="metric-card">
          <div className="metric-icon">🤖</div>
          <div className="metric-label">ML Accuracy</div>
          <div className="metric-value blue">{(enhancedMetrics.ml_accuracy * 100).toFixed(1)}%</div>
        </div>
        <div className="metric-card">
          <div className="metric-icon">📈</div>
          <div className="metric-label">Throughput</div>
          <div className="metric-value orange">{metrics.throughput.toFixed(2)} t/hr</div>
        </div>
        <div className="metric-card">
          <div className="metric-icon">⏰</div>
          <div className="metric-label">Avg Delay</div>
          <div className="metric-value red">{metrics.avgDelay.toFixed(1)} ticks</div>
        </div>
        <div className="metric-card">
          <div className="metric-icon">📊</div>
          <div className="metric-label">Utilization</div>
          <div className="metric-value purple">{metrics.utilization.toFixed(1)}%</div>
        </div>
        <div className="metric-card">
          <div className="metric-icon">💡</div>
          <div className="metric-label">Optimizations</div>
          <div className="metric-value cyan">
            {enhancedMetrics.recommendations_accepted}/{enhancedMetrics.total_recommendations}
          </div>
        </div>
      </div>
      
      <div className="performance-charts">
        <div className="chart-container">
          <div className="chart-title">Network Health</div>
          <div className="health-indicators">
            <div className={`health-indicator ${metrics.utilization < 70 ? 'good' : metrics.utilization < 85 ? 'warning' : 'critical'}`}>
              <div className="health-label">Capacity</div>
              <div className="health-bar">
                <div className="health-fill" style={{width: `${metrics.utilization}%`}}></div>
              </div>
              <div className="health-value">{metrics.utilization.toFixed(0)}%</div>
            </div>
            <div className={`health-indicator ${enhancedMetrics.on_time_percentage > 90 ? 'good' : enhancedMetrics.on_time_percentage > 80 ? 'warning' : 'critical'}`}>
              <div className="health-label">On-Time Performance</div>
              <div className="health-bar">
                <div className="health-fill" style={{width: `${enhancedMetrics.on_time_percentage}%`}}></div>
              </div>
              <div className="health-value">{enhancedMetrics.on_time_percentage.toFixed(0)}%</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderDelayInjector = () => (
    showDelayInjector && (
      <div className="modal-overlay" onClick={() => setShowDelayInjector(false)}>
        <div className="delay-injector-modal" onClick={e => e.stopPropagation()}>
          <div className="modal-header">
            <h3>⚠️ Inject Delay (Testing)</h3>
            <button className="close-btn" onClick={() => setShowDelayInjector(false)}>×</button>
          </div>
          <div className="modal-content">
            <div className="form-group">
              <label>Select Train:</label>
              <select 
                value={selectedTrainForDelay} 
                onChange={e => setSelectedTrainForDelay(e.target.value)}
              >
                <option value="">Choose a train...</option>
                {trains.filter(t => t.statusType !== 'completed').map(train => (
                  <option key={train.id} value={train.id}>
                    {train.number} - {train.name} (Priority: P{train.priority || 99})
                  </option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label>Delay (minutes): {delayMinutes}</label>
              <input 
                type="range" 
                min="1" 
                max="30" 
                value={delayMinutes}
                onChange={e => setDelayMinutes(parseInt(e.target.value))}
              />
              <div className="delay-slider-labels">
                <span>1 min</span>
                <span>15 min</span>
                <span>30 min</span>
              </div>
            </div>
            <div className="modal-actions">
              <button 
                className="inject-btn" 
                onClick={injectDelay} 
                disabled={!selectedTrainForDelay}
              >
                🎯 INJECT DELAY
              </button>
              <button 
                className="cancel-btn" 
                onClick={() => setShowDelayInjector(false)}
              >
                CANCEL
              </button>
            </div>
          </div>
        </div>
      </div>
    )
  );

  if (loading) { 
    return ( 
      <div className="tms-container">
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <div className="loading-text">Connecting to Railway Control System...</div>
          <div className="loading-subtext">Initializing ML models and optimization engines...</div>
        </div>
      </div> 
    ); 
  }

  if (loading) { return ( <div className="tms-container"><div className="loading-overlay"><div className="loading-spinner"></div></div></div> ); }

  return (
    <div className="tms-container" onMouseMove={handleMouseMove}>
      <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
        {connected ? '● BACKEND CONNECTED' : '● BACKEND DISCONNECTED (MOCK DATA)'}
      </div>
      <div className="tms-header">
        <div className="header-left">
          <div className="system-title">INDIAN RAILWAY CONTROL SYSTEM</div>
          <div className="system-subtitle">ML-POWERED TRAFFIC MANAGEMENT</div>
        </div>
        <div className="header-center">
          <div className="status-group">
            <div className="status-display green">{freeBlocksCount()}</div>
            <div className="status-label">FREE SLOTS</div>
          </div>
          <div className="status-group">
            <div className="status-display blue">{String(trains.filter(t => t.statusType === 'running').length).padStart(2, '0')}</div>
            <div className="status-label">ACTIVE</div>
          </div>
          <div className="status-group">
            <div className="status-display orange">{String(trains.filter(t => t.waitingForBlock).length).padStart(2, '0')}</div>
            <div className="status-label">WAITING</div>
          </div>
          <div className="status-group">
            <div className="status-display purple">{String(optimizationRecommendations.length).padStart(2, '0')}</div>
            <div className="status-label">RECOMMENDATIONS</div>
          </div>
          <div className="time-display">
            {currentTime.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </div>
        </div>
        <div className="header-right">
          <div className="control-buttons">
            <button className={`control-btn ${activeButtons.overview ? 'active' : ''}`} onClick={() => handleButtonClick('overview')}>OVERVIEW</button>
            <button className={`control-btn ${activeButtons.signals ? 'active' : ''}`} onClick={() => handleButtonClick('signals')}>SIGNALS</button>
            <button className={`control-btn ${activeButtons.speed ? 'active' : ''}`} onClick={() => handleButtonClick('speed')}>SPEED</button>
            <button className={`control-btn ${activeButtons.alerts ? 'active' : ''}`} onClick={() => handleButtonClick('alerts')}>ALERTS</button>
          </div>
          <div className="compass">N</div>
        </div>
      </div>
      
      <div className="main-display">
        <div className="track-container">
          <svg className="track-svg" viewBox="0 0 800 500">
            {CONNECTIONS.map((conn, index) => 
              <path key={index} d={conn.path} className="connection-line" />
            )}
            {TRACK_SECTIONS.map(section => {
              const state = getSectionState(section.id);
              const trainsInSection = getTrainsInSection(section.id);
              const isSelected = selectedTrain && trainsInSection.some(t => t.id === selectedTrain.id);
              return (
                <g key={section.id}>
                  <rect x={section.x} y={section.y} width={section.width} height={section.height}
                    className={`track-section ${section.type === 'station' ? 'track-station' : section.type === 'junction' ? 'track-junction' : 'track-block'} ${state === 'occupied' ? 'track-occupied' : state === 'partial' ? 'track-partial' : 'track-free'} ${isSelected ? 'track-selected' : ''}`}
                    rx="4" />
                  
                  {/* Station/Junction Name Display */}
                  {(section.type === 'station' || section.type === 'junction') && (
                    <>
                      <text x={section.x + section.width / 2} y={section.y - 15} className="station-name-main">
                        {section.name}
                      </text>
                      {section.type === 'junction' && (
                        <text x={section.x + section.width / 2} y={section.y - 28} className="junction-indicator-main">JN</text>
                      )}
                    </>
                  )}
                  
                  {/* Block ID for non-station sections */}
                  {section.type === 'block' && (
                    <text x={section.x + section.width / 2} y={section.y - 8} className="section-id-label">{section.id}</text>
                  )}
                  
                  {/* Platform indicators */}
                  {(section.type === 'station' || section.type === 'junction') && (
                    <>
                      <text x={section.x + section.width / 2} y={section.y + 25} className="platform-count-label">{section.platforms}P</text>
                      <g className="platform-indicators">
                        {Object.entries(stationPlatforms[section.id] || {}).map(([platformNum, occupant], idx) => (
                          <g key={platformNum}>
                            <circle cx={section.x + 15 + (idx * 15)} cy={section.y + 35} r="5" 
                                    className={`platform-indicator ${occupant ? 'occupied' : 'free'}`} />
                            <text x={section.x + 15 + (idx * 15)} y={section.y + 39} className="platform-number">{platformNum}</text>
                          </g>
                        ))}
                      </g>
                    </>
                  )}
                  
                  {/* Train representations */}
                  {trainsInSection.map((train, trainIndex) => {
                    const center = getSectionCenter(section);
                    let offsetY = 0, offsetX = 0;
                    if (section.type === 'station' || section.type === 'junction') { 
                      offsetY = (trainIndex * 18) - ((trainsInSection.length - 1) * 9); 
                      offsetX = (trainIndex * 10) - ((trainsInSection.length - 1) * 5); 
                    }
                    const isTrainSelected = selectedTrain?.id === train.id;
                    const hasPrediction = mlPredictions[train.id];
                    const hasHighDelay = hasPrediction && mlPredictions[train.id].predicted_delay > 3;
                    
                    return (
                      <g key={train.id} className={`train-group ${isTrainSelected ? 'selected' : ''} ${train.waitingForBlock ? 'waiting' : ''}`} 
                         onClick={(e) => handleTrainClick(train, e)} 
                         onMouseEnter={(e) => handleTrainHover(train, e)} 
                         onMouseLeave={handleTrainLeave}>
                        <rect x={center.x - 20 + offsetX} y={center.y - 10 + offsetY} width={40} height={20} rx="10" 
                              className={`train-body train-${train.statusType} ${isTrainSelected ? 'train-selected' : ''} ${train.waitingForBlock ? 'train-waiting' : ''} ${hasHighDelay ? 'train-high-delay' : ''}`} />
                        <text x={center.x + offsetX} y={center.y + offsetY + 3} className="train-number-label">{train.number}</text>
                        {train.waitingForBlock && 
                          <circle cx={center.x + 25 + offsetX} cy={center.y - 5 + offsetY} r="4" className="waiting-indicator" />
                        }
                        {hasHighDelay && 
                          <circle cx={center.x - 25 + offsetX} cy={center.y - 5 + offsetY} r="4" className="delay-warning-indicator" />
                        }
                      </g>
                    );
                  })}
                </g>
              );
            })}
          </svg>
        </div>
        
        <div className="simulation-controls">
          <div className="control-row">
            <button onClick={() => handleSimulationControl(isRunning ? 'pause' : 'start')} 
                    className={`sim-btn ${isRunning ? 'pause' : 'start'}`} disabled={!connected}>
              {isRunning ? 'PAUSE' : 'START'}
            </button>
            <button onClick={() => handleSimulationControl('reset')} className="sim-btn reset" disabled={!connected}>
              RESET
            </button>
          </div>
          <div className="sim-time">SIM TIME: {String(Math.floor(simulationTime / 60)).padStart(2, '0')}:{String(simulationTime % 60).padStart(2, '0')}</div>
          <div className="sim-stats">
            <span className="stat-running">RUN: {trains.filter(t => t.statusType === 'running').length}</span>
            <span className="stat-waiting">WAIT: {trains.filter(t => t.waitingForBlock).length}</span>
            <span className="stat-completed">DONE: {trains.filter(t => t.statusType === 'completed').length}</span>
          </div>
          
          <div className="notification-panel">
            <div className="notification-header">SYSTEM EVENTS</div>
            {notifications.slice(0, 4).map(notif => (
              <div key={notif.id} className="notification-item">
                <div className="notification-time">{new Date().toLocaleTimeString()}</div>
                <div className="notification-text">{notif.text}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="control-panel">
        <div className="panel-section">
          <div className="panel-header">NAVIGATION</div>
          {menuItems.map(item => (
            <div key={item.id} 
                 className={`menu-item ${activeMenuItem === item.id ? 'active' : ''} ${item.id === 'audit-trail' ? 'audit-trail-item' : ''}`} 
                 onClick={() => handleMenuItemClick(item.id)}>
              <div className={`menu-icon ${item.icon}`}></div>
              {item.label}
              {item.id === 'optimization' && optimizationRecommendations.length > 0 && (
                <div className="menu-badge">{optimizationRecommendations.length}</div>
              )}
            </div>
          ))}
        </div>

        {activeMenuItem === 'live-monitoring' && (
          <>
            <div className="panel-section">
              <div className="panel-header">BLOCK STATUS</div>
              <div className="block-status-grid">
                {Object.entries(blockOccupancy).slice(0, 12).map(([blockId, occupant]) => (
                  <div key={blockId} className={`block-status-item ${occupant ? 'occupied' : 'free'}`}>
                    <div className="block-id">{blockId}</div>
                    <div className="block-occupant">{occupant || 'FREE'}</div>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="panel-section">
              <div className="panel-header">ACTIVE TRAINS ({trains.length})</div>
              <div className="train-list">
                {trains.map(train => {
                  const currentSection = TRACK_SECTIONS.find(s => s.id === train.section);
                  const currentLocationDisplay = currentSection ? 
                    `${currentSection.name} - ${train.section}` : 
                    train.section;
                  const isSelected = selectedTrain?.id === train.id;
                  const routeIndex = getRouteIndex(train.id);
                  const prediction = mlPredictions[train.id];
                  
                  return (
                    <div key={train.id} 
                         className={`train-item ${isSelected ? 'selected' : ''} ${train.waitingForBlock ? 'waiting' : ''}`} 
                         onClick={() => setSelectedTrain(isSelected ? null : train)}>
                      <div className={`train-status-dot ${train.statusType} ${train.waitingForBlock ? 'waiting' : ''}`}></div>
                      <div className="train-details">
                        <div className="train-name">{train.name}</div>
                        <div className="train-info">
                          {train.number} | {currentLocationDisplay} | {Math.round(train.speed)} km/h
                          {prediction && prediction.predicted_delay > 0 && 
                            <span className="predicted-delay"> | ML: +{prediction.predicted_delay}min</span>
                          }
                          {train.waitingForBlock && <span className="waiting-status"> | WAITING</span>}
                        </div>
                        <div className="train-route-info">
                          Progress: {routeIndex + 1}/{train.route?.length || 0} | Priority: P{train.priority || 99}
                          {train.injected_delay > 0 && 
                            <span className="injected-delay"> | Delayed: {train.injected_delay}min</span>
                          }
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </>
        )}

        {activeMenuItem === 'schedule-view' && renderScheduleView()}
        {activeMenuItem === 'station-status' && renderStationStatus()}
        {activeMenuItem === 'ml-predictions' && renderMLPredictions()}
        {activeMenuItem === 'optimization' && renderOptimizationPanel()}
        {activeMenuItem === 'performance-dashboard' && renderPerformanceMetrics()}

        {activeMenuItem === 'audit-trail' && (
          <div className="panel-section audit-trail-panel">
            <div className="panel-header">RECENT SYSTEM EVENTS</div>
            <div className="audit-trail">
              {notifications.map(notif => (
                <div key={notif.id} className="audit-item">
                  <div className="audit-timestamp">
                    {new Date().toLocaleTimeString()}
                  </div>
                  <div className="audit-message">{notif.text}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {hoveredTrain && (
        <div className="train-tooltip" style={{ 
          left: Math.min(mousePos.x + 20, window.innerWidth - 420), 
          top: Math.max(mousePos.y - 280, 10) 
        }}>
          <div className="tooltip-header">{hoveredTrain.name}</div>
          <div className="tooltip-content">
            <div className="tooltip-section">
              <div className="tooltip-row">
                <span className="tooltip-label">Train Number:</span>
                <span className="tooltip-value">{hoveredTrain.number}</span>
              </div>
              <div className="tooltip-row">
                <span className="tooltip-label">Current Speed:</span>
                <span className="tooltip-value tooltip-speed">{Math.round(hoveredTrain.speed)} km/h</span>
              </div>
              <div className="tooltip-row">
                <span className="tooltip-label">Current Location:</span>
                <span className="tooltip-value tooltip-section-id">
                  {getStationDisplayName(hoveredTrain.section)}
                </span>
              </div>
              <div className="tooltip-row">
                <span className="tooltip-label">Departure Time:</span>
                <span className="tooltip-value">{formatTime(hoveredTrain.departureTime || 0)}</span>
              </div>
              <div className="tooltip-row">
                <span className="tooltip-label">Status:</span>
                <span className={`tooltip-value tooltip-status ${hoveredTrain.statusType}`}>
                  {hoveredTrain.status}
                </span>
              </div>
              {mlPredictions[hoveredTrain.id] && (
                <>
                  <div className="tooltip-row">
                    <span className="tooltip-label">ML Predicted Delay:</span>
                    <span className={`tooltip-value ${mlPredictions[hoveredTrain.id].predicted_delay > 3 ? 'warning' : 'normal'}`}>
                      +{mlPredictions[hoveredTrain.id].predicted_delay} ticks
                    </span>
                  </div>
                  <div className="tooltip-row">
                    <span className="tooltip-label">Prediction Confidence:</span>
                    <span className="tooltip-value">
                      {(mlPredictions[hoveredTrain.id].confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </>
              )}
              <div className="tooltip-row">
                <span className="tooltip-label">Route Progress:</span>
                <span className="tooltip-value">
                  {getRouteIndex(hoveredTrain.id) + 1} of {hoveredTrain.route?.length || 0}
                </span>
              </div>
              <div className="tooltip-row">
                <span className="tooltip-label">Priority:</span>
                <span className="tooltip-value">P{hoveredTrain.priority || 99}</span>
              </div>
              {hoveredTrain.injected_delay > 0 && (
                <div className="tooltip-row">
                  <span className="tooltip-label">Injected Delay:</span>
                  <span className="tooltip-value warning">
                    {hoveredTrain.injected_delay} minutes
                  </span>
                </div>
              )}
              {hoveredTrain.waiting_since && (
                <div className="tooltip-row">
                  <span className="tooltip-label">Waiting Since:</span>
                  <span className="tooltip-value warning">
                    Tick {hoveredTrain.waiting_since}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {renderDelayInjector()}

      {error && (
        <div className="error-notification">
          <div className="error-content">
            <span className="error-icon">⚠</span>
            <span className="error-message">{error}</span>
          </div>
          <button onClick={() => setError(null)} className="error-close">×</button>
        </div>
      )}
      
      {optimizationRecommendations.length > 0 && (
        <div className="floating-recommendations">
          <div className="floating-rec-header">
            <span className="floating-rec-count">{optimizationRecommendations.length}</span>
            <span className="floating-rec-text">New Recommendations</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default TrainTrafficControl;