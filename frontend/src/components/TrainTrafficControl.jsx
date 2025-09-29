import React, { useState, useEffect, useRef, useMemo } from 'react';
import './TrainTrafficControl.css';

// Expanded track sections with more stations and blocks
const TRACK_SECTIONS = [
  // Main line (horizontal)
  { id: 'STN_A',   x: 50,  y: 300, width: 60, height: 8,  type: 'station', station: 'A', platforms: 4, name: 'New Delhi', category: 'major' },
  { id: 'BLOCK_A1',x: 120, y: 300, width: 60, height: 8,  type: 'block',   name: 'Block A1' },
  { id: 'BLOCK_A2',x: 190, y: 300, width: 60, height: 8,  type: 'block',   name: 'Block A2' },
  { id: 'STN_B',   x: 260, y: 300, width: 60, height: 8,  type: 'junction', station: 'B', platforms: 3, name: 'Ghaziabad Jn', category: 'junction' },
  { id: 'BLOCK_B1',x: 330, y: 300, width: 60, height: 8,  type: 'block',   name: 'Block B1' },
  { id: 'BLOCK_B2',x: 400, y: 300, width: 60, height: 8,  type: 'block',   name: 'Block B2' },
  { id: 'STN_C',   x: 470, y: 300, width: 60, height: 8,  type: 'junction', station: 'C', platforms: 3, name: 'Kanpur Jn', category: 'junction' },
  { id: 'BLOCK_C1',x: 540, y: 300, width: 60, height: 8,  type: 'block',   name: 'Block C1' },
  { id: 'BLOCK_C2',x: 610, y: 300, width: 60, height: 8,  type: 'block',   name: 'Block C2' },
 { id: 'STN_D',   x: 680, y: 300, width: 60, height: 8,  type: 'station', station: 'D', platforms: 2, name: 'Howrah', category: 'terminal' },
  
  // Northern branch
 { id: 'STN_E',   x: 50,  y: 180, width: 60, height: 8,  type: 'station', station: 'E', platforms: 3, name: 'Chandigarh', category: 'major' },
  { id: 'BLOCK_E1',x: 120, y: 180, width: 60, height: 8,  type: 'block',   name: 'Block E1' },
  { id: 'BLOCK_E2',x: 190, y: 180, width: 60, height: 8,  type: 'block',   name: 'Block E2' },
  { id: 'STN_F',   x: 260, y: 180, width: 60, height: 8,  type: 'junction', station: 'F', platforms: 2, name: 'Ambala Jn', category: 'junction' },
   { id: 'BLOCK_F1',x: 330, y: 180, width: 60, height: 8,  type: 'block',   name: 'Block F1' },
  { id: 'BLOCK_F2',x: 400, y: 180, width: 60, height: 8,  type: 'block',   name: 'Block F2' },
  { id: 'STN_G',   x: 470, y: 180, width: 60, height: 8,  type: 'station', station: 'G', platforms: 2, name: 'Amritsar', category: 'major' },
  
  // Upper branch
  { id: 'STN_H',   x: 190, y: 60,  width: 60, height: 8,  type: 'station', station: 'H', platforms: 2, name: 'Shimla', category: 'hill' },
  { id: 'BLOCK_H1',x: 260, y: 60,  width: 60, height: 8,  type: 'block',   name: 'Block H1' },
  { id: 'BLOCK_H2',x: 330, y: 60,  width: 60, height: 8,  type: 'block',   name: 'Block H2' },
 { id: 'STN_I',   x: 400, y: 60,  width: 60, height: 8,  type: 'station', station: 'I', platforms: 2, name: 'Manali', category: 'hill' },
  
  // Southern branch
 { id: 'STN_J',   x: 120, y: 420, width: 60, height: 8,  type: 'station', station: 'J', platforms: 3, name: 'Chennai', category: 'major' },
  { id: 'BLOCK_J1',x: 190, y: 420, width: 60, height: 8,  type: 'block',   name: 'Block J1' },
  { id: 'BLOCK_J2',x: 260, y: 420, width: 60, height: 8,  type: 'block',   name: 'Block J2' },
 { id: 'STN_K',   x: 330, y: 420, width: 60, height: 8,  type: 'junction', station: 'K', platforms: 2, name: 'Vijayawada Jn', category: 'junction' },
  { id: 'BLOCK_K1',x: 400, y: 420, width: 60, height: 8,  type: 'block',   name: 'Block K1' },
  { id: 'STN_L',   x: 470, y: 420, width: 60, height: 8,  type: 'station', station: 'L', platforms: 2, name: 'Kochi', category: 'coastal' },
  // Connecting blocks for junctions
  { id: 'BLOCK_V_A_E', x: 50,  y: 240, width: 60, height: 8, type: 'block', name: 'V-Block (A-E)' },
  { id: 'BLOCK_V_A_J', x: 85,  y: 360, width: 60, height: 8, type: 'block', name: 'V-Block (A-J)' },
  { id: 'BLOCK_V_B_F', x: 260, y: 240, width: 60, height: 8, type: 'block', name: 'V-Block (B-F)' },
  { id: 'BLOCK_V_F_H', x: 225, y: 120, width: 60, height: 8, type: 'block', name: 'V-Block (F-H)' },
  { id: 'BLOCK_V_B_K', x: 295, y: 360, width: 60, height: 8, type: 'block', name: 'V-Block (B-K)' },
  { id: 'BLOCK_V_C_G', x: 470, y: 240, width: 60, height: 8, type: 'block', name: 'V-Block (C-G)' },
];
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
// Expanded connections
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
const junctionStationIds = new Set(
  TRACK_SECTIONS.filter(s => s.type === 'junction').map(s => s.id)
);

const TrainTrafficControl = () => {
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
  const [auditTrail, setAuditTrail] = useState([]);
  

  const API_BASE_URL = 'http://localhost:8000';
  const WS_URL = 'ws://localhost:8000/ws';

  const menuItems = [
    { id: 'live-monitoring', label: 'Live Monitoring', icon: 'standard', category: 'operations' },
    { id: 'station-status', label: 'Station Status', icon: 'standard', category: 'operations' },
    { id: 'ml-predictions', label: 'ML Predictions', icon: 'ai', category: 'ai' },
    { id: 'optimization', label: 'Optimization', icon: 'optimization', category: 'optimization' },
    { id: 'audit-trail', label: 'Audit Trail', icon: 'standard', category: 'operations' },
    { id: 'performance-dashboard', label: 'Performance Dashboard', icon: 'analysis', category: 'analysis' },
  ];

  // *** FIX STARTS HERE ***
  // Create an augmented version of stationPlatforms that includes completed trains for UI consistency.
  const augmentedStationPlatforms = useMemo(() => {
    // Start with a deep copy to avoid mutating state
    const newPlatforms = JSON.parse(JSON.stringify(stationPlatforms));
    const completedTrains = trains.filter(t => t.statusType === 'completed');

    for (const train of completedTrains) {
        const stationId = train.section;
        const station = TRACK_SECTIONS.find(s => s.id === stationId);

        // Only act if the train is in a station section
        if (station && (station.type === 'station' || station.type === 'junction')) {
            // Ensure the station exists in our new platform map
            if (!newPlatforms[stationId]) {
                newPlatforms[stationId] = {};
                for (let i = 1; i <= station.platforms; i++) {
                    newPlatforms[stationId][i] = null;
                }
            }

            const platforms = newPlatforms[stationId];
            const isTrainAlreadyPlaced = Object.values(platforms).includes(train.id);

            if (!isTrainAlreadyPlaced) {
                // Find the first free platform and place the completed train there for UI purposes
                const firstFreePlatform = Object.keys(platforms).find(pNum => platforms[pNum] === null);
                if (firstFreePlatform) {
                    newPlatforms[stationId][firstFreePlatform] = train.id;
                }
            }
        }
    }
    return newPlatforms;
  }, [trains, stationPlatforms]);
  // *** FIX ENDS HERE ***

    const addAuditLog = (type, message) => {
    const newLog = {
      id: Date.now() + Math.random(),
      timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
      type: type.toUpperCase(), // e.g., 'SYSTEM', 'USER', 'EVENT'
      message,
    };
    // Add the new log to the start of the array and keep the last 100 logs
    setAuditTrail(prev => [newLog, ...prev].slice(0, 100));
  };

  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket(WS_URL);
        ws.onopen = () => { 
        console.log('WebSocket connected'); 
        setConnected(true); 
        setError(null); 
        setLoading(false); 
        addAuditLog('SYSTEM', 'Connected to simulation backend.'); // Add this line
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
          addAuditLog('SYSTEM', 'Disconnected from simulation backend.');
          reconnectTimeoutRef.current = setTimeout(() => { 
            console.log('Attempting to reconnect...'); 
            addAuditLog('SYSTEM', 'Attempting to reconnect...');
            connectWebSocket(); 
          }, 3000);
        };
        ws.onerror = (error) => { 
          console.error('WebSocket error:', error); 
          setError('Connection failed'); 
          setLoading(false); 
          addAuditLog('SYSTEM', 'WebSocket connection error.');
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

  const updateSystemState = (data) => {
    setTrains(data.trains || []);
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
      data.events.forEach(eventText => addAuditLog('EVENT', eventText)); 
    }
  };

  const controlSimulation = async (action) => {
    addAuditLog('USER', `Simulation action: ${action.toUpperCase()}`);
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
    } catch (err) { 
      setError(`Failed to ${action} simulation`); 
      addAuditLog('SYSTEM', `Error during simulation control: ${action}`);
    }
  };

  const injectDelay = async () => {
    if (!selectedTrainForDelay) return;
    const trainNumber = trains.find(t => t.id === selectedTrainForDelay)?.number || 'Unknown';
  addAuditLog('USER', `Injecting ${delayMinutes} min delay for Train ${trainNumber}.`);
    try {
      const response = await fetch(`${API_BASE_URL}/inject-delay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          train_id: selectedTrainForDelay, 
          delay_minutes: delayMinutes 
        }),
      });
      if (!response.ok) throw new Error('Failed to inject delay');
      
      setShowDelayInjector(false);
      setSelectedTrainForDelay('');
    } catch (err) {
      setError('Failed to inject delay');
      addAuditLog('SYSTEM', `Failed to inject delay for Train ${trainNumber}.`);
    }
  };

  const applyOptimization = async (recommendationId, accept) => {
     const rec = optimizationRecommendations[recommendationId];
  if (rec) {
      const action = accept ? 'Accepted' : 'Rejected';
      addAuditLog('USER', `${action} recommendation for Train ${rec.train_number}: ${rec.reason}`); 
  }
    try {
      const response = await fetch(`${API_BASE_URL}/apply-optimization`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          recommendation_id: recommendationId, 
          accept 
        }),
      });
      if (!response.ok) throw new Error('Failed to apply optimization');
    } catch (err) {
      setError('Failed to apply optimization');
      addAuditLog('SYSTEM', 'Failed to apply optimization.');
    }
  };

  useEffect(() => {
    const clock = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(clock);
  }, []);
  
  const getSectionState = (sectionId) => {
    const section = TRACK_SECTIONS.find(s => s.id === sectionId);
    if (!section) return 'free';
    if (section.type === 'block') { 
      return blockOccupancy[sectionId] ? 'occupied' : 'free'; 
    }
    if (section.type === 'station' || section.type === 'junction') {
      const platforms = augmentedStationPlatforms[sectionId] || {}; // USE AUGMENTED DATA
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
  const handleTrainClick = (train, event) => { event.stopPropagation(); setSelectedTrain(selectedTrain?.id === train.id ? null : train); };
  const handleTrainHover = (train, event) => { event.stopPropagation(); setHoveredTrain(train); };
  const handleTrainLeave = () => setHoveredTrain(null);
  const handleButtonClick = (buttonName) => setActiveButtons(prev => ({ ...prev, [buttonName]: !prev[buttonName] }));
  const handleMenuItemClick = (itemId) => setActiveMenuItem(itemId);
  const handleSimulationControl = (action) => controlSimulation(action);
  const getRouteIndex = (trainId) => (trainProgress[trainId]?.currentRouteIndex || 0);
  
  // Helper function to convert ticks to minutes (assuming 1 tick = 1 minute)
  const ticksToMinutes = (ticks) => {
    return ticks; // 1 tick = 1 minute in this simulation
  };
  
  // Helper function to format minutes for display
  const formatMinutes = (minutes) => {
    if (minutes < 60) {
      return `${minutes} min`;
    } else {
      const hours = Math.floor(minutes / 60);
      const mins = minutes % 60;
      return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
    }
  };
  
  const freeBlocksCount = () => {
    let totalSlots = 0;
    let occupiedSlots = 0;
    totalSlots += Object.keys(blockOccupancy).length;
    occupiedSlots += Object.keys(blockOccupancy).filter(id => blockOccupancy[id] !== null).length;
    Object.values(augmentedStationPlatforms).forEach(platformMap => { // USE AUGMENTED DATA
      totalSlots += Object.keys(platformMap).length;
      occupiedSlots += Object.values(platformMap).filter(occupant => occupant !== null).length;
    });
    return totalSlots - occupiedSlots;
  };

  const getStationOccupancyStatus = (stationId) => {
    const platforms = augmentedStationPlatforms[stationId] || {}; // USE AUGMENTED DATA
    const occupied = Object.values(platforms).filter(Boolean).length;
    const total = Object.keys(platforms).length;
    return { occupied, total, percentage: total > 0 ? (occupied / total) * 100 : 0 };
  };
    const renderAuditTrail = () => (
    <div className="panel-section">
      <div className="panel-header">AUDIT TRAIL</div>
      <div className="audit-trail-container">
        {auditTrail.length === 0 ? (
          <div className="audit-trail-empty">No system events logged yet.</div>
        ) : (
          auditTrail.map(log => (
            <div key={log.id} className="audit-log-item">
              <div className="log-timestamp">{log.timestamp}</div>
              <div className="log-details">
                <span className={`log-type log-type-${log.type.toLowerCase()}`}>{log.type}</span>
                <span className="log-message">{log.message}</span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );

  const renderStationStatus = () => {
    const stations = TRACK_SECTIONS.filter(s => s.type === 'station' || s.type === 'junction');
    
    return (
      <div className="panel-section">
        <div className="panel-header">STATION STATUS</div>
        <div className="station-status-grid">
          {stations.map(station => {
            const status = getStationOccupancyStatus(station.id);
            const platforms = augmentedStationPlatforms[station.id] || {}; // USE AUGMENTED DATA
            
            return (
              <div key={station.id} className="station-status-card">
                <div className="station-header">
                  <div className="station-name">{station.name}</div>
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
      {Object.entries(mlPredictions).map(([trainId, prediction]) => {
        const train = trains.find(t => t.id === trainId);
        if (!train) return null;
        
        // Convert ticks to minutes for display
        const predictedDelayMinutes = ticksToMinutes(prediction.predicted_delay);
        const predictedETAMinutes = ticksToMinutes(prediction.predicted_eta);
        
        return (
          <div key={trainId} className="prediction-item">
            <div className="prediction-header">
              <span className="train-name">{train.number}</span>
              <span className={`delay-indicator ${predictedDelayMinutes > 3 ? 'high-delay' : 'normal'}`}>
                {predictedDelayMinutes > 0 ? `+${formatMinutes(predictedDelayMinutes)}` : 'On Time'}
              </span>
            </div>
            <div className="prediction-details">
              <div>Predicted ETA: {formatMinutes(predictedETAMinutes)}</div>
              <div>Confidence: {(prediction.confidence * 100).toFixed(0)}%</div>
            </div>
          </div>
        );
      })}
    </div>
  );

  const renderOptimizationPanel = () => (
    <div className="panel-section">
      <div className="panel-header">OPTIMIZATION RECOMMENDATIONS</div>
      <div className="optimization-controls">
        <button 
          className="control-btn optimization-btn"
          onClick={() => setShowDelayInjector(true)}
        >
          INJECT DELAY
        </button>
      </div>
      
      {optimizationRecommendations.map((rec, index) => (
        <div key={index} className="recommendation-item">
          <div className="rec-header">
            <span className={`rec-type ${rec.type}`}>
              {rec.type === 'speed_adjustment' ? '⚡ SPEED' : '🏆 PRIORITY'}
            </span>
            <span className="train-number">{rec.train_number}</span>
          </div>
          <div className="rec-details">{rec.reason}</div>
          <div className="rec-actions">
            <button 
              className="rec-btn accept"
              onClick={() => applyOptimization(index, true)}
            >
              ACCEPT
            </button>
            <button 
              className="rec-btn reject"
              onClick={() => applyOptimization(index, false)}
            >
              REJECT
            </button>
          </div>
        </div>
      ))}
      
      {optimizationRecommendations.length === 0 && (
        <div className="no-recommendations">No optimization recommendations at this time.</div>
      )}
    </div>
  );

  const renderPerformanceMetrics = () => (
    <div className="panel-section">
      <div className="panel-header">ENHANCED METRICS</div>
      <div className="enhanced-metrics-grid">
        <div className="metric-card">
          <div className="metric-label">On-Time %</div>
          <div className="metric-value green">{enhancedMetrics.on_time_percentage.toFixed(1)}%</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">ML Accuracy</div>
          <div className="metric-value blue">{(enhancedMetrics.ml_accuracy * 100).toFixed(1)}%</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Throughput</div>
          <div className="metric-value orange">{metrics.throughput.toFixed(2)} t/hr</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Avg Delay</div>
          <div className="metric-value red">{formatMinutes(metrics.avgDelay)}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Utilization</div>
          <div className="metric-value purple">{metrics.utilization.toFixed(1)}%</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Recommendations</div>
          <div className="metric-value cyan">
            {enhancedMetrics.recommendations_accepted}/{enhancedMetrics.total_recommendations}
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
            <h3>Inject Delay</h3>
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
                    {train.number} - {train.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label>Delay (minutes):</label>
              <input 
                type="number" 
                min="1" 
                max="30" 
                value={delayMinutes}
                onChange={e => setDelayMinutes(parseInt(e.target.value) || 5)}
              />
            </div>
            <div className="modal-actions">
              <button className="inject-btn" onClick={injectDelay} disabled={!selectedTrainForDelay}>
                INJECT DELAY
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
        </div>
      </div> 
    ); 
  }

  return (
    <div className="tms-container" onMouseMove={handleMouseMove}>
      <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
        {connected ? '● BACKEND CONNECTED' : '● BACKEND DISCONNECTED'}
      </div>
      <div className="tms-header">
        <div className="header-left">
          <div className="system-title">INTELLIGENT RAILWAY CONTROL SYSTEM</div>
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
              
              // --- POSITIONING CONSTANTS (Adjusted for better clarity) ---
              // Block ID positioning logic
              const isVerticalBlock = section.id.startsWith('BLOCK_V_');
              const blockIdYAbove = section.y - 8; // Standard position above track
              const blockIdYBelow = section.y + section.height + 15; // Position below track (Pushes block ID down)

              // Station label positioning constants (grouped below the track)
              const stationNameY = section.y + 25;
              const platformCountY = stationNameY + 13;
              const platformIndicatorBaseY = platformCountY + 12;
              const platformNumberY = platformIndicatorBaseY + 4;
              // --- END POSITIONING CONSTANTS ---

              // Paste this new code in its place
              return (
                <g key={section.id}>
                  {(() => {
                    const isJunction = junctionStationIds.has(section.id);

                    const platformIndicators = (baseY) => (
                      <g className="platform-indicators">
                        {Object.entries(augmentedStationPlatforms[section.id] || {}).map(([platformNum, occupant], idx) => { // USE AUGMENTED DATA
                          const totalPlatforms = section.platforms || 1;
                          const spacing = 18;
                          const startX = section.x + section.width / 2 - ((totalPlatforms - 1) * spacing / 2);
                          return (
                            <g key={platformNum}>
                              <circle cx={startX + (idx * spacing)} cy={baseY} r="7" className={`platform-indicator ${occupant != null ? 'occupied' : 'free'}`} />
                              <text x={startX + (idx * spacing)} y={baseY + 1} className="platform-number">{platformNum}</text>
                            </g>
                          );
                        })}
                      </g>
                    );

                    if (isJunction) {
                      return (
                        <>
                          <path
                            d={`M ${section.x} ${section.y + 4} L ${section.x + section.width} ${section.y + 4}`}
                            className={`track-section track-junction ${state === 'occupied' ? 'track-occupied' : state === 'partial' ? 'track-partial' : 'track-free'} ${isSelected ? 'track-selected' : ''}`}
                          />
                          <text x={section.x + section.width / 2} y={section.y - 14} className="station-name-label junction-name-label">
                            {section.name}
                          </text>
                          <text x={section.x + section.width / 2} y={section.y - 4} className="section-id-label">
                            {section.id}
                          </text>
                          <text x={section.x + section.width / 2} y={section.y + 24} className="platform-count-label">
                            {section.platforms}P
                          </text>
                          {platformIndicators(section.y + 38)}
                        </>
                      );
                    } 
                    
                    else {
                      return (
                        <>
                          <rect x={section.x} y={section.y} width={section.width} height={section.height}
                            className={`track-section ${section.type === 'station' ? 'track-station' : 'track-block'} ${state === 'occupied' ? 'track-occupied' : state === 'partial' ? 'track-partial' : 'track-free'} ${isSelected ? 'track-selected' : ''}`}
                            rx="4" />
                          <text
                            x={section.x + section.width / 2}
                            y={section.type === 'station' ? blockIdYAbove : (isVerticalBlock ? blockIdYBelow : blockIdYAbove)}
                            className="section-id-label"
                          >
                            {section.id}
                          </text>
                          {section.type === 'station' && (
                            <>
                              <text x={section.x + section.width / 2} y={stationNameY} className="station-name-label">{section.name}</text>
                              <text x={section.x + section.width / 2} y={platformCountY} className="platform-count-label">{section.platforms}P</text>
                              {platformIndicators(platformIndicatorBaseY)}
                            </>
                          )}
                        </>
                      );
                    }
                  })()}

      {/* Train Visualization (This part is the same for all sections) */}
      {trainsInSection.map((train, trainIndex) => {
        const center = getSectionCenter(section);
        let offsetY = 0, offsetX = 0;
        if (section.type === 'station' || section.type === 'junction') {
          offsetY = (trainIndex * 18) - ((trainsInSection.length - 1) * 9);
          offsetX = (trainIndex * 10) - ((trainsInSection.length - 1) * 5);
        }
        const isTrainSelected = selectedTrain?.id === train.id;
        const hasPrediction = mlPredictions[train.id];
        const predictedDelayMinutes = hasPrediction ? ticksToMinutes(mlPredictions[train.id]?.predicted_delay) : 0;
        const hasHighDelay = hasPrediction && predictedDelayMinutes > 3;

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
            {notifications.map(notif => (
              <div key={notif.id} className="notification-item">
                {notif.text}
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
                 className={`menu-item ${activeMenuItem === item.id ? 'active' : ''}`} 
                 onClick={() => handleMenuItemClick(item.id)}>
              <div className={`menu-icon ${item.icon}`}></div>
              {item.label}
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
              {trains.map(train => {
                const currentSection = TRACK_SECTIONS.find(s => s.id === train.section);
                const isSelected = selectedTrain?.id === train.id;
                const routeIndex = getRouteIndex(train.id);
                const prediction = mlPredictions[train.id];
                const predictedDelayMinutes = prediction ? ticksToMinutes(prediction.predicted_delay) : 0;
                
                return (
                  <div key={train.id} 
                       className={`train-item ${isSelected ? 'selected' : ''} ${train.waitingForBlock ? 'waiting' : ''}`} 
                       onClick={() => setSelectedTrain(isSelected ? null : train)}>
                    <div className={`train-status-dot ${train.statusType} ${train.waitingForBlock ? 'waiting' : ''}`}></div>
                    <div className="train-details">
                      <div className="train-name">{train.name}</div>
                      <div className="train-info">
                        {train.number} | {currentSection?.name || train.section} | {Math.round(train.speed)} km/h
                        {prediction && predictedDelayMinutes > 0 && 
                          <span className="predicted-delay"> | ML: +{formatMinutes(predictedDelayMinutes)}</span>
                        }
                        {train.waitingForBlock && <span className="waiting-status"> | WAITING</span>}
                      </div>
                      <div className="train-route-info">
                        Progress: {routeIndex + 1}/{train.route?.length || 0}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}

        {activeMenuItem === 'station-status' && renderStationStatus()}
        {activeMenuItem === 'ml-predictions' && renderMLPredictions()}
        {activeMenuItem === 'optimization' && renderOptimizationPanel()}
         {activeMenuItem === 'audit-trail' && renderAuditTrail()}
        {activeMenuItem === 'performance-dashboard' && renderPerformanceMetrics()}
      </div>

      {hoveredTrain && (
        <div className="train-tooltip" style={{ 
          left: Math.min(mousePos.x + 20, window.innerWidth - 420), 
          top: Math.max(mousePos.y - 250, 10) 
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
                  {TRACK_SECTIONS.find(s => s.id === hoveredTrain.section)?.name || hoveredTrain.section}
                </span>
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
                    <span className={`tooltip-value ${ticksToMinutes(mlPredictions[hoveredTrain.id].predicted_delay) > 3 ? 'warning' : 'normal'}`}>
                      +{formatMinutes(ticksToMinutes(mlPredictions[hoveredTrain.id].predicted_delay))}
                    </span>
                  </div>
                  <div className="tooltip-row">
                    <span className="tooltip-label">Predicted ETA:</span>
                    <span className="tooltip-value">
                      {formatMinutes(ticksToMinutes(mlPredictions[hoveredTrain.id].predicted_eta))}
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
            </div>
          </div>
        </div>
      )}

      {renderDelayInjector()}

      {error && (
        <div className="error-notification">
          {error}
          <button onClick={() => setError(null)} className="error-close">×</button>
        </div>
      )}
    </div>
  );
};

export default TrainTrafficControl;