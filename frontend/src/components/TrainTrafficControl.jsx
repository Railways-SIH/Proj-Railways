import React, { useState, useEffect, useRef } from 'react';
import './TrainTrafficControl.css';

// Your TRACK_SECTIONS and CONNECTIONS constants are unchanged
const TRACK_SECTIONS = [
  { id: 'STN_A',   x: 100, y: 200, width: 60, height: 8,  type: 'station', station: 'A', platforms: 3, name: 'STA A' },
  { id: 'BLOCK_A1',x: 170, y: 200, width: 60, height: 8,  type: 'block',   name: 'Block A1' },
  { id: 'BLOCK_A2',x: 240, y: 200, width: 60, height: 8,  type: 'block',   name: 'Block A2' },
  { id: 'STN_B',   x: 310, y: 200, width: 60, height: 8,  type: 'station', station: 'B', platforms: 2, name: 'STA B' },
  { id: 'BLOCK_B1',x: 380, y: 200, width: 60, height: 8,  type: 'block',   name: 'Block B1' },
  { id: 'BLOCK_B2',x: 450, y: 200, width: 60, height: 8,  type: 'block',   name: 'Block B2' },
  { id: 'STN_C',   x: 520, y: 200, width: 60, height: 8,  type: 'station', station: 'C', platforms: 2, name: 'STA C' },
  { id: 'STN_D',   x: 100, y:  80, width: 60, height: 8,  type: 'station', station: 'D', platforms: 2, name: 'STA D' },
  { id: 'BLOCK_D1',x: 170, y:  80, width: 60, height: 8,  type: 'block',   name: 'Block D1' },
  { id: 'BLOCK_D2',x: 240, y:  80, width: 60, height: 8,  type: 'block',   name: 'Block D2' },
  { id: 'STN_E',   x: 240, y:  20, width: 60, height: 8,  type: 'station', station: 'E', platforms: 2, name: 'STA E' },
  { id: 'BLOCK_D3',x: 310, y:  80, width: 60, height: 8,  type: 'block',   name: 'Block D3' },
  { id: 'BLOCK_D4',x: 380, y:  80, width: 60, height: 8,  type: 'block',   name: 'Block D4' },
  { id: 'BLOCK_D5', x: 410, y: 140, width: 60, height: 8, type: 'block', name: 'Block D5' },
  { id: 'BLOCK_V_D2_A2', x: 240, y: 140, width: 60, height: 8, type: 'block', name: 'Block (D2-A2)' },
  { id: 'BLOCK_F1', x: 170, y: 260, width: 60, height: 8, type: 'block',   name: 'Block F1' },
  { id: 'BLOCK_F2', x: 170, y: 320, width: 60, height: 8, type: 'block',   name: 'Block F2' },
  { id: 'STN_F',   x: 170, y: 380, width: 60, height: 8, type: 'station', station: 'F', platforms: 2, name: 'STA F' },
];
const CONNECTIONS = [
  { from: 'STN_A',    to: 'BLOCK_A1',  type: 'main', path: `M130,204 L200,204` }, { from: 'BLOCK_A1', to: 'BLOCK_A2',  type: 'main', path: `M200,204 L270,204` },
  { from: 'BLOCK_A2', to: 'STN_B',     type: 'main', path: `M270,204 L340,204` }, { from: 'STN_B',    to: 'BLOCK_B1',  type: 'main', path: `M340,204 L410,204` },
  { from: 'BLOCK_B1', to: 'BLOCK_B2',  type: 'main', path: `M410,204 L480,204` }, { from: 'BLOCK_B2', to: 'STN_C',     type: 'main', path: `M480,204 L550,204` },
  { from: 'STN_D',    to: 'BLOCK_D1',  type: 'branch', path: `M130,84 L200,84` }, { from: 'BLOCK_D1', to: 'BLOCK_D2',  type: 'branch', path: `M200,84 L270,84` },
  { from: 'STN_E',    to: 'BLOCK_D2',  type: 'junction', path: `M270,28 L270,84` }, { from: 'BLOCK_D2', to: 'BLOCK_D3',  type: 'branch', path: `M270,84 L340,84` },
  { from: 'BLOCK_D3', to: 'BLOCK_D4',  type: 'branch', path: `M340,84 L410,84` }, { from: 'BLOCK_D4', to: 'BLOCK_D5', type: 'branch', path: `M410,84 L440,144` },
  { from: 'BLOCK_D5', to: 'BLOCK_B1', type: 'junction', path: `M440,144 L410,204` }, { from: 'BLOCK_D2', to: 'BLOCK_V_D2_A2', type: 'junction', path: `M270,84 L270,144` },
  { from: 'BLOCK_V_D2_A2', to: 'BLOCK_A2', type: 'junction', path: `M270,144 L270,204` }, { from: 'BLOCK_A1', to: 'BLOCK_F1', type: 'branch', path: `M200,204 L200,260` },
  { from: 'BLOCK_F1', to: 'BLOCK_F2', type: 'branch', path: `M200,260 L200,320` }, { from: 'BLOCK_F2', to: 'STN_F',   type: 'branch', path: `M200,320 L200,380` },
];


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
  
  const [metrics, setMetrics] = useState({ throughput: 0, avgDelay: 0, utilization: 0, avgSpeed: 0 });
  // === NEW: State to manage the list of notifications ===
  const [notifications, setNotifications] = useState([]);

  const API_BASE_URL = 'http://localhost:8000';
  const WS_URL = 'ws://localhost:8000/ws';

  const menuItems = [ { id: 'live-monitoring', label: 'Live Monitoring', icon: 'standard', category: 'operations' }, { id: 'audit-trail', label: 'Audit Trail', icon: 'standard', category: 'operations' }, { id: 'train-precedence', label: 'Train Precedence', icon: 'optimization', category: 'optimization' }, { id: 'crossing-optimization', label: 'Crossing Optimization', icon: 'optimization', category: 'optimization' }, { id: 'route-planning', label: 'Route Planning', icon: 'optimization', category: 'optimization' }, { id: 'resource-utilization', label: 'Resource Utilization', icon: 'optimization', category: 'optimization' }, { id: 'conflict-resolution', label: 'Conflict Resolution', icon: 'ai', category: 'ai' }, { id: 'ai-recommendations', label: 'AI Recommendations', icon: 'ai', category: 'ai' }, { id: 'predictive-analysis', label: 'Predictive Analysis', icon: 'ai', category: 'ai' }, { id: 'disruption-management', label: 'Disruption Management', icon: 'ai', category: 'ai' }, { id: 'what-if-simulation', label: 'What-If Simulation', icon: 'analysis', category: 'analysis' }, { id: 'scenario-analysis', label: 'Scenario Analysis', icon: 'analysis', category: 'analysis' }, { id: 'performance-dashboard', label: 'Performance Dashboard', icon: 'analysis', category: 'analysis' }, { id: 'throughput-analysis', label: 'Throughput Analysis', icon: 'analysis', category: 'analysis' }, { id: 'delay-analytics', label: 'Delay Analytics', icon: 'analysis', category: 'analysis' }, ];

  // === FIXED: The warning about missing dependency is resolved here ===
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket(WS_URL);
        ws.onopen = () => { console.log('WebSocket connected'); setConnected(true); setError(null); setLoading(false); };
        ws.onmessage = (event) => {
          try { const data = JSON.parse(event.data); updateSystemState(data); }
          catch (err) { console.error('Error parsing WebSocket message:', err); }
        };
        ws.onclose = () => {
          console.log('WebSocket disconnected'); setConnected(false);
          reconnectTimeoutRef.current = setTimeout(() => { console.log('Attempting to reconnect...'); connectWebSocket(); }, 3000);
        };
        ws.onerror = (error) => { console.error('WebSocket error:', error); setError('Connection failed'); setLoading(false); };
        wsRef.current = ws;
      } catch (err) { console.error('Failed to create WebSocket connection:', err); setError('Failed to connect to backend'); setLoading(false); }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) { wsRef.current.close(); }
      if (reconnectTimeoutRef.current) { clearTimeout(reconnectTimeoutRef.current); }
    };
  }, []);

  // In TrainTrafficControl.jsx

  const updateSystemState = (data) => {
    setTrains(data.trains || []);
    setBlockOccupancy(data.blockOccupancy || {});
    setStationPlatforms(data.stationPlatforms || {});
    setSimulationTime(data.simulationTime || 0);
    setIsRunning(data.isRunning || false);
    setTrainProgress(data.trainProgress || {});
    setMetrics(data.metrics || { throughput: 0, avgDelay: 0, utilization: 0, avgSpeed: 0 });

    // === MODIFIED: Logic to handle a scrolling history of notifications ===
    if (data.events && data.events.length > 0) {
      const newNotifications = data.events.map(eventText => ({
        id: Date.now() + Math.random(), // Unique ID for each notification
        text: eventText,
      }));
      // Add new notifications to the top and keep a history of the last 20
      setNotifications(prev => [...newNotifications, ...prev].slice(0, 20)); 
    }
  };

  const controlSimulation = async (action) => {
    try {
      const response = await fetch(`${API_BASE_URL}/simulation-control`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action }),
      });
      if (!response.ok) { throw new Error(`HTTP error! status: ${response.status}`); }
      await response.json();
    } catch (err) { setError(`Failed to ${action} simulation`); }
  };

  useEffect(() => {
    const clock = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(clock);
  }, []);
  
  const getSectionState = (sectionId) => {
    const section = TRACK_SECTIONS.find(s => s.id === sectionId);
    if (!section) return 'free';
    if (section.type === 'block') { return blockOccupancy[sectionId] ? 'occupied' : 'free'; }
    if (section.type === 'station') {
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
  const handleTrainClick = (train, event) => { event.stopPropagation(); setSelectedTrain(selectedTrain?.id === train.id ? null : train); };
  const handleTrainHover = (train, event) => { event.stopPropagation(); setHoveredTrain(train); };
  const handleTrainLeave = () => setHoveredTrain(null);
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
  }

  if (loading) { return ( <div className="tms-container"><div className="loading-overlay"><div className="loading-spinner"></div></div></div> ); }

  return (
    <div className="tms-container" onMouseMove={handleMouseMove}>
      <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
        {connected ? '● BACKEND CONNECTED' : '● BACKEND DISCONNECTED'}
      </div>
      <div className="tms-header">
        <div className="header-left">
          <div className="system-title">INTELLIGENT RAILWAY CONTROL SYSTEM</div>
          <div className="system-subtitle">BLOCK SIGNALING & TRAFFIC MANAGEMENT</div>
        </div>
        <div className="header-center">
          <div className="status-group"><div className="status-display green">{freeBlocksCount()}</div><div className="status-label">FREE BLOCKS</div></div>
          <div className="status-group"><div className="status-display blue">{String(trains.filter(t => t.statusType === 'running').length).padStart(2, '0')}</div><div className="status-label">ACTIVE</div></div>
          <div className="status-group"><div className="status-display orange">{String(trains.filter(t => t.waitingForBlock).length).padStart(2, '0')}</div><div className="status-label">WAITING</div></div>
          <div className="status-group"><div className="status-display red">00</div><div className="status-label">ALERTS</div></div>
          <div className="time-display">{currentTime.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}</div>
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
          <svg className="track-svg" viewBox="0 0 900 500">
            {CONNECTIONS.map((conn, index) => <path key={index} d={conn.path} className="connection-line" />)}
            {TRACK_SECTIONS.map(section => {
              const state = getSectionState(section.id);
              const trainsInSection = getTrainsInSection(section.id);
              const isSelected = selectedTrain && trainsInSection.some(t => t.id === selectedTrain.id);
              return (
                <g key={section.id}>
                  <rect x={section.x} y={section.y} width={section.width} height={section.height}
                    className={`track-section ${section.type === 'station' ? 'track-station' : 'track-block'} ${state === 'occupied' ? 'track-occupied' : state === 'partial' ? 'track-partial' : 'track-free'} ${isSelected ? 'track-selected' : ''}`}
                    rx="4" />
                  <text x={section.x + section.width / 2} y={section.y - 8} className="section-id-label">{section.id}</text>
                  {section.type === 'station' && (
                    <>
                      <text x={section.x + section.width / 2} y={section.y + 25} className="station-name-label">{section.name}</text>
                      <text x={section.x + section.width / 2} y={section.y + 38} className="platform-count-label">{section.platforms}P</text>
                      <g className="platform-indicators">
                        {Object.entries(stationPlatforms[section.id] || {}).map(([platformNum, occupant], idx) => (
                          <g key={platformNum}><circle cx={section.x + 15 + (idx * 15)} cy={section.y + 50} r="5" className={`platform-indicator ${occupant ? 'occupied' : 'free'}`} /><text x={section.x + 15 + (idx * 15)} y={section.y + 54} className="platform-number">{platformNum}</text></g>
                        ))}
                      </g>
                    </>
                  )}
                  {trainsInSection.map((train, trainIndex) => {
                    const center = getSectionCenter(section);
                    let offsetY = 0, offsetX = 0;
                    if (section.type === 'station') { offsetY = (trainIndex * 18) - ((trainsInSection.length - 1) * 9); offsetX = (trainIndex * 10) - ((trainsInSection.length - 1) * 5); }
                    const isTrainSelected = selectedTrain?.id === train.id;
                    return (
                      <g key={train.id} className={`train-group ${isTrainSelected ? 'selected' : ''} ${train.waitingForBlock ? 'waiting' : ''}`} onClick={(e) => handleTrainClick(train, e)} onMouseEnter={(e) => handleTrainHover(train, e)} onMouseLeave={handleTrainLeave}>
                        <rect x={center.x - 20 + offsetX} y={center.y - 10 + offsetY} width={40} height={20} rx="10" className={`train-body train-${train.statusType} ${isTrainSelected ? 'train-selected' : ''} ${train.waitingForBlock ? 'train-waiting' : ''}`} />
                        <text x={center.x + offsetX} y={center.y + offsetY + 3} className="train-number-label">{train.number}</text>
                        {train.waitingForBlock && <circle cx={center.x + 25 + offsetX} cy={center.y - 5 + offsetY} r="4" className="waiting-indicator" />}
                      </g>
                    );
                  })}
                </g>
              );
            })}
            {/*<circle cx="200" cy="175" r="8" className={`signal ${blockOccupancy['STN_B'] ? 'signal-red' : 'signal-green'}`} />
            <circle cx="280" cy="225" r="8" className={`signal ${blockOccupancy['BLOCK_A1'] ? 'signal-red' : 'signal-green'}`} />
            <circle cx="520" cy="175" r="8" className={`signal ${blockOccupancy['STN_C'] ? 'signal-red' : 'signal-green'}`} />
            <circle cx="580" cy="225" r="8" className={`signal ${blockOccupancy['BLOCK_B1'] ? 'signal-red' : 'signal-green'}`} /> */}
          </svg>
        </div>
        <div className="simulation-controls">
          <div className="control-row">
            <button onClick={() => handleSimulationControl(isRunning ? 'pause' : 'start')} className={`sim-btn ${isRunning ? 'pause' : 'start'}`} disabled={!connected}>
              {isRunning ? '⏸ PAUSE' : '▶️ START'}
            </button>
            <button onClick={() => handleSimulationControl('reset')} className="sim-btn reset" disabled={!connected}>🔄 RESET</button>
          </div>
          <div className="sim-time">SIM TIME: {String(Math.floor(simulationTime / 60)).padStart(2, '0')}:{String(simulationTime % 60).padStart(2, '0')}</div>
          <div className="sim-stats">
            <span className="stat-running">RUN: {trains.filter(t => t.statusType === 'running').length}</span>
            <span className="stat-waiting">WAIT: {trains.filter(t => t.waitingForBlock).length}</span>
            <span className="stat-completed">DONE: {trains.filter(t => t.statusType === 'completed').length}</span>
          </div>
          <div className="performance-metrics">
            <div className="metric-item"><span className="metric-label">Throughput:</span><span className="metric-value">{metrics.throughput.toFixed(2)} t/hr</span></div>
            <div className="metric-item"><span className="metric-label">Avg Delay:</span><span className="metric-value">{metrics.avgDelay.toFixed(2)} ticks</span></div>
            <div className="metric-item"><span className="metric-label">Utilization:</span><span className="metric-value">{metrics.utilization.toFixed(1)}%</span></div>
        
          </div>
          {/* === NEW: Notification Panel === */}
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
          <div className="panel-header">BLOCK STATUS</div>
          <div className="block-status-grid">
            {Object.entries(blockOccupancy).slice(0, 8).map(([blockId, occupant]) => (
              <div key={blockId} className={`block-status-item ${occupant ? 'occupied' : 'free'}`}><div className="block-id">{blockId}</div><div className="block-occupant">{occupant || 'FREE'}</div></div>
            ))}
          </div>
        </div>
        <div className="panel-section">
          <div className="panel-header">STATION STATUS</div>
          {TRACK_SECTIONS.filter(s => s.type === 'station').map(station => {
            const platforms = stationPlatforms[station.id] || {};
            const occupiedCount = Object.values(platforms).filter(p => p !== null).length;
            return (
              <div key={station.id} className="station-status-item">
                <div className="station-header"><span className="station-name">{station.name} ({station.station})</span><span className="platform-count">Platforms: {station.platforms}</span></div>
                <div className="platform-status"><span className="occupancy-info">Occupied: {occupiedCount}/{station.platforms}</span><div className="platform-indicators-panel">{Object.entries(platforms).map(([platformNum, occupant]) => <div key={platformNum} className={`platform-dot ${occupant ? 'occupied' : 'free'}`}>P{platformNum}</div>)}</div></div>
              </div>
            );
          })}
        </div>
        <div className="panel-section">
          <div className="panel-header">OPERATIONS</div>
          {menuItems.filter(item => item.category === 'operations').map(item => (
            <div key={item.id} className={`menu-item ${activeMenuItem === item.id ? 'active' : ''}`} onClick={() => handleMenuItemClick(item.id)}><div className={`menu-icon ${item.icon}`}></div>{item.label}</div>
          ))}
        </div>
        <div className="panel-section">
          <div className="panel-header">ACTIVE TRAINS ({trains.length})</div>
          {trains.map(train => {
            const currentSection = TRACK_SECTIONS.find(s => s.id === train.section);
            const isSelected = selectedTrain?.id === train.id;
            const routeIndex = getRouteIndex(train.id);
            return (
              <div key={train.id} className={`train-item ${isSelected ? 'selected' : ''} ${train.waitingForBlock ? 'waiting' : ''}`} onClick={() => setSelectedTrain(isSelected ? null : train)}>
                <div className={`train-status-dot ${train.statusType} ${train.waitingForBlock ? 'waiting' : ''}`}></div>
                <div className="train-details">
                  <div className="train-name">{train.name}</div>
                  <div className="train-info">{train.number} | {currentSection?.name || train.section} → Terminal | {Math.round(train.speed)} km/h {train.delay > 0 && ` | +${train.delay}min`} {train.waitingForBlock && <span className="waiting-status"> | WAITING</span>}</div>
                  <div className="train-route-info">Progress: {routeIndex + 1}/{train.route?.length || 0} {trainProgress[train.id]?.waitingForSection && <span className="waiting-for">{' '}| Waiting for {TRACK_SECTIONS.find(s => s.id === trainProgress[train.id].waitingForSection)?.name || trainProgress[train.id].waitingForSection}</span>}</div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      {hoveredTrain && (
        <div className="train-tooltip" style={{ left: Math.min(mousePos.x + 20, window.innerWidth - 420), top: Math.max(mousePos.y - 250, 10) }}>
          <div className="tooltip-header">{hoveredTrain.name}</div>
          <div className="tooltip-content">
            <div className="tooltip-section">
              <div className="tooltip-row"><span className="tooltip-label">Train Number:</span><span className="tooltip-value">{hoveredTrain.number}</span></div>
              <div className="tooltip-row"><span className="tooltip-label">Current Speed:</span><span className="tooltip-value tooltip-speed">{Math.round(hoveredTrain.speed)} km/h</span></div>
              <div className="tooltip-row"><span className="tooltip-label">Current Location:</span><span className="tooltip-value tooltip-section-id">{TRACK_SECTIONS.find(s => s.id === hoveredTrain.section)?.name || hoveredTrain.section}</span></div>
              <div className="tooltip-row"><span className="tooltip-label">Status:</span><span className={`tooltip-value tooltip-status ${hoveredTrain.statusType}`}>{hoveredTrain.status}</span></div>
              <div className="tooltip-row"><span className="tooltip-label">Block Status:</span><span className={`tooltip-value ${hoveredTrain.waitingForBlock ? 'waiting' : 'clear'}`}>{hoveredTrain.waitingForBlock ? 'WAITING FOR BLOCK' : 'CLEAR TO PROCEED'}</span></div>
              {/* === FIXED: This line was causing the critical error === */}
              <div className="tooltip-row"><span className="tooltip-label">Route Progress:</span><span className="tooltip-value">{getRouteIndex(hoveredTrain.id) + 1} of {hoveredTrain.route?.length || 0}</span></div>
              <div className="tooltip-row"><span className="tooltip-label">Backend Status:</span><span className={`tooltip-value ${connected ? 'clear' : 'waiting'}`}>{connected ? 'CONNECTED' : 'DISCONNECTED'}</span></div>
            </div>
          </div>
        </div>
      )}
      {error && (
        <div style={{ position: 'fixed', top: '100px', right: '20px', background: 'rgba(255, 100, 100, 0.9)', color: 'white', padding: '10px 15px', borderRadius: '4px', zIndex: 1000, fontSize: '12px' }}>
          {error}
          <button onClick={() => setError(null)} style={{ background: 'transparent', border: 'none', color: 'white', marginLeft: '10px', cursor: 'pointer' }}>×</button>
        </div>
      )}
    </div>
  );
};

export default TrainTrafficControl;