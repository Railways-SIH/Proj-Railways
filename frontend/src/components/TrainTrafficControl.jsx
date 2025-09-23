import React, { useState, useEffect, useRef } from 'react';
import './TrainTrafficControl.css';
// Enhanced Track sections with realistic positioning
const TRACK_SECTIONS = [
  // Main horizontal line (A -> B -> C)
  { id: 'STN_A',   x: 100, y: 200, width: 60, height: 8,  type: 'station', station: 'A', platforms: 3, name: 'STA A' },
  { id: 'BLOCK_A1',x: 170, y: 200, width: 60, height: 8,  type: 'block',   name: 'Block A1' },
  { id: 'BLOCK_A2',x: 240, y: 200, width: 60, height: 8,  type: 'block',   name: 'Block A2' },
  { id: 'STN_B',   x: 310, y: 200, width: 60, height: 8,  type: 'station', station: 'B', platforms: 2, name: 'STA B' },
  { id: 'BLOCK_B1',x: 380, y: 200, width: 60, height: 8,  type: 'block',   name: 'Block B1' },
  { id: 'BLOCK_B2',x: 450, y: 200, width: 60, height: 8,  type: 'block',   name: 'Block B2' },
  { id: 'STN_C',   x: 520, y: 200, width: 60, height: 8,  type: 'station', station: 'C', platforms: 2, name: 'STA C' },

  // Upper branch (D --> E --> junction)
  { id: 'STN_D',   x: 100, y:  80, width: 60, height: 8,  type: 'station', station: 'D', platforms: 2, name: 'STA D' },
  { id: 'BLOCK_D1',x: 170, y:  80, width: 60, height: 8,  type: 'block',   name: 'Block D1' },
  { id: 'BLOCK_D2',x: 240, y:  80, width: 60, height: 8,  type: 'block',   name: 'Block D2' },
  { id: 'STN_E',   x: 240, y:  20, width: 60, height: 8,  type: 'station', station: 'E', platforms: 2, name: 'STA E' },
  { id: 'BLOCK_D3',x: 310, y:  80, width: 60, height: 8,  type: 'block',   name: 'Block D3' },
  { id: 'BLOCK_D4',x: 380, y:  80, width: 60, height: 8,  type: 'block',   name: 'Block D4' },
  { id: 'BLOCK_D5', x: 410, y: 140, width: 60, height: 8, type: 'block', name: 'Block D5' },
  { id: 'BLOCK_V_D2_A2', x: 240, y: 140, width: 60, height: 8, type: 'block', name: 'Block (D2-A2)' },

  // Lower branch (A1 -> F)
  { id: 'BLOCK_F1', x: 170, y: 260, width: 60, height: 8, type: 'block',   name: 'Block F1' },
  { id: 'BLOCK_F2', x: 170, y: 320, width: 60, height: 8, type: 'block',   name: 'Block F2' },
  { id: 'STN_F',   x: 170, y: 380, width: 60, height: 8, type: 'station', station: 'F', platforms: 2, name: 'STA F' },
];

const CONNECTIONS = [
  // Main line
  { from: 'STN_A',    to: 'BLOCK_A1',  type: 'main', path: `M130,204 L200,204` },
  { from: 'BLOCK_A1', to: 'BLOCK_A2',  type: 'main', path: `M200,204 L270,204` },
  { from: 'BLOCK_A2', to: 'STN_B',     type: 'main', path: `M270,204 L340,204` },
  { from: 'STN_B',    to: 'BLOCK_B1',  type: 'main', path: `M340,204 L410,204` },
  { from: 'BLOCK_B1', to: 'BLOCK_B2',  type: 'main', path: `M410,204 L480,204` },
  { from: 'BLOCK_B2', to: 'STN_C',     type: 'main', path: `M480,204 L550,204` },

  // Upper branch
  { from: 'STN_D',    to: 'BLOCK_D1',  type: 'branch', path: `M130,84 L200,84` },
  { from: 'BLOCK_D1', to: 'BLOCK_D2',  type: 'branch', path: `M200,84 L270,84` },
  { from: 'STN_E',    to: 'BLOCK_D2',  type: 'junction', path: `M270,28 L270,84` },
  { from: 'BLOCK_D2', to: 'BLOCK_D3',  type: 'branch', path: `M270,84 L340,84` },
  { from: 'BLOCK_D3', to: 'BLOCK_D4',  type: 'branch', path: `M340,84 L410,84` },
  { from: 'BLOCK_D4', to: 'BLOCK_D5', type: 'branch', path: `M410,84 L440,144` },
  { from: 'BLOCK_D5', to: 'BLOCK_B1', type: 'junction', path: `M440,144 L410,204` },
  { from: 'BLOCK_D2',      to: 'BLOCK_V_D2_A2', type: 'junction', path: `M270,84 L270,144` },
  { from: 'BLOCK_V_D2_A2', to: 'BLOCK_A2',      type: 'junction', path: `M270,144 L270,204` },

  // Lower branch
  { from: 'BLOCK_A1', to: 'BLOCK_F1', type: 'branch', path: `M200,204 L200,260` },
  { from: 'BLOCK_F1', to: 'BLOCK_F2', type: 'branch', path: `M200,260 L200,320` },
  { from: 'BLOCK_F2', to: 'STN_F',   type: 'branch', path: `M200,320 L200,380` },
];

const OptimizedTrainTrafficControl = () => {
  // State management
  const [trains, setTrains] = useState([]);
  const [blockOccupancy, setBlockOccupancy] = useState({});
  const [stationPlatforms, setStationPlatforms] = useState({});
  const [simulationTime, setSimulationTime] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [trainProgress, setTrainProgress] = useState({});
  const [scheduledDepartures, setScheduledDepartures] = useState({});
  const [hoveredTrain, setHoveredTrain] = useState(null);
  const [selectedTrain, setSelectedTrain] = useState(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [currentTime, setCurrentTime] = useState(new Date());
  const [activeMenuItem, setActiveMenuItem] = useState('live-monitoring');
  const [showOptimizationPanel, setShowOptimizationPanel] = useState(false);
  
  // Backend connection state
  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [optimizationStats, setOptimizationStats] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  
  const API_BASE_URL = 'http://localhost:8000';
  const WS_URL = 'ws://localhost:8000/ws';

  // WebSocket connection management
  useEffect(() => {
    connectWebSocket();
    fetchOptimizationStats();
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
    };
  }, []);

  const connectWebSocket = () => {
    try {
      const ws = new WebSocket(WS_URL);
      
      ws.onopen = () => {
        console.log('WebSocket connected to optimized backend');
        setConnected(true);
        setError(null);
        setLoading(false);
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          updateSystemState(data);
        } catch (err) {
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
        setError('Optimized backend connection failed');
        setLoading(false);
      };
      
      wsRef.current = ws;
    } catch (err) {
      console.error('Failed to create WebSocket connection:', err);
      setError('Failed to connect to optimized backend');
      setLoading(false);
    }
  };

  const updateSystemState = (data) => {
    setTrains(data.trains || []);
    setBlockOccupancy(data.blockOccupancy || {});
    setStationPlatforms(data.stationPlatforms || {});
    setSimulationTime(data.simulationTime || 0);
    setIsRunning(data.isRunning || false);
    setTrainProgress(data.trainProgress || {});
    setScheduledDepartures(data.scheduledDepartures || {});
  };

  const fetchOptimizationStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/optimization-stats`);
      if (response.ok) {
        const stats = await response.json();
        setOptimizationStats(stats);
      }
    } catch (err) {
      console.error('Error fetching optimization stats:', err);
    }
  };

  // Fetch optimization stats periodically
  useEffect(() => {
    const interval = setInterval(fetchOptimizationStats, 5000);
    return () => clearInterval(interval);
  }, []);

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
      
      const result = await response.json();
      console.log(`Optimized simulation ${action}:`, result);
    } catch (err) {
      console.error(`Error ${action} simulation:`, err);
      setError(`Failed to ${action} optimized simulation`);
    }
  };

  // Clock update
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
    } else if (section.type === 'station') {
      const platforms = stationPlatforms[sectionId] || {};
      const occupiedPlatforms = Object.values(platforms).filter(occupant => occupant !== null).length;
      if (occupiedPlatforms === 0) return 'free';
      if (occupiedPlatforms < (section.platforms || 1)) return 'partial';
      return 'occupied';
    }
    return 'free';
  };

  const getTrainsInSection = (sectionId) => {
    return trains.filter(train => train.section === sectionId);
  };

  const getSectionCenter = (section) => ({
    x: section.x + section.width / 2,
    y: section.y + section.height / 2
  });

  const getRouteIndex = (trainId) => {
    const progress = trainProgress[trainId];
    return progress?.currentRouteIndex || 0;
  };

  const getTrainTypeColor = (trainType) => {
    const colors = {
      'Express': '#ff6b6b',    // Red for express
      'Passenger': '#4ecdc4',  // Teal for passenger
      'Freight': '#45b7d1'     // Blue for freight
    };
    return colors[trainType] || '#95a5a6';
  };

  const handleMouseMove = (e) => {
    setMousePos({ x: e.clientX, y: e.clientY });
  };

  const handleTrainClick = (train, event) => {
    event.stopPropagation();
    setSelectedTrain(selectedTrain?.id === train.id ? null : train);
  };

  const handleTrainHover = (train, event) => {
    event.stopPropagation();
    setHoveredTrain(train);
  };

  const handleTrainLeave = () => {
    setHoveredTrain(null);
  };

  if (loading) {
    return (
      <div className="tms-container">
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <div className="loading-text">Initializing Optimized Railway Control System...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="tms-container" onMouseMove={handleMouseMove}>
      {/* Enhanced Connection Status */}
      <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
        {connected ? '● OPTIMIZED BACKEND CONNECTED' : '● OPTIMIZED BACKEND DISCONNECTED'}
        {optimizationStats && (
          <span className="optimization-indicator">
            | THROUGHPUT: {optimizationStats.trains?.throughput_efficiency || 0}%
          </span>
        )}
      </div>

      {/* Enhanced Header with Optimization Stats */}
      <div className="tms-header">
        <div className="header-left">
          <div className="system-title">INTELLIGENT RAILWAY CONTROL SYSTEM v3.0</div>
          <div className="system-subtitle">OPTIMIZED BLOCK SIGNALING & TRAFFIC MANAGEMENT</div>
        </div>
        
        <div className="header-center">
          <div className="status-group">
            <div className="status-display green">
              {Object.values(blockOccupancy).filter(occupant => occupant === null).length}
            </div>
            <div className="status-label">FREE BLOCKS</div>
          </div>
          
          <div className="status-group">
            <div className="status-display blue">
              {String(trains.filter(t => t.statusType === 'running').length).padStart(2, '0')}
            </div>
            <div className="status-label">RUNNING</div>
          </div>
          
          <div className="status-group">
            <div className="status-display orange">
              {String(trains.filter(t => t.waitingForBlock).length).padStart(2, '0')}
            </div>
            <div className="status-label">WAITING</div>
          </div>

          <div className="status-group">
            <div className="status-display green">
              {String(trains.filter(t => t.statusType === 'completed').length).padStart(2, '0')}
            </div>
            <div className="status-label">ARRIVED</div>
          </div>
          
          <div className="time-display">
            {currentTime.toLocaleTimeString('en-US', { 
              hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
            })}
          </div>
        </div>

        <div className="header-right">
          <div className="control-buttons">
            <button 
              className={`control-btn ${showOptimizationPanel ? 'active' : ''}`}
              onClick={() => setShowOptimizationPanel(!showOptimizationPanel)}
            >
              OPTIMIZATION
            </button>
            <button className="control-btn">ROUTES</button>
            <button className="control-btn">SIGNALS</button>
            <button className="control-btn">ALERTS</button>
          </div>
          <div className="compass">N</div>
        </div>
      </div>

      {/* Main Display */}
      <div className="main-display">
        <div className="track-container">
          <svg className="track-svg" viewBox="0 0 900 500">
            {/* Draw connection lines */}
            {CONNECTIONS.map((conn, index) => (
              <path
                key={index}
                d={conn.path}
                className={`connection-line ${conn.type}`}
              />
            ))}

            {/* Draw track sections */}
            {TRACK_SECTIONS.map(section => {
              const state = getSectionState(section.id);
              const trainsInSection = getTrainsInSection(section.id);
              const isSelected = selectedTrain && trainsInSection.some(t => t.id === selectedTrain.id);
              
              return (
                <g key={section.id}>
                  <rect
                    x={section.x}
                    y={section.y}
                    width={section.width}
                    height={section.height}
                    className={`track-section ${
                      section.type === 'station' ? 'track-station' : 'track-block'
                    } ${
                      state === 'occupied' ? 'track-occupied' : 
                      state === 'partial' ? 'track-partial' : 'track-free'
                    } ${isSelected ? 'track-selected' : ''}`}
                    rx="4"
                  />
                  
                  {/* Section ID label */}
                  <text
                    x={section.x + section.width / 2}
                    y={section.y - 8}
                    className="section-id-label"
                  >
                    {section.id}
                  </text>
                  
                  {/* Station-specific labels and platform indicators */}
                  {section.type === 'station' && (
                    <>
                      <text
                        x={section.x + section.width / 2}
                        y={section.y + 25}
                        className="station-name-label"
                      >
                        {section.name}
                      </text>
                      
                      {/* Enhanced platform status indicators */}
                      <g className="platform-indicators">
                        {Object.entries(stationPlatforms[section.id] || {}).map(([platformNum, occupant], idx) => (
                          <g key={platformNum}>
                            <circle
                              cx={section.x + 15 + (idx * 15)}
                              cy={section.y + 40}
                              r="6"
                              className={`platform-indicator ${occupant ? 'occupied' : 'free'}`}
                            />
                            <text
                              x={section.x + 15 + (idx * 15)}
                              y={section.y + 44}
                              className="platform-number"
                            >
                              P{platformNum}
                            </text>
                          </g>
                        ))}
                      </g>
                    </>
                  )}
                  
                  {/* Enhanced train rendering with type-based colors */}
                  {trainsInSection.map((train, trainIndex) => {
                    const center = getSectionCenter(section);
                    let offsetY = 0;
                    let offsetX = 0;
                    
                    if (section.type === 'station') {
                      offsetY = (trainIndex * 18) - ((trainsInSection.length - 1) * 9);
                      offsetX = (trainIndex * 10) - ((trainsInSection.length - 1) * 5);
                    }
                    
                    const isTrainSelected = selectedTrain?.id === train.id;
                    
                    return (
                      <g 
                        key={train.id}
                        className={`train-group ${isTrainSelected ? 'selected' : ''} ${train.waitingForBlock ? 'waiting' : ''}`}
                        onClick={(e) => handleTrainClick(train, e)}
                        onMouseEnter={(e) => handleTrainHover(train, e)}
                        onMouseLeave={handleTrainLeave}
                      >
                        {/* Enhanced train body with type-based styling */}
                        <rect
                          x={center.x - 22 + offsetX}
                          y={center.y - 10 + offsetY}
                          width={44}
                          height={20}
                          rx="10"
                          className={`train-body train-${train.statusType} train-type-${train.type?.toLowerCase()}`}
                          style={{
                            fill: isTrainSelected ? '#fff' : getTrainTypeColor(train.type),
                            stroke: isTrainSelected ? getTrainTypeColor(train.type) : 'none',
                            strokeWidth: isTrainSelected ? '2px' : '0'
                          }}
                        />
                        
                        {/* Train number */}
                        <text
                          x={center.x + offsetX}
                          y={center.y + offsetY + 3}
                          className="train-number-label"
                          style={{ fill: isTrainSelected ? getTrainTypeColor(train.type) : '#fff' }}
                        >
                          {train.number}
                        </text>
                        
                        {/* Enhanced status indicators */}
                        {train.waitingForBlock && (
                          <circle
                            cx={center.x + 28 + offsetX}
                            cy={center.y - 8 + offsetY}
                            r="4"
                            className="waiting-indicator"
                          />
                        )}
                        
                        {/* Route optimization indicator */}
                        {train.type === 'Express' && (
                          <polygon
                            points={`${center.x - 28 + offsetX},${center.y - 8 + offsetY} ${center.x - 20 + offsetX},${center.y - 12 + offsetY} ${center.x - 20 + offsetX},${center.y - 4 + offsetY}`}
                            className="express-indicator"
                          />
                        )}
                      </g>
                    );
                  })}
                </g>
              );
            })}

            {/* Enhanced signal system */}
            <g className="signal-system">
              <circle cx="200" cy="175" r="8" className={`signal ${blockOccupancy['BLOCK_A1'] ? 'signal-red' : 'signal-green'}`} />
              <circle cx="340" cy="175" r="8" className={`signal ${blockOccupancy['STN_B'] ? 'signal-red' : 'signal-green'}`} />
              <circle cx="480" cy="175" r="8" className={`signal ${blockOccupancy['BLOCK_B2'] ? 'signal-red' : 'signal-green'}`} />
              <circle cx="270" cy="60" r="8" className={`signal ${blockOccupancy['BLOCK_D2'] ? 'signal-red' : 'signal-green'}`} />
              <circle cx="270" cy="120" r="8" className={`signal ${blockOccupancy['BLOCK_V_D2_A2'] ? 'signal-red' : 'signal-green'}`} />
            </g>
          </svg>
        </div>
        
        {/* Enhanced simulation controls */}
        <div className="simulation-controls">
          <div className="control-row">
            <button
              onClick={() => controlSimulation(isRunning ? 'pause' : 'start')}
              className={`sim-btn ${isRunning ? 'pause' : 'start'}`}
              disabled={!connected}
            >
              {isRunning ? '⏸ PAUSE' : '▶ START'}
            </button>
            <button
              onClick={() => controlSimulation('reset')}
              className="sim-btn reset"
              disabled={!connected}
            >
              🔄 RESET
            </button>
          </div>
          <div className="sim-time">
            SIM TIME: {String(Math.floor(simulationTime / 60)).padStart(2, '0')}:{String(simulationTime % 60).padStart(2, '0')}
          </div>
          <div className="sim-stats">
            <span className="stat-running">RUN: {trains.filter(t => t.statusType === 'running').length}</span>
            <span className="stat-waiting">WAIT: {trains.filter(t => t.waitingForBlock).length}</span>
            <span className="stat-completed">DONE: {trains.filter(t => t.statusType === 'completed').length}</span>
            {optimizationStats && (
              <span className="stat-throughput">
                EFFICIENCY: {optimizationStats.trains?.throughput_efficiency || 0}%
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Enhanced control panel */}
      <div className="control-panel">
        {/* Optimization Panel */}
        {showOptimizationPanel && optimizationStats && (
          <div className="panel-section optimization-panel">
            <div className="panel-header">OPTIMIZATION STATUS</div>
            <div className="optimization-stats">
              <div className="stat-row">
                <span className="stat-label">Throughput Efficiency:</span>
                <span className="stat-value">{optimizationStats.trains?.throughput_efficiency || 0}%</span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Block Utilization:</span>
                <span className="stat-value">{optimizationStats.infrastructure?.blocks?.utilization || 0}%</span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Platform Utilization:</span>
                <span className="stat-value">{optimizationStats.infrastructure?.platforms?.utilization || 0}%</span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Route Optimization:</span>
                <span className="stat-value active">ACTIVE</span>
              </div>
            </div>
          </div>
        )}

        {/* Train fleet with enhanced details */}
        <div className="panel-section">
          <div className="panel-header">TRAIN FLEET ({trains.length}) - ALL TO STATION C</div>
          
          {trains.map(train => {
            const currentSection = TRACK_SECTIONS.find(s => s.id === train.section);
            const isSelected = selectedTrain?.id === train.id;
            const routeIndex = getRouteIndex(train.id);
            const progress = trainProgress[train.id];
            
            return (
              <div 
                key={train.id} 
                className={`train-item enhanced ${isSelected ? 'selected' : ''} ${train.waitingForBlock ? 'waiting' : ''}`}
                onClick={() => setSelectedTrain(isSelected ? null : train)}
              >
                <div 
                  className={`train-icon type-${train.type?.toLowerCase()}`}
                  style={{ backgroundColor: getTrainTypeColor(train.type) }}
                >
                  {train.number}
                </div>
                
                <div className="train-details">
                  <div className="train-main">
                    <span className="train-number">{train.number}</span>
                    <span className="train-type">{train.type}</span>
                  </div>
                  <div className="train-status">
                    <span className={`status-badge ${train.statusType}`}>
                      {train.statusType.toUpperCase()}
                    </span>
                    {train.waitingForBlock && (
                      <span className="status-badge waiting">WAITING</span>
                    )}
                  </div>
                  <div className="train-location">
                    {currentSection ? currentSection.name : 'Unknown'}
                  </div>
                  {progress && (
                    <div className="train-progress">
                      Route Index: {routeIndex} | Progress: {progress.progress || 0}%
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default OptimizedTrainTrafficControl;