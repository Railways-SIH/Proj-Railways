import React, { useState, useEffect } from 'react';
import './TrainTrafficControl.css';

// Properly connected track network
const TRACK_SECTIONS = [
  // Main horizontal line (left to right)
  { id: '1R', x: 80, y: 200, width: 60, height: 8, type: 'horizontal', connections: ['2R'] },
  { id: '2R', x: 160, y: 200, width: 60, height: 8, type: 'horizontal', connections: ['1R', '3L'] },
  { id: '3L', x: 240, y: 200, width: 60, height: 8, type: 'horizontal', connections: ['2R', '4L'] },
  { id: '4L', x: 320, y: 200, width: 60, height: 8, type: 'horizontal', connections: ['3L', '5L'] },
  { id: '5L', x: 400, y: 200, width: 60, height: 8, type: 'horizontal', connections: ['4L', '6L'] },
  { id: '6L', x: 480, y: 200, width: 60, height: 8, type: 'horizontal', connections: ['5L', '7L'] },
  { id: '7L', x: 560, y: 200, width: 60, height: 8, type: 'horizontal', connections: ['6L', '8L'] },
  { id: '8L', x: 640, y: 200, width: 60, height: 8, type: 'horizontal', connections: ['7L', '9L'] },
  { id: '9L', x: 720, y: 200, width: 60, height: 8, type: 'horizontal', connections: ['8L'] },
  
  // Upper branch line
  { id: '101L', x: 240, y: 120, width: 80, height: 8, type: 'horizontal', connections: ['102L'] },
  { id: '102L', x: 340, y: 120, width: 80, height: 8, type: 'horizontal', connections: ['101L', '103L'] },
  { id: '103L', x: 440, y: 120, width: 80, height: 8, type: 'horizontal', connections: ['102L', '104L'] },
  { id: '104L', x: 540, y: 120, width: 80, height: 8, type: 'horizontal', connections: ['103L'] },
  
  // Lower branch line
  { id: '201L', x: 240, y: 280, width: 80, height: 8, type: 'horizontal', connections: ['202L'] },
  { id: '202L', x: 340, y: 280, width: 80, height: 8, type: 'horizontal', connections: ['201L', '203L'] },
  { id: '203L', x: 440, y: 280, width: 80, height: 8, type: 'horizontal', connections: ['202L', '204L'] },
  { id: '204L', x: 540, y: 280, width: 80, height: 8, type: 'horizontal', connections: ['203L'] },
  
  // Yard tracks
  { id: '301Y', x: 80, y: 350, width: 100, height: 8, type: 'horizontal', connections: ['302Y'] },
  { id: '302Y', x: 200, y: 350, width: 100, height: 8, type: 'horizontal', connections: ['301Y', '303Y'] },
  { id: '303Y', x: 320, y: 350, width: 100, height: 8, type: 'horizontal', connections: ['302Y', '304Y'] },
  { id: '304Y', x: 440, y: 350, width: 100, height: 8, type: 'horizontal', connections: ['303Y'] },
];

// Connection paths between sections
const CONNECTIONS = [
  // Main line connections (horizontal)
  { from: '1R', to: '2R', type: 'main', path: 'M140,204 L160,204' },
  { from: '2R', to: '3L', type: 'main', path: 'M220,204 L240,204' },
  { from: '3L', to: '4L', type: 'main', path: 'M300,204 L320,204' },
  { from: '4L', to: '5L', type: 'main', path: 'M380,204 L400,204' },
  { from: '5L', to: '6L', type: 'main', path: 'M460,204 L480,204' },
  { from: '6L', to: '7L', type: 'main', path: 'M540,204 L560,204' },
  { from: '7L', to: '8L', type: 'main', path: 'M620,204 L640,204' },
  { from: '8L', to: '9L', type: 'main', path: 'M700,204 L720,204' },
  
  // Upper branch connections (horizontal)
  { from: '101L', to: '102L', type: 'branch', path: 'M320,124 L340,124' },
  { from: '102L', to: '103L', type: 'branch', path: 'M420,124 L440,124' },
  { from: '103L', to: '104L', type: 'branch', path: 'M520,124 L540,124' },
  
  // Lower branch connections (horizontal)
  { from: '201L', to: '202L', type: 'branch', path: 'M320,284 L340,284' },
  { from: '202L', to: '203L', type: 'branch', path: 'M420,284 L440,284' },
  { from: '203L', to: '204L', type: 'branch', path: 'M520,284 L540,284' },
  
  // Yard connections (horizontal)
  { from: '301Y', to: '302Y', type: 'yard', path: 'M180,354 L200,354' },
  { from: '302Y', to: '303Y', type: 'yard', path: 'M300,354 L320,354' },
  { from: '303Y', to: '304Y', type: 'yard', path: 'M420,354 L440,354' },
  
  // Junction connections (vertical/diagonal)
  { from: '2R', to: '101L', type: 'junction', path: 'M190,200 L190,160 L240,160 L240,132' },
  { from: '104L', to: '6L', type: 'junction', path: 'M580,132 L580,160 L510,160 L510,200' },
  { from: '3L', to: '201L', type: 'junction', path: 'M270,208 L270,240 L280,240 L280,272' },
  { from: '204L', to: '7L', type: 'junction', path: 'M580,288 L580,240 L590,240 L590,208' },
];

const TRAINS = [
  {
    id: 'T1',
    name: 'Rajdhani Express',
    number: '12301',
    section: '2R',
    speed: 120,
    destination: 'New Delhi Junction',
    status: 'Running',
    delay: 0,
    route: ['2R', '3L', '4L', '5L', '6L', '7L', '8L', '9L', '8L', '7L', '6L', '5L', '4L', '3L'],
    statusType: 'running'
  },
  {
    id: 'T2', 
    name: 'Shatabdi Express',
    number: '12002',
    section: '101L',
    speed: 110,
    destination: 'Mumbai Central',
    status: 'Running',
    delay: 3,
    route: ['101L', '102L', '103L', '104L', '103L', '102L'],
    statusType: 'delayed'
  },
  {
    id: 'T3',
    name: 'Duronto Express', 
    number: '12259',
    section: '301Y',
    speed: 40,                 // give it a low default speed so it moves visibly
    destination: 'Kolkata Howrah',
    status: 'Running',         // <-- changed from 'Stopped'
    delay: 0,
    route: ['301Y', '302Y', '303Y', '304Y', '303Y', '302Y'],
    statusType: 'running'     // <-- changed from 'stopped'
  }
];

const TrainTrafficControl = () => {
  const [trains, setTrains] = useState(TRAINS);
  const [hoveredTrain, setHoveredTrain] = useState(null);
  const [selectedTrain, setSelectedTrain] = useState(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [currentTime, setCurrentTime] = useState(new Date());
  const [routeIndex, setRouteIndex] = useState({});
  const [activeMenuItem, setActiveMenuItem] = useState('live-monitoring');

  // Railway optimization menu items based on your project description
  const menuItems = [
    { id: 'live-monitoring', label: 'Live Monitoring', icon: 'standard', category: 'operations' },
    { id: 'train-precedence', label: 'Train Precedence', icon: 'optimization', category: 'optimization' },
    { id: 'crossing-optimization', label: 'Crossing Optimization', icon: 'optimization', category: 'optimization' },
    { id: 'route-planning', label: 'Route Planning', icon: 'optimization', category: 'optimization' },
    { id: 'conflict-resolution', label: 'Conflict Resolution', icon: 'ai', category: 'ai' },
    { id: 'ai-recommendations', label: 'AI Recommendations', icon: 'ai', category: 'ai' },
    { id: 'predictive-analysis', label: 'Predictive Analysis', icon: 'ai', category: 'ai' },
    { id: 'what-if-simulation', label: 'What-If Simulation', icon: 'analysis', category: 'analysis' },
    { id: 'scenario-analysis', label: 'Scenario Analysis', icon: 'analysis', category: 'analysis' },
    { id: 'performance-dashboard', label: 'Performance Dashboard', icon: 'analysis', category: 'analysis' },
    { id: 'throughput-analysis', label: 'Throughput Analysis', icon: 'analysis', category: 'analysis' },
    { id: 'delay-analytics', label: 'Delay Analytics', icon: 'analysis', category: 'analysis' },
    { id: 'resource-utilization', label: 'Resource Utilization', icon: 'optimization', category: 'optimization' },
    { id: 'disruption-management', label: 'Disruption Management', icon: 'ai', category: 'ai' },
    { id: 'audit-trail', label: 'Audit Trail', icon: 'standard', category: 'operations' },
  ];

  // Visualization buttons state (no logic, just UI)
  const [activeButtons, setActiveButtons] = useState({
    overview: true,
    signals: false,
    speed: false,
    alerts: false
  });

 // Initialize route indices
  useEffect(() => {
    const initialIndices = {};
    trains.forEach(train => {
      initialIndices[train.id] = 0;
    });
    setRouteIndex(initialIndices);
  }, []);

  // Clock
  useEffect(() => {
    const clock = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(clock);
  }, []);

  // 🚆 Train movement (only one interval now)
  useEffect(() => {
    const moveInterval = setInterval(() => {
      setTrains(prevTrains =>
        prevTrains.map(train => {
          if (train.statusType === 'stopped') return train;

          const route = train.route;
          if (!route || route.length === 0) return train;

          let index = routeIndex[train.id] ?? 0;
          let nextIndex = (index + 1) % route.length;

          setRouteIndex(prev => ({ ...prev, [train.id]: nextIndex }));

          // Add some random speed variation
          const delta = (Math.random() - 0.5) * 10;
          let newSpeed = Math.max(40, Math.min(130, train.speed + delta));

          return { ...train, section: route[nextIndex], speed: Math.round(newSpeed) };
        })
      );
    }, 2000);

    return () => clearInterval(moveInterval);
  }, [routeIndex]);

  // Simulate realistic train movement

  const getSectionState = (sectionId) => {
    const hasTrainInSection = trains.some(train => train.section === sectionId);
    return hasTrainInSection ? 'occupied' : 'free';
  };

  const getTrainInSection = (sectionId) => {
    return trains.find(train => train.section === sectionId);
  };

  const getSectionCenter = (section) => {
    return {
      x: section.x + section.width / 2,
      y: section.y + section.height / 2
    };
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

  const handleButtonClick = (buttonName) => {
    setActiveButtons(prev => ({
      ...prev,
      [buttonName]: !prev[buttonName]
    }));
  };

  const handleMenuItemClick = (itemId) => {
    setActiveMenuItem(itemId);
    console.log(`Selected: ${itemId}`); // For now, just log the selection
  };

  return (
    <div className="tms-container" onMouseMove={handleMouseMove}>
      {/* Enhanced Header */}
      <div className="tms-header">
        <div className="header-left">
          <div className="system-title">Train Traffic Control</div>
          <div className="system-subtitle">Intelligent Decision Support System v3.0</div>
        </div>
        
        <div className="header-center">
          <div className="status-group">
            <div className="status-display green">98</div>
            <div className="status-label">Sections</div>
          </div>
          
          <div className="status-group">
            <div className="status-display yellow">03</div>
            <div className="status-label">Active</div>
          </div>
          
          <div className="status-group">
            <div className="status-display">00</div>
            <div className="status-label">Alerts</div>
          </div>
          
          <div className="time-display">
            {currentTime.toLocaleTimeString('en-US', { 
              hour12: false,
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit'
            })}
          </div>
        </div>

        <div className="header-right">
          <div className="control-buttons">
            <button 
              className={`control-btn ${activeButtons.overview ? 'active' : ''}`}
              onClick={() => handleButtonClick('overview')}
            >
              Overview
            </button>
            <button 
              className={`control-btn ${activeButtons.signals ? 'active' : ''}`}
              onClick={() => handleButtonClick('signals')}
            >
              Signals
            </button>
            <button 
              className={`control-btn ${activeButtons.speed ? 'active' : ''}`}
              onClick={() => handleButtonClick('speed')}
            >
              Speed
            </button>
            <button 
              className={`control-btn ${activeButtons.alerts ? 'active' : ''}`}
              onClick={() => handleButtonClick('alerts')}
            >
              Alerts
            </button>
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
                className="connection-line"
              />
            ))}

            {/* Draw track sections */}
            {TRACK_SECTIONS.map(section => {
              const state = getSectionState(section.id);
              const trainInSection = getTrainInSection(section.id);
              
              return (
                <g key={section.id}>
                  <rect
                    x={section.x}
                    y={section.y}
                    width={section.width}
                    height={section.height}
                    className={`track-section track-${state}`}
                    rx="3"
                  />
                  <text
                    x={section.x + section.width / 2}
                    y={section.y - 15}
                    className="track-label"
                  >
                    {section.id}
                  </text>
                  
                  {/* Draw train if present */}
                  {trainInSection && (
                    <g 
                      className={`train-group ${selectedTrain?.id === trainInSection.id ? 'selected' : ''}`}
                      onClick={(e) => handleTrainClick(trainInSection, e)}
                      onMouseEnter={(e) => handleTrainHover(trainInSection, e)}
                      onMouseLeave={handleTrainLeave}
                    >
                      <rect
                        x={getSectionCenter(section).x - 18}
                        y={getSectionCenter(section).y - 8}
                        width={36}
                        height={16}
                        rx="8"
                        className="train-body"
                      />
                      <text
                        x={getSectionCenter(section).x}
                        y={getSectionCenter(section).y + 25}
                        className="train-label"
                      >
                        {trainInSection.number}
                      </text>
                    </g>
                  )}
                </g>
              );
            })}

            {/* Signals at key junctions */}
            <circle cx="200" cy="175" r="5" className="signal signal-green" />
            <circle cx="280" cy="225" r="5" className="signal signal-red" />
            <circle cx="520" cy="175" r="5" className="signal signal-green" />
            <circle cx="580" cy="225" r="5" className="signal signal-green" />
          </svg>
        </div>
      </div>

      {/* Control Panel with Railway Optimization Options */}
      <div className="control-panel">
        {/* Operations Section */}
        <div className="panel-section">
          <div className="panel-header">Operations</div>
          {menuItems.filter(item => item.category === 'operations').map(item => (
            <div 
              key={item.id}
              className={`menu-item ${activeMenuItem === item.id ? 'active' : ''}`}
              onClick={() => handleMenuItemClick(item.id)}
            >
              <div className={`menu-icon ${item.icon}`}></div>
              {item.label}
            </div>
          ))}
        </div>

        {/* Optimization Section */}
        <div className="panel-section">
          <div className="panel-header">Optimization</div>
          {menuItems.filter(item => item.category === 'optimization').map(item => (
            <div 
              key={item.id}
              className={`menu-item ${activeMenuItem === item.id ? 'active' : ''}`}
              onClick={() => handleMenuItemClick(item.id)}
            >
              <div className={`menu-icon ${item.icon}`}></div>
              {item.label}
            </div>
          ))}
        </div>

        {/* AI & Decision Support */}
        <div className="panel-section">
          <div className="panel-header">AI & Decision Support</div>
          {menuItems.filter(item => item.category === 'ai').map(item => (
            <div 
              key={item.id}
              className={`menu-item ${activeMenuItem === item.id ? 'active' : ''}`}
              onClick={() => handleMenuItemClick(item.id)}
            >
              <div className={`menu-icon ${item.icon}`}></div>
              {item.label}
            </div>
          ))}
        </div>

        {/* Analysis & Reporting */}
        <div className="panel-section">
          <div className="panel-header">Analysis & Reporting</div>
          {menuItems.filter(item => item.category === 'analysis').map(item => (
            <div 
              key={item.id}
              className={`menu-item ${activeMenuItem === item.id ? 'active' : ''}`}
              onClick={() => handleMenuItemClick(item.id)}
            >
              <div className={`menu-icon ${item.icon}`}></div>
              {item.label}
            </div>
          ))}
        </div>

        {/* Active Trains Section */}
        <div className="panel-section">
          <div className="panel-header">Active Trains ({trains.length})</div>
          
          {trains.map(train => (
            <div 
              key={train.id} 
              className={`train-item ${selectedTrain?.id === train.id ? 'selected' : ''}`}
              onClick={() => setSelectedTrain(selectedTrain?.id === train.id ? null : train)}
            >
              <div className={`train-status-dot ${train.statusType}`}></div>
              <div className="train-details">
                <div className="train-name">{train.name}</div>
                <div className="train-info">
                  {train.number} | {train.section} | {Math.round(train.speed)} km/h
                  {train.delay > 0 && ` | +${train.delay}min`}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Enhanced Tooltip */}
      {hoveredTrain && (
        <div 
          className="train-tooltip"
          style={{
            left: Math.min(mousePos.x + 20, window.innerWidth - 340),
            top: Math.max(mousePos.y - 180, 10)
          }}
        >
          <div className="tooltip-header">{hoveredTrain.name}</div>
          
          <div className="tooltip-section">
            <div className="tooltip-row">
              <span className="tooltip-label">Train Number:</span>
              <span className="tooltip-value">{hoveredTrain.number}</span>
            </div>
            <div className="tooltip-row">
              <span className="tooltip-label">Current Speed:</span>
              <span className="tooltip-value tooltip-speed">
                {Math.round(hoveredTrain.speed)} km/h
              </span>
            </div>
            <div className="tooltip-row">
              <span className="tooltip-label">Current Section:</span>
              <span className="tooltip-value tooltip-section-id">{hoveredTrain.section}</span>
            </div>
            <div className="tooltip-row">
              <span className="tooltip-label">Destination:</span>
              <span className="tooltip-value tooltip-destination">{hoveredTrain.destination}</span>
            </div>
            <div className="tooltip-row">
              <span className="tooltip-label">Status:</span>
              <span className="tooltip-value tooltip-status">{hoveredTrain.status}</span>
            </div>
            <div className="tooltip-row">
              <span className="tooltip-label">Delay:</span>
              <span className="tooltip-value">
                {hoveredTrain.delay > 0 ? `+${hoveredTrain.delay} min` : 'On Time'}
              </span>
            </div>
            <div className="tooltip-row">
              <span className="tooltip-label">Next Section:</span>
              <span className="tooltip-value tooltip-section-id">
                {hoveredTrain.route[(routeIndex[hoveredTrain.id] + 1) % hoveredTrain.route.length]}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TrainTrafficControl;