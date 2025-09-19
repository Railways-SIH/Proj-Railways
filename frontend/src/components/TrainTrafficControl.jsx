import React, { useState, useEffect } from 'react';

// Professional TMS styling with enhanced interactivity
const styles = `
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    background: #1a2332;
    font-family: 'JetBrains Mono', monospace;
    overflow: hidden;
  }

  .tms-container {
    width: 100vw;
    height: 100vh;
    background: linear-gradient(135deg, #1a2332 0%, #2a3441 100%);
    position: relative;
    overflow: hidden;
  }

  .tms-header {
    position: absolute;
    top: 15px;
    left: 20px;
    right: 20px;
    height: 80px;
    background: rgba(20, 30, 45, 0.95);
    border: 1px solid #3a4a5a;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 25px;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.3);
  }

  .header-left {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }

  .system-title {
    color: #ff6b6b;
    font-size: 18px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
  }

  .system-subtitle {
    color: #9aa5b1;
    font-size: 12px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }

  .header-center {
    display: flex;
    gap: 20px;
    align-items: center;
  }

  .status-group {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }

  .status-display {
    background: #000;
    color: #ff4444;
    padding: 6px 12px;
    border: 2px solid #333;
    border-radius: 4px;
    font-size: 20px;
    font-weight: 700;
    text-align: center;
    min-width: 70px;
    text-shadow: 0 0 10px currentColor;
    box-shadow: inset 0 0 10px rgba(255, 68, 68, 0.2);
  }

  .status-display.green {
    color: #44ff44;
    text-shadow: 0 0 10px #44ff44;
    box-shadow: inset 0 0 10px rgba(68, 255, 68, 0.2);
  }

  .status-display.yellow {
    color: #ffdd44;
    text-shadow: 0 0 10px #ffdd44;
    box-shadow: inset 0 0 10px rgba(255, 221, 68, 0.2);
  }

  .status-label {
    color: #9aa5b1;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .header-right {
    display: flex;
    gap: 15px;
    align-items: center;
  }

  .control-buttons {
    display: flex;
    gap: 10px;
  }

  .control-btn {
    background: rgba(74, 90, 106, 0.3);
    border: 1px solid #4a5a6a;
    border-radius: 4px;
    padding: 8px 12px;
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .control-btn:hover {
    background: rgba(74, 90, 106, 0.6);
    border-color: #6a8a9a;
    transform: translateY(-1px);
  }

  .control-btn.active {
    background: rgba(255, 107, 107, 0.2);
    border-color: #ff6b6b;
    color: #ff6b6b;
  }

  .time-display {
    background: #000;
    color: #44ff44;
    padding: 8px 16px;
    border: 2px solid #333;
    border-radius: 4px;
    font-size: 16px;
    font-weight: 700;
    text-shadow: 0 0 8px #44ff44;
    box-shadow: inset 0 0 8px rgba(68, 255, 68, 0.2);
    text-align: center;
    min-width: 90px;
  }

  .compass {
    width: 45px;
    height: 45px;
    background: #2a3441;
    border: 2px solid #4a5a6a;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    font-size: 24px;
    font-weight: 700;
    position: relative;
  }

  .compass::after {
    content: '';
    position: absolute;
    top: 2px;
    right: 2px;
    width: 6px;
    height: 6px;
    background: #44ff44;
    border-radius: 50%;
    box-shadow: 0 0 6px #44ff44;
    animation: compass-blink 3s ease-in-out infinite;
  }

  @keyframes compass-blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  .main-display {
    position: absolute;
    top: 110px;
    left: 20px;
    right: 320px;
    bottom: 20px;
    background: rgba(20, 30, 45, 0.95);
    border: 1px solid #3a4a5a;
    border-radius: 6px;
    overflow: hidden;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.3);
  }

  .control-panel {
    position: absolute;
    top: 110px;
    right: 20px;
    width: 280px;
    bottom: 20px;
    background: rgba(20, 30, 45, 0.95);
    border: 1px solid #3a4a5a;
    border-radius: 6px;
    overflow-y: auto;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.3);
  }

  .panel-section {
    padding: 20px;
    border-bottom: 1px solid rgba(58, 74, 90, 0.3);
  }

  .panel-header {
    color: #ff6b6b;
    font-size: 14px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 15px;
    text-shadow: 0 0 8px rgba(255, 107, 107, 0.5);
  }

  .menu-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 15px;
    margin-bottom: 8px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .menu-item:hover {
    background: rgba(255, 255, 255, 0.1);
    border-color: rgba(255, 107, 107, 0.4);
    transform: translateX(3px);
    color: #ff6b6b;
  }

  .menu-item.active {
    background: rgba(255, 107, 107, 0.2);
    border-color: #ff6b6b;
    color: #ff6b6b;
  }

  .menu-icon {
    width: 12px;
    height: 12px;
    background: #44ff44;
    border-radius: 2px;
    box-shadow: 0 0 6px #44ff44;
  }

  .menu-icon.optimization {
    background: #ffaa44;
    box-shadow: 0 0 6px #ffaa44;
  }

  .menu-icon.ai {
    background: #44aaff;
    box-shadow: 0 0 6px #44aaff;
  }

  .menu-icon.analysis {
    background: #aa44ff;
    box-shadow: 0 0 6px #aa44ff;
  }

  .track-container {
    width: 100%;
    height: 100%;
    position: relative;
    background: 
      radial-gradient(circle at 20% 30%, rgba(0, 255, 0, 0.02) 0%, transparent 50%),
      radial-gradient(circle at 80% 70%, rgba(255, 255, 0, 0.02) 0%, transparent 50%);
  }

  .track-svg {
    width: 100%;
    height: 100%;
  }

  /* Track sections */
  .track-section {
    transition: all 0.3s ease;
    stroke-width: 0;
    cursor: default;
  }

  .track-free {
    fill: #2d5a2d;
    filter: drop-shadow(0 0 2px #2d5a2d);
  }

  .track-occupied {
    fill: #ffdd44;
    filter: drop-shadow(0 0 12px #ffdd44);
    animation: track-pulse 1.5s ease-in-out infinite;
  }

  .track-fault {
    fill: #ff4444;
    filter: drop-shadow(0 0 8px #ff4444);
    animation: track-alert 1s ease-in-out infinite alternate;
  }

  @keyframes track-pulse {
    0%, 100% { opacity: 0.8; filter: drop-shadow(0 0 8px #ffdd44); }
    50% { opacity: 1; filter: drop-shadow(0 0 16px #ffdd44); }
  }

  @keyframes track-alert {
    0% { opacity: 0.7; }
    100% { opacity: 1; }
  }

  /* Track labels */
  .track-label {
    fill: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 500;
    text-anchor: middle;
    dominant-baseline: middle;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    pointer-events: none;
  }

  /* Connections */
  .connection-line {
    stroke: #4a5a6a;
    stroke-width: 4;
    fill: none;
    filter: drop-shadow(0 0 3px #4a5a6a);
  }

  .connection-active {
    stroke: #ffaa44;
    filter: drop-shadow(0 0 6px #ffaa44);
    animation: connection-pulse 2s ease-in-out infinite;
  }

  @keyframes connection-pulse {
    0%, 100% { opacity: 0.7; }
    50% { opacity: 1; }
  }

  /* Signals */
  .signal {
    fill: #666;
    stroke: #999;
    stroke-width: 1;
  }

  .signal-green {
    fill: #44ff44;
    filter: drop-shadow(0 0 6px #44ff44);
    animation: signal-glow 3s ease-in-out infinite;
  }

  .signal-red {
    fill: #ff4444;
    filter: drop-shadow(0 0 6px #ff4444);
    animation: signal-glow 3s ease-in-out infinite;
  }

  @keyframes signal-glow {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
  }

  /* Enhanced train interactions - FIXED */
  .train-group {
    cursor: pointer;
    transition: all 0.3s ease;
    pointer-events: all;
  }

  .train-body {
    fill: #ff6b6b;
    stroke: #ffffff;
    stroke-width: 2;
    filter: drop-shadow(0 0 8px #ff6b6b);
    transition: all 0.3s ease;
    pointer-events: all;
  }

  .train-group:hover .train-body {
    fill: #ff8a8a;
    filter: drop-shadow(0 0 20px #ff6b6b);
    stroke-width: 3;
  }

  .train-group.selected .train-body {
    fill: #ffaa44;
    filter: drop-shadow(0 0 25px #ffaa44);
    stroke: #ffff88;
    stroke-width: 4;
  }

  .train-label {
    fill: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    text-anchor: middle;
    dominant-baseline: middle;
    pointer-events: none;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    transition: all 0.3s ease;
  }

  .train-group:hover .train-label {
    fill: #ffff88;
    font-size: 11px;
    filter: drop-shadow(0 0 6px #ffff88);
  }

  .train-group.selected .train-label {
    fill: #ffff88;
    font-size: 12px;
    filter: drop-shadow(0 0 8px #ffff88);
  }

  /* Enhanced tooltip */
  .train-tooltip {
    position: fixed;
    background: linear-gradient(135deg, rgba(20, 30, 45, 0.98) 0%, rgba(30, 40, 55, 0.98) 100%);
    border: 2px solid #4a5a6a;
    border-radius: 8px;
    padding: 20px;
    min-width: 320px;
    z-index: 1000;
    pointer-events: none;
    box-shadow: 
      0 12px 40px rgba(0, 0, 0, 0.6),
      inset 0 2px 6px rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(15px);
    font-family: 'JetBrains Mono', monospace;
    animation: tooltip-appear 0.3s ease-out;
  }

  @keyframes tooltip-appear {
    from { 
      opacity: 0; 
      transform: translateY(15px) scale(0.9); 
    }
    to { 
      opacity: 1; 
      transform: translateY(0) scale(1); 
    }
  }

  .tooltip-header {
    color: #ff6b6b;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 1px;
    text-shadow: 0 0 8px rgba(255, 107, 107, 0.5);
    border-bottom: 1px solid rgba(255, 107, 107, 0.3);
    padding-bottom: 8px;
  }

  .tooltip-section {
    margin-bottom: 12px;
  }

  .tooltip-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    font-size: 14px;
  }

  .tooltip-label {
    color: #9aa5b1;
    font-weight: 500;
    text-transform: uppercase;
    font-size: 12px;
    letter-spacing: 0.5px;
  }

  .tooltip-value {
    color: #ffffff;
    font-weight: 700;
  }

  .tooltip-speed {
    color: #44ff44;
    text-shadow: 0 0 6px rgba(68, 255, 68, 0.4);
  }

  .tooltip-section-id {
    color: #ffdd44;
    text-shadow: 0 0 6px rgba(255, 221, 68, 0.4);
  }

  .tooltip-status {
    color: #ff6b6b;
    text-shadow: 0 0 6px rgba(255, 107, 107, 0.4);
  }

  .tooltip-destination {
    color: #88ddff;
    text-shadow: 0 0 6px rgba(136, 221, 255, 0.4);
  }

  /* Train info in panel */
  .train-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    margin-bottom: 8px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .train-item:hover {
    background: rgba(255, 255, 255, 0.1);
    border-color: rgba(255, 107, 107, 0.4);
    transform: translateX(3px);
  }

  .train-item.selected {
    background: rgba(255, 107, 107, 0.2);
    border-color: #ff6b6b;
  }

  .train-status-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #44ff44;
    box-shadow: 0 0 8px #44ff44;
    animation: status-pulse 2s ease-in-out infinite;
    flex-shrink: 0;
  }

  .train-status-dot.delayed {
    background: #ffaa44;
    box-shadow: 0 0 8px #ffaa44;
  }

  .train-status-dot.stopped {
    background: #ff4444;
    box-shadow: 0 0 8px #ff4444;
  }

  @keyframes status-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.7; transform: scale(1.1); }
  }

  .train-details {
    flex: 1;
    min-width: 0;
  }

  .train-name {
    color: #ffffff;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 2px;
    text-overflow: ellipsis;
    overflow: hidden;
    white-space: nowrap;
  }

  .train-info {
    color: #9aa5b1;
    font-size: 10px;
    text-overflow: ellipsis;
    overflow: hidden;
    white-space: nowrap;
  }

  /* Scrollbar styling */
  .control-panel::-webkit-scrollbar {
    width: 4px;
  }

  .control-panel::-webkit-scrollbar-track {
    background: rgba(58, 74, 90, 0.2);
  }

  .control-panel::-webkit-scrollbar-thumb {
    background: rgba(255, 107, 107, 0.3);
    border-radius: 2px;
  }

  .control-panel::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 107, 107, 0.5);
  }
`;

// Inject styles
const styleSheet = document.createElement("style");
styleSheet.textContent = styles;
document.head.appendChild(styleSheet);

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
  // Main line to upper branch
  { from: '2R', to: '101L', path: 'M200,200 Q220,160 240,128' },
  { from: '104L', to: '6L', path: 'M580,128 Q580,160 510,200' },
  
  // Main line to lower branch  
  { from: '3L', to: '201L', path: 'M270,208 Q270,240 280,272' },
  { from: '204L', to: '7L', path: 'M580,288 Q590,240 590,208' },
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

  // Update time every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);
 useEffect(() => {
    const moveInterval = setInterval(() => {
      setTrains(prevTrains =>
        prevTrains.map(train => {
          const route = train.route;
          if (!route || route.length === 0) return train;

          let index = routeIndex[train.id] ?? 0;
          let nextIndex = (index + 1) % route.length;

          // update routeIndex
          setRouteIndex(prev => ({ ...prev, [train.id]: nextIndex }));

          return { ...train, section: route[nextIndex] };
        })
      );
    }, 2000);

    return () => clearInterval(moveInterval);
  }, [routeIndex]);
  useEffect(() => {
    const init = {};
    TRAINS.forEach(t => {
      init[t.id] = 0;
    });
    setRouteIndex(init);
  }, []);
  // Simulate realistic train movement
  useEffect(() => {
    const interval = setInterval(() => {
      setTrains(prevTrains => {
        return prevTrains.map(train => {
          if (train.statusType === 'stopped') return train;
          
          const currentIndex = routeIndex[train.id] || 0;
          const nextIndex = (currentIndex + 1) % train.route.length;
          const nextSection = train.route[nextIndex];
          
          // Update route index
          setRouteIndex(prev => ({
            ...prev,
            [train.id]: nextIndex
          }));
          
          return {
            ...train,
            section: nextSection,
            speed: train.statusType === 'stopped' ? 0 : Math.max(60, Math.min(140, train.speed + (Math.random() - 0.5) * 15))
          };
        });
      });
    }, 3500);

    return () => clearInterval(interval);
  }, [routeIndex]);

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