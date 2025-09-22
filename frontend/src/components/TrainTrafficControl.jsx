import React, { useState, useEffect } from 'react';
import './TrainTrafficControl.css';

// Track sections with realistic block names and station definitions
const TRACK_SECTIONS = [
  // Main horizontal line with proper block names
  { id: 'ENTRY_BLOCK', x: 80, y: 200, width: 60, height: 8, type: 'block', name: 'Entry Block' },
  { id: 'STN_A', x: 160, y: 200, width: 60, height: 8, type: 'station', station: 'A', platforms: 3, name: 'Central Stn' },
  { id: 'STN_B', x: 240, y: 200, width: 60, height: 8, type: 'station', station: 'B', platforms: 2, name: 'Junction Stn' },
  { id: 'BLOCK_AB', x: 320, y: 200, width: 60, height: 8, type: 'block', name: 'AB Block' },
  { id: 'BLOCK_BC', x: 400, y: 200, width: 60, height: 8, type: 'block', name: 'BC Block' },
  { id: 'STN_C', x: 480, y: 200, width: 60, height: 8, type: 'station', station: 'C', platforms: 2, name: 'Metro Stn' },
  { id: 'BLOCK_CD1', x: 560, y: 200, width: 60, height: 8, type: 'block', name: 'CD Block 1' },
  { id: 'BLOCK_CD2', x: 640, y: 200, width: 60, height: 8, type: 'block', name: 'CD Block 2' },
  { id: 'STN_D', x: 720, y: 200, width: 60, height: 8, type: 'station', station: 'D', platforms: 4, name: 'Terminal Stn' },
  
  // Upper branch line
  { id: 'BRANCH_N1', x: 240, y: 120, width: 80, height: 8, type: 'block', name: 'North Branch 1' },
  { id: 'BRANCH_N2', x: 340, y: 120, width: 80, height: 8, type: 'block', name: 'North Branch 2' },
  { id: 'BRANCH_N3', x: 440, y: 120, width: 80, height: 8, type: 'block', name: 'North Branch 3' },
  { id: 'BRANCH_N4', x: 540, y: 120, width: 80, height: 8, type: 'block', name: 'North Branch 4' },
  
  // Lower branch line
  { id: 'BRANCH_S1', x: 240, y: 280, width: 80, height: 8, type: 'block', name: 'South Branch 1' },
  { id: 'BRANCH_S2', x: 340, y: 280, width: 80, height: 8, type: 'block', name: 'South Branch 2' },
  { id: 'BRANCH_S3', x: 440, y: 280, width: 80, height: 8, type: 'block', name: 'South Branch 3' },
  { id: 'BRANCH_S4', x: 540, y: 280, width: 80, height: 8, type: 'block', name: 'South Branch 4' },
  
  // Yard tracks
  { id: 'YARD_1', x: 80, y: 350, width: 100, height: 8, type: 'block', name: 'Yard Block 1' },
  { id: 'YARD_2', x: 200, y: 350, width: 100, height: 8, type: 'block', name: 'Yard Block 2' },
  { id: 'YARD_3', x: 320, y: 350, width: 100, height: 8, type: 'block', name: 'Yard Block 3' },
  { id: 'YARD_4', x: 440, y: 350, width: 100, height: 8, type: 'block', name: 'Yard Block 4' },
];

// Connection paths between sections
const CONNECTIONS = [
  { from: 'ENTRY_BLOCK', to: 'STN_A', type: 'main', path: 'M140,204 L160,204' },
  { from: 'STN_A', to: 'STN_B', type: 'main', path: 'M220,204 L240,204' },
  { from: 'STN_B', to: 'BLOCK_AB', type: 'main', path: 'M300,204 L320,204' },
  { from: 'BLOCK_AB', to: 'BLOCK_BC', type: 'main', path: 'M380,204 L400,204' },
  { from: 'BLOCK_BC', to: 'STN_C', type: 'main', path: 'M460,204 L480,204' },
  { from: 'STN_C', to: 'BLOCK_CD1', type: 'main', path: 'M540,204 L560,204' },
  { from: 'BLOCK_CD1', to: 'BLOCK_CD2', type: 'main', path: 'M620,204 L640,204' },
  { from: 'BLOCK_CD2', to: 'STN_D', type: 'main', path: 'M700,204 L720,204' },
  { from: 'BRANCH_N1', to: 'BRANCH_N2', type: 'branch', path: 'M320,124 L340,124' },
  { from: 'BRANCH_N2', to: 'BRANCH_N3', type: 'branch', path: 'M420,124 L440,124' },
  { from: 'BRANCH_N3', to: 'BRANCH_N4', type: 'branch', path: 'M520,124 L540,124' },
  { from: 'BRANCH_S1', to: 'BRANCH_S2', type: 'branch', path: 'M320,284 L340,284' },
  { from: 'BRANCH_S2', to: 'BRANCH_S3', type: 'branch', path: 'M420,284 L440,284' },
  { from: 'BRANCH_S3', to: 'BRANCH_S4', type: 'branch', path: 'M520,284 L540,284' },
  { from: 'YARD_1', to: 'YARD_2', type: 'yard', path: 'M180,354 L200,354' },
  { from: 'YARD_2', to: 'YARD_3', type: 'yard', path: 'M300,354 L320,354' },
  { from: 'YARD_3', to: 'YARD_4', type: 'yard', path: 'M420,354 L440,354' },
  { from: 'STN_A', to: 'BRANCH_N1', type: 'junction', path: 'M190,200 L190,160 L240,160 L240,132' },
  { from: 'BRANCH_N4', to: 'STN_C', type: 'junction', path: 'M580,132 L580,160 L510,160 L510,200' },
  { from: 'STN_B', to: 'BRANCH_S1', type: 'junction', path: 'M270,208 L270,240 L280,240 L280,272' },
  { from: 'BRANCH_S4', to: 'BLOCK_CD1', type: 'junction', path: 'M580,288 L580,240 L590,240 L590,208' },
];

const TrainTrafficControl = () => {
  const [trains, setTrains] = useState([]);
  const [hoveredTrain, setHoveredTrain] = useState(null);
  const [selectedTrain, setSelectedTrain] = useState(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [currentTime, setCurrentTime] = useState(new Date());
  const [routeIndex, setRouteIndex] = useState({});
  const [activeMenuItem, setActiveMenuItem] = useState('live-monitoring');
  const [simulationTime, setSimulationTime] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [schedule, setSchedule] = useState({});
  const [trainProgress, setTrainProgress] = useState({});
  const [blockOccupancy, setBlockOccupancy] = useState({});
  const [stationPlatforms, setStationPlatforms] = useState({});
  const [activeButtons, setActiveButtons] = useState({
    overview: true,
    signals: false,
    speed: false,
    alerts: false
  });

  // Menu items
  const menuItems = [
    { id: 'live-monitoring', label: 'Live Monitoring', icon: 'standard', category: 'operations' },
    { id: 'audit-trail', label: 'Audit Trail', icon: 'standard', category: 'operations' },
    { id: 'train-precedence', label: 'Train Precedence', icon: 'optimization', category: 'optimization' },
    { id: 'crossing-optimization', label: 'Crossing Optimization', icon: 'optimization', category: 'optimization' },
    { id: 'route-planning', label: 'Route Planning', icon: 'optimization', category: 'optimization' },
    { id: 'resource-utilization', label: 'Resource Utilization', icon: 'optimization', category: 'optimization' },
    { id: 'conflict-resolution', label: 'Conflict Resolution', icon: 'ai', category: 'ai' },
    { id: 'ai-recommendations', label: 'AI Recommendations', icon: 'ai', category: 'ai' },
    { id: 'predictive-analysis', label: 'Predictive Analysis', icon: 'ai', category: 'ai' },
    { id: 'disruption-management', label: 'Disruption Management', icon: 'ai', category: 'ai' },
    { id: 'what-if-simulation', label: 'What-If Simulation', icon: 'analysis', category: 'analysis' },
    { id: 'scenario-analysis', label: 'Scenario Analysis', icon: 'analysis', category: 'analysis' },
    { id: 'performance-dashboard', label: 'Performance Dashboard', icon: 'analysis', category: 'analysis' },
    { id: 'throughput-analysis', label: 'Throughput Analysis', icon: 'analysis', category: 'analysis' },
    { id: 'delay-analytics', label: 'Delay Analytics', icon: 'analysis', category: 'analysis' },
  ];

  // Initialize block occupancy and station platforms
  useEffect(() => {
    const initialBlockOccupancy = {};
    const initialStationPlatforms = {};
    
    TRACK_SECTIONS.forEach(section => {
      if (section.type === 'block') {
        initialBlockOccupancy[section.id] = null;
      } else if (section.type === 'station') {
        initialStationPlatforms[section.id] = {};
        for (let i = 1; i <= (section.platforms || 1); i++) {
          initialStationPlatforms[section.id][i] = null;
        }
      }
    });
    
    setBlockOccupancy(initialBlockOccupancy);
    setStationPlatforms(initialStationPlatforms);
  }, []);

  // Check if a section is available for a train
  const isSectionAvailable = (sectionId, trainId) => {
    const section = TRACK_SECTIONS.find(s => s.id === sectionId);
    if (!section) return false;
    
    if (section.type === 'block') {
      return blockOccupancy[sectionId] === null || blockOccupancy[sectionId] === trainId;
    } else if (section.type === 'station') {
      const platforms = stationPlatforms[sectionId] || {};
      return Object.values(platforms).some(occupant => occupant === null || occupant === trainId);
    }
    return false;
  };

  // Occupy a section with a train
  const occupySection = (sectionId, trainId) => {
    const section = TRACK_SECTIONS.find(s => s.id === sectionId);
    if (!section) return;
    
    if (section.type === 'block') {
      setBlockOccupancy(prev => ({
        ...prev,
        [sectionId]: trainId
      }));
    } else if (section.type === 'station') {
      setStationPlatforms(prev => {
        const newPlatforms = { ...prev };
        const platforms = newPlatforms[sectionId] || {};
        
        // Find first available platform or platform already occupied by this train
        for (let platformNum = 1; platformNum <= (section.platforms || 1); platformNum++) {
          if (platforms[platformNum] === null || platforms[platformNum] === trainId) {
            platforms[platformNum] = trainId;
            break;
          }
        }
        
        newPlatforms[sectionId] = platforms;
        return newPlatforms;
      });
    }
  };

  // Release a section from a train
  const releaseSection = (sectionId, trainId) => {
    const section = TRACK_SECTIONS.find(s => s.id === sectionId);
    if (!section) return;
    
    if (section.type === 'block') {
      setBlockOccupancy(prev => ({
        ...prev,
        [sectionId]: prev[sectionId] === trainId ? null : prev[sectionId]
      }));
    } else if (section.type === 'station') {
      setStationPlatforms(prev => {
        const newPlatforms = { ...prev };
        const platforms = newPlatforms[sectionId] || {};
        
        // Release platform occupied by this train
        Object.keys(platforms).forEach(platformNum => {
          if (platforms[platformNum] === trainId) {
            platforms[platformNum] = null;
          }
        });
        
        newPlatforms[sectionId] = platforms;
        return newPlatforms;
      });
    }
  };

  // Mock data initialization
  useEffect(() => {
    const mockTrains = [
      {
        id: 'T1',
        name: 'Rajdhani Express',
        number: '12301',
        section: 'STN_A',
        speed: 80,
        destination: 'STN_D',
        status: 'Scheduled',
        delay: 0,
        route: ['STN_A', 'STN_B', 'BLOCK_AB', 'BLOCK_BC', 'STN_C', 'BLOCK_CD1', 'BLOCK_CD2', 'STN_D'],
        statusType: 'scheduled',
        departureTime: 0,
        schedule: { 'STN_B': [5, 1], 'STN_C': [12, 2], 'STN_D': [20, 1] },
        platform: null,
        waitingForBlock: false
      },
      {
        id: 'T2',
        name: 'Shatabdi Express',
        number: '12002',
        section: 'STN_A',
        speed: 60,
        destination: 'STN_D',
        status: 'Scheduled',
        delay: 0,
        route: ['STN_A', 'STN_B', 'BLOCK_AB', 'BLOCK_BC', 'STN_C', 'BLOCK_CD1', 'BLOCK_CD2', 'STN_D'],
        statusType: 'scheduled',
        departureTime: 3,
        schedule: { 'STN_B': [8, 2], 'STN_C': [16, 1], 'STN_D': [25, 2] },
        platform: null,
        waitingForBlock: false
      },
      {
        id: 'T3',
        name: 'Duronto Express',
        number: '12259',
        section: 'STN_A',
        speed: 45,
        destination: 'STN_D',
        status: 'Scheduled',
        delay: 0,
        route: ['STN_A', 'STN_B', 'BLOCK_AB', 'BLOCK_BC', 'STN_C', 'BLOCK_CD1', 'BLOCK_CD2', 'STN_D'],
        statusType: 'scheduled',
        departureTime: 6,
        schedule: { 'STN_B': [11, 1], 'STN_C': [20, 2], 'STN_D': [30, 1] },
        platform: null,
        waitingForBlock: false
      }
    ];

    setTrains(mockTrains);
    
    // Initialize progress for mock data
    const initialProgress = {};
    mockTrains.forEach(train => {
      initialProgress[train.id] = {
        currentRouteIndex: 0,
        lastMoveTime: 0,
        isMoving: false,
        nextScheduledTime: 0,
        waitingForSection: null
      };
    });
    setTrainProgress(initialProgress);
  }, []);

  // Initialize route indices and occupy initial sections
  useEffect(() => {
    if (trains.length === 0) return;
    
    const initialIndices = {};
    trains.forEach(train => {
      initialIndices[train.id] = 0;
      // Occupy initial sections with a slight delay to ensure state is ready
      setTimeout(() => {
        occupySection(train.section, train.id);
      }, 100);
    });
    setRouteIndex(initialIndices);
  }, [trains]);

  // Clock
  useEffect(() => {
    const clock = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(clock);
  }, []);

  // Enhanced simulation control with better train tracking
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setSimulationTime(prev => prev + 1);
      
      // Update train positions with enhanced block control
      setTrains(prevTrains => {
        const updatedTrains = [...prevTrains];
        
        prevTrains.forEach((train, trainIdx) => {
          const progress = trainProgress[train.id];
          if (!progress || !train.route || train.route.length === 0) return;

          const newTrain = { ...train };
          
          // Check if train should start moving
          if (simulationTime >= train.departureTime) {
            newTrain.statusType = 'running';
            newTrain.status = 'Running';
            
            const currentIndex = progress.currentRouteIndex;
            const currentSection = train.route[currentIndex];
            const nextSection = train.route[currentIndex + 1];
            
            if (nextSection && currentIndex < train.route.length - 1) {
              const schedule = train.schedule || {};
              let shouldMove = false;
              
              // Check scheduled arrival times
              if (schedule[nextSection]) {
                const [scheduledArrival] = schedule[nextSection];
                if (simulationTime >= scheduledArrival) {
                  shouldMove = true;
                }
              } else {
                // Use travel time calculation for intermediate sections
                const timeInCurrentSection = simulationTime - progress.lastMoveTime;
                const baseTimePerSection = Math.max(4, Math.floor(90 / train.speed)); // Minimum 4 minutes per section
                
                if (timeInCurrentSection >= baseTimePerSection) {
                  shouldMove = true;
                }
              }
              
              // Check if next section is available (enhanced block control)
              if (shouldMove) {
                if (isSectionAvailable(nextSection, train.id)) {
                  // Release current section
                  releaseSection(currentSection, train.id);
                  
                  // Occupy next section
                  occupySection(nextSection, train.id);
                  
                  // Move to next section
                  setTrainProgress(prevProgress => ({
                    ...prevProgress,
                    [train.id]: {
                      ...progress,
                      currentRouteIndex: currentIndex + 1,
                      lastMoveTime: simulationTime,
                      isMoving: true,
                      waitingForSection: null
                    }
                  }));
                  
                  newTrain.section = nextSection;
                  newTrain.waitingForBlock = false;
                  newTrain.status = 'Running';
                  
                  // Update route index
                  setRouteIndex(prevIndex => ({
                    ...prevIndex,
                    [train.id]: currentIndex + 1
                  }));
                  
                } else {
                  // Train is waiting for next section
                  newTrain.waitingForBlock = true;
                  newTrain.status = `Waiting for ${TRACK_SECTIONS.find(s => s.id === nextSection)?.name || nextSection}`;
                  
                  setTrainProgress(prevProgress => ({
                    ...prevProgress,
                    [train.id]: {
                      ...progress,
                      waitingForSection: nextSection
                    }
                  }));
                }
              } else {
                newTrain.section = currentSection;
                newTrain.waitingForBlock = false;
              }
            } else if (currentIndex >= train.route.length - 1) {
              // Train has reached destination
              newTrain.section = train.route[train.route.length - 1];
              newTrain.status = 'Arrived at Terminal Station';
              newTrain.statusType = 'completed';
              newTrain.waitingForBlock = false;
            }
            
            // Add realistic speed variation
            const speedVariation = (Math.random() - 0.5) * 8;
            newTrain.speed = Math.max(25, Math.min(120, train.speed + speedVariation));
          } else {
            // Train not yet departed
            newTrain.section = train.route[0];
            newTrain.statusType = 'scheduled';
            newTrain.status = `Departing at ${String(Math.floor(train.departureTime / 60)).padStart(2, '0')}:${String(train.departureTime % 60).padStart(2, '0')}`;
            newTrain.waitingForBlock = false;
          }
          
          updatedTrains[trainIdx] = newTrain;
        });
        
        return updatedTrains;
      });
    }, 1800);

    return () => clearInterval(interval);
  }, [isRunning, simulationTime, trainProgress, blockOccupancy, stationPlatforms]);

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
  };

  const resetSimulation = () => {
    setSimulationTime(0);
    setIsRunning(false);
    
    // Clear all occupancy
    const resetBlockOccupancy = {};
    const resetStationPlatforms = {};
    
    TRACK_SECTIONS.forEach(section => {
      if (section.type === 'block') {
        resetBlockOccupancy[section.id] = null;
      } else if (section.type === 'station') {
        resetStationPlatforms[section.id] = {};
        for (let i = 1; i <= (section.platforms || 1); i++) {
          resetStationPlatforms[section.id][i] = null;
        }
      }
    });
    
    setBlockOccupancy(resetBlockOccupancy);
    setStationPlatforms(resetStationPlatforms);
    
    // Reset route indices
    const resetIndices = {};
    trains.forEach(train => {
      resetIndices[train.id] = 0;
    });
    setRouteIndex(resetIndices);
    
    // Reset train progress
    const resetProgress = {};
    trains.forEach(train => {
      resetProgress[train.id] = {
        currentRouteIndex: 0,
        lastMoveTime: 0,
        isMoving: false,
        nextScheduledTime: 0,
        waitingForSection: null
      };
    });
    setTrainProgress(resetProgress);
    
    // Reset train positions and re-occupy initial sections
    setTrains(prevTrains => 
      prevTrains.map(train => {
        const resetTrain = {
          ...train,
          section: train.route[0],
          statusType: 'scheduled',
          status: 'Scheduled',
          waitingForBlock: false,
          platform: null
        };
        
        // Re-occupy initial section after a brief delay
        setTimeout(() => {
          occupySection(train.route[0], train.id);
        }, 200);
        
        return resetTrain;
      })
    );
  };

  // Get platform assignment for a train at a station
  const getTrainPlatform = (trainId, sectionId) => {
    const platforms = stationPlatforms[sectionId] || {};
    for (let [platformNum, occupantId] of Object.entries(platforms)) {
      if (occupantId === trainId) {
        return parseInt(platformNum);
      }
    }
    return 1;
  };

  return (
    <div className="tms-container" onMouseMove={handleMouseMove}>
      {/* Enhanced Header */}
      <div className="tms-header">
        <div className="header-left">
          <div className="system-title">INTELLIGENT RAILWAY CONTROL SYSTEM</div>
          <div className="system-subtitle">BLOCK SIGNALING & TRAFFIC MANAGEMENT V4.0</div>
        </div>
        
        <div className="header-center">
          <div className="status-group">
            <div className="status-display green">
              {Object.values(blockOccupancy).filter(occupant => occupant === null).length}
            </div>
            <div className="status-label">FREE BLOCKS</div>
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
            <div className="status-display red">00</div>
            <div className="status-label">ALERTS</div>
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
              OVERVIEW
            </button>
            <button 
              className={`control-btn ${activeButtons.signals ? 'active' : ''}`}
              onClick={() => handleButtonClick('signals')}
            >
              SIGNALS
            </button>
            <button 
              className={`control-btn ${activeButtons.speed ? 'active' : ''}`}
              onClick={() => handleButtonClick('speed')}
            >
              SPEED
            </button>
            <button 
              className={`control-btn ${activeButtons.alerts ? 'active' : ''}`}
              onClick={() => handleButtonClick('alerts')}
            >
              ALERTS
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
                    } ${
                      isSelected ? 'track-selected' : ''
                    }`}
                    rx="4"
                  />
                  
                  {/* Section ID label - positioned above */}
                  <text
                    x={section.x + section.width / 2}
                    y={section.y - 8}
                    className="section-id-label"
                  >
                    {section.id}
                  </text>
                  
                  {/* Station-specific labels */}
                  {section.type === 'station' && (
                    <>
                      <text
                        x={section.x + section.width / 2}
                        y={section.y + 25}
                        className="station-name-label"
                      >
                        {section.name}
                      </text>
                      <text
                        x={section.x + section.width / 2}
                        y={section.y + 38}
                        className="platform-count-label"
                      >
                        {section.platforms}P
                      </text>
                      
                      {/* Platform status indicators */}
                      <g className="platform-indicators">
                        {Object.entries(stationPlatforms[section.id] || {}).map(([platformNum, occupant], idx) => (
                          <g key={platformNum}>
                            <circle
                              cx={section.x + 15 + (idx * 15)}
                              cy={section.y + 50}
                              r="5"
                              className={`platform-indicator ${occupant ? 'occupied' : 'free'}`}
                            />
                            <text
                              x={section.x + 15 + (idx * 15)}
                              y={section.y + 54}
                              className="platform-number"
                            >
                              {platformNum}
                            </text>
                          </g>
                        ))}
                      </g>
                    </>
                  )}
                  
                  {/* Draw trains with enhanced positioning */}
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
                        <rect
                          x={center.x - 20 + offsetX}
                          y={center.y - 10 + offsetY}
                          width={40}
                          height={20}
                          rx="10"
                          className={`train-body train-${train.statusType} ${
                            isTrainSelected ? 'train-selected' : ''
                          } ${train.waitingForBlock ? 'train-waiting' : ''}`}
                        />
                        <text
                          x={center.x + offsetX}
                          y={center.y + offsetY + 3}
                          className="train-number-label"
                        >
                          {train.number}
                        </text>
                        {train.waitingForBlock && (
                          <circle
                            cx={center.x + 25 + offsetX}
                            cy={center.y - 5 + offsetY}
                            r="4"
                            className="waiting-indicator"
                          />
                        )}
                      </g>
                    );
                  })}
                </g>
              );
            })}

            {/* Enhanced Signals */}
            <circle cx="200" cy="175" r="8" className={`signal ${blockOccupancy['STN_B'] ? 'signal-red' : 'signal-green'}`} />
            <circle cx="280" cy="225" r="8" className={`signal ${blockOccupancy['BLOCK_AB'] ? 'signal-red' : 'signal-green'}`} />
            <circle cx="520" cy="175" r="8" className={`signal ${blockOccupancy['STN_C'] ? 'signal-red' : 'signal-green'}`} />
            <circle cx="580" cy="225" r="8" className={`signal ${blockOccupancy['BLOCK_CD1'] ? 'signal-red' : 'signal-green'}`} />
          </svg>
        </div>
        
        {/* Simulation Controls */}
        <div className="simulation-controls">
          <div className="control-row">
            <button
              onClick={() => setIsRunning(!isRunning)}
              className={`sim-btn ${isRunning ? 'pause' : 'start'}`}
            >
              {isRunning ? '⏸ PAUSE' : '▶ START'}
            </button>
            <button
              onClick={resetSimulation}
              className="sim-btn reset"
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
          </div>
        </div>
      </div>

      {/* Control Panel */}
      <div className="control-panel">
        {/* Block Status Section */}
        <div className="panel-section">
          <div className="panel-header">BLOCK STATUS</div>
          <div className="block-status-grid">
            {Object.entries(blockOccupancy).slice(0, 8).map(([blockId, occupant]) => (
              <div key={blockId} className={`block-status-item ${occupant ? 'occupied' : 'free'}`}>
                <div className="block-id">{blockId}</div>
                <div className="block-occupant">{occupant || 'FREE'}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Station Status Section */}
        <div className="panel-section">
          <div className="panel-header">STATION STATUS</div>
          {TRACK_SECTIONS.filter(s => s.type === 'station').map(station => {
            const platforms = stationPlatforms[station.id] || {};
            const occupiedCount = Object.values(platforms).filter(p => p !== null).length;
            
            return (
              <div key={station.id} className="station-status-item">
                <div className="station-header">
                  <span className="station-name">{station.name} ({station.station})</span>
                  <span className="platform-count">Platforms: {station.platforms}</span>
                </div>
                <div className="platform-status">
                  <span className="occupancy-info">Occupied: {occupiedCount}/{station.platforms}</span>
                  <div className="platform-indicators-panel">
                    {Object.entries(platforms).map(([platformNum, occupant]) => (
                      <div key={platformNum} className={`platform-dot ${occupant ? 'occupied' : 'free'}`}>
                        P{platformNum}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Operations Section */}
        <div className="panel-section">
          <div className="panel-header">OPERATIONS</div>
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

        {/* Active Trains Section */}
        <div className="panel-section">
          <div className="panel-header">ACTIVE TRAINS ({trains.length})</div>
          
          {trains.map(train => {
            const currentSection = TRACK_SECTIONS.find(s => s.id === train.section);
            const isSelected = selectedTrain?.id === train.id;
            
            return (
              <div 
                key={train.id} 
                className={`train-item ${isSelected ? 'selected' : ''} ${train.waitingForBlock ? 'waiting' : ''}`}
                onClick={() => setSelectedTrain(isSelected ? null : train)}
              >
                <div className={`train-status-dot ${train.statusType} ${train.waitingForBlock ? 'waiting' : ''}`}></div>
                <div className="train-details">
                  <div className="train-name">{train.name}</div>
                  <div className="train-info">
                    {train.number} | {currentSection?.name || train.section} → Terminal | {Math.round(train.speed)} km/h
                    {train.delay > 0 && ` | +${train.delay}min`}
                    {train.waitingForBlock && (
                      <span className="waiting-status"> | WAITING</span>
                    )}
                  </div>
                  <div className="train-route-info">
                    Progress: {routeIndex[train.id] + 1 || 1}/{train.route?.length || 0}
                    {trainProgress[train.id]?.waitingForSection && (
                      <span className="waiting-for">
                        {' '}| Waiting for {TRACK_SECTIONS.find(s => s.id === trainProgress[train.id].waitingForSection)?.name || trainProgress[train.id].waitingForSection}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Enhanced Tooltip */}
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
              <div className="tooltip-row">
                <span className="tooltip-label">Block Status:</span>
                <span className={`tooltip-value ${hoveredTrain.waitingForBlock ? 'waiting' : 'clear'}`}>
                  {hoveredTrain.waitingForBlock ? 'WAITING FOR BLOCK' : 'CLEAR TO PROCEED'}
                </span>
              </div>
              <div className="tooltip-row">
                <span className="tooltip-label">Route Progress:</span>
                <span className="tooltip-value">{routeIndex[hoveredTrain.id] + 1 || 1} of {hoveredTrain.route?.length || 0}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TrainTrafficControl;