import React, { useState, useEffect, useRef } from 'react';

// --- Expanded Network Configuration ---
const TRACK_SECTIONS = [
    // Main East-West Line
    { id: 'STN_A', x: 80, y: 200, width: 60, height: 8, type: 'station', station: 'A', platforms: 3, name: 'Central A' },
    { id: 'BLOCK_A1', x: 150, y: 200, width: 50, height: 8, type: 'block', name: 'Block A1' },
    { id: 'JCN_MAIN_1', x: 210, y: 200, width: 40, height: 8, type: 'junction', name: 'Junction 1' },
    { id: 'BLOCK_A2', x: 260, y: 200, width: 50, height: 8, type: 'block', name: 'Block A2' },
    { id: 'STN_B', x: 320, y: 200, width: 60, height: 8, type: 'station', station: 'B', platforms: 4, name: 'Metro B' },
    { id: 'BLOCK_B1', x: 390, y: 200, width: 50, height: 8, type: 'block', name: 'Block B1' },
    { id: 'CROSSOVER_1', x: 450, y: 200, width: 30, height: 8, type: 'crossover', name: 'XO-1' },
    { id: 'BLOCK_B2', x: 490, y: 200, width: 50, height: 8, type: 'block', name: 'Block B2' },
    { id: 'STN_C', x: 550, y: 200, width: 60, height: 8, type: 'station', station: 'C', platforms: 2, name: 'Terminal C' },
    { id: 'BLOCK_C1', x: 620, y: 200, width: 50, height: 8, type: 'block', name: 'Block C1' },
    { id: 'STN_C_EXT', x: 680, y: 200, width: 60, height: 8, type: 'station', station: 'C2', platforms: 2, name: 'C Extension' },

    // Northern Branch Line
    { id: 'STN_D', x: 80, y: 80, width: 60, height: 8, type: 'station', station: 'D', platforms: 2, name: 'North D' },
    { id: 'BLOCK_D1', x: 150, y: 80, width: 50, height: 8, type: 'block', name: 'Block D1' },
    { id: 'SIGNAL_D1', x: 205, y: 80, width: 10, height: 8, type: 'signal', name: 'Sig D1' },
    { id: 'BLOCK_D2', x: 220, y: 80, width: 50, height: 8, type: 'block', name: 'Block D2' },
    { id: 'STN_E', x: 220, y: 20, width: 60, height: 8, type: 'station', station: 'E', platforms: 2, name: 'Mountain E' },
    { id: 'BLOCK_E1', x: 290, y: 20, width: 50, height: 8, type: 'block', name: 'Block E1' },
    { id: 'STN_E_NORTH', x: 350, y: 20, width: 60, height: 8, type: 'station', station: 'E2', platforms: 1, name: 'E North' },
    { id: 'BLOCK_D3', x: 280, y: 80, width: 50, height: 8, type: 'block', name: 'Block D3' },
    { id: 'JCN_NORTH', x: 340, y: 80, width: 40, height: 8, type: 'junction', name: 'N Junction' },
    { id: 'BLOCK_D4', x: 390, y: 80, width: 50, height: 8, type: 'block', name: 'Block D4' },
    { id: 'BLOCK_D5', x: 450, y: 120, width: 50, height: 8, type: 'block', name: 'Block D5' },

    // Southern Branch Line
    { id: 'BLOCK_F1', x: 150, y: 280, width: 50, height: 8, type: 'block', name: 'Block F1' },
    { id: 'SIGNAL_F1', x: 205, y: 280, width: 10, height: 8, type: 'signal', name: 'Sig F1' },
    { id: 'BLOCK_F2', x: 220, y: 280, width: 50, height: 8, type: 'block', name: 'Block F2' },
    { id: 'STN_F', x: 280, y: 280, width: 60, height: 8, type: 'station', station: 'F', platforms: 2, name: 'South F' },
    { id: 'BLOCK_F3', x: 350, y: 280, width: 50, height: 8, type: 'block', name: 'Block F3' },
    { id: 'CROSSOVER_2', x: 410, y: 280, width: 30, height: 8, type: 'crossover', name: 'XO-2' },
    { id: 'BLOCK_F4', x: 450, y: 280, width: 50, height: 8, type: 'block', name: 'Block F4' },
    { id: 'STN_G', x: 510, y: 280, width: 60, height: 8, type: 'station', station: 'G', platforms: 3, name: 'Industrial G' },

    // Western Extension
    { id: 'STN_W1', x: 20, y: 200, width: 50, height: 8, type: 'station', station: 'W1', platforms: 2, name: 'West End' },
    { id: 'BLOCK_W1', x: 20, y: 140, width: 50, height: 8, type: 'block', name: 'Block W1' },
    { id: 'STN_W2', x: 20, y: 80, width: 50, height: 8, type: 'station', station: 'W2', platforms: 1, name: 'West 2' },

    // Connecting Junction Blocks
    { id: 'BLOCK_V_D2_A2', x: 220, y: 140, width: 50, height: 8, type: 'block', name: 'Connector 1' },
    { id: 'BLOCK_V_F2_A1', x: 150, y: 240, width: 50, height: 8, type: 'block', name: 'Connector 2' },
    { id: 'BLOCK_LOOP_1', x: 380, y: 140, width: 50, height: 8, type: 'block', name: 'Loop 1' },
    { id: 'BLOCK_LOOP_2', x: 450, y: 160, width: 50, height: 8, type: 'block', name: 'Loop 2' },

    // Eastern Extension
    { id: 'BLOCK_EAST_1', x: 750, y: 200, width: 50, height: 8, type: 'block', name: 'East 1' },
    { id: 'STN_EAST', x: 810, y: 200, width: 60, height: 8, type: 'station', station: 'EAST', platforms: 2, name: 'Eastern Term' },
    { id: 'BLOCK_EAST_2', x: 750, y: 160, width: 50, height: 8, type: 'block', name: 'East 2' },
    { id: 'STN_EAST_2', x: 810, y: 160, width: 60, height: 8, type: 'station', station: 'E3', platforms: 1, name: 'East Branch' },

    // Additional Signals
    { id: 'SIGNAL_A1', x: 135, y: 200, width: 10, height: 8, type: 'signal', name: 'Sig A1' },
    { id: 'SIGNAL_B1', x: 375, y: 200, width: 10, height: 8, type: 'signal', name: 'Sig B1' },
    { id: 'SIGNAL_C1', x: 535, y: 200, width: 10, height: 8, type: 'signal', name: 'Sig C1' },
    { id: 'SIGNAL_JCN_1', x: 325, y: 80, width: 10, height: 8, type: 'signal', name: 'Sig N1' },
];

const CONNECTIONS = [
    // Main East-West Line
    { from: 'STN_W1', to: 'STN_A', type: 'main', path: `M50,204 L110,204` },
    { from: 'STN_A', to: 'BLOCK_A1', type: 'main', path: `M140,204 L175,204` },
    { from: 'SIGNAL_A1', to: 'BLOCK_A1', type: 'main', path: `M145,204 L175,204` },
    { from: 'BLOCK_A1', to: 'JCN_MAIN_1', type: 'main', path: `M200,204 L230,204` },
    { from: 'JCN_MAIN_1', to: 'BLOCK_A2', type: 'main', path: `M250,204 L285,204` },
    { from: 'BLOCK_A2', to: 'STN_B', type: 'main', path: `M310,204 L350,204` },
    { from: 'STN_B', to: 'BLOCK_B1', type: 'main', path: `M380,204 L415,204` },
    { from: 'SIGNAL_B1', to: 'BLOCK_B1', type: 'main', path: `M385,204 L415,204` },
    { from: 'BLOCK_B1', to: 'CROSSOVER_1', type: 'main', path: `M440,204 L465,204` },
    { from: 'CROSSOVER_1', to: 'BLOCK_B2', type: 'main', path: `M480,204 L515,204` },
    { from: 'BLOCK_B2', to: 'STN_C', type: 'main', path: `M540,204 L580,204` },
    { from: 'SIGNAL_C1', to: 'STN_C', type: 'main', path: `M545,204 L580,204` },
    { from: 'STN_C', to: 'BLOCK_C1', type: 'main', path: `M610,204 L645,204` },
    { from: 'BLOCK_C1', to: 'STN_C_EXT', type: 'main', path: `M670,204 L710,204` },

    // Eastern Extension
    { from: 'STN_C_EXT', to: 'BLOCK_EAST_1', type: 'branch', path: `M740,204 L775,204` },
    { from: 'BLOCK_EAST_1', to: 'STN_EAST', type: 'branch', path: `M800,204 L840,204` },
    { from: 'BLOCK_EAST_1', to: 'BLOCK_EAST_2', type: 'branch', path: `M775,200 L775,164` },
    { from: 'BLOCK_EAST_2', to: 'STN_EAST_2', type: 'branch', path: `M800,164 L840,164` },

    // Northern Branch
    { from: 'STN_W2', to: 'STN_D', type: 'branch', path: `M70,84 L110,84` },
    { from: 'STN_D', to: 'BLOCK_D1', type: 'branch', path: `M140,84 L175,84` },
    { from: 'BLOCK_D1', to: 'SIGNAL_D1', type: 'branch', path: `M200,84 L210,84` },
    { from: 'SIGNAL_D1', to: 'BLOCK_D2', type: 'branch', path: `M215,84 L245,84` },
    { from: 'STN_E', to: 'BLOCK_D2', type: 'junction', path: `M250,28 L245,84` },
    { from: 'STN_E', to: 'BLOCK_E1', type: 'branch', path: `M280,24 L315,24` },
    { from: 'BLOCK_E1', to: 'STN_E_NORTH', type: 'branch', path: `M340,24 L380,24` },
    { from: 'BLOCK_D2', to: 'BLOCK_D3', type: 'branch', path: `M270,84 L305,84` },
    { from: 'BLOCK_D3', to: 'JCN_NORTH', type: 'branch', path: `M330,84 L360,84` },
    { from: 'SIGNAL_JCN_1', to: 'JCN_NORTH', type: 'branch', path: `M335,84 L360,84` },
    { from: 'JCN_NORTH', to: 'BLOCK_D4', type: 'branch', path: `M380,84 L415,84` },
    { from: 'BLOCK_D4', to: 'BLOCK_D5', type: 'branch', path: `M440,84 L475,124` },

    // Southern Branch  
    { from: 'BLOCK_A1', to: 'BLOCK_V_F2_A1', type: 'junction', path: `M175,204 L175,244` },
    { from: 'BLOCK_V_F2_A1', to: 'BLOCK_F1', type: 'branch', path: `M175,244 L175,284` },
    { from: 'BLOCK_F1', to: 'SIGNAL_F1', type: 'branch', path: `M200,284 L210,284` },
    { from: 'SIGNAL_F1', to: 'BLOCK_F2', type: 'branch', path: `M215,284 L245,284` },
    { from: 'BLOCK_F2', to: 'STN_F', type: 'branch', path: `M270,284 L310,284` },
    { from: 'STN_F', to: 'BLOCK_F3', type: 'branch', path: `M340,284 L375,284` },
    { from: 'BLOCK_F3', to: 'CROSSOVER_2', type: 'branch', path: `M400,284 L425,284` },
    { from: 'CROSSOVER_2', to: 'BLOCK_F4', type: 'branch', path: `M440,284 L475,284` },
    { from: 'BLOCK_F4', to: 'STN_G', type: 'branch', path: `M500,284 L540,284` },

    // Connector Lines
    { from: 'JCN_MAIN_1', to: 'BLOCK_V_D2_A2', type: 'junction', path: `M230,200 L245,144` },
    { from: 'BLOCK_V_D2_A2', to: 'BLOCK_D2', type: 'junction', path: `M245,144 L245,84` },
    { from: 'BLOCK_D5', to: 'BLOCK_LOOP_1', type: 'junction', path: `M475,124 L405,144` },
    { from: 'BLOCK_LOOP_1', to: 'BLOCK_LOOP_2', type: 'junction', path: `M430,144 L475,164` },
    { from: 'BLOCK_LOOP_2', to: 'CROSSOVER_1', type: 'junction', path: `M475,164 L465,204` },
    { from: 'CROSSOVER_2', to: 'BLOCK_LOOP_2', type: 'junction', path: `M425,284 L475,164` },

    // Western Connections
    { from: 'STN_W2', to: 'BLOCK_W1', type: 'branch', path: `M45,80 L45,144` },
    { from: 'BLOCK_W1', to: 'STN_W1', type: 'branch', path: `M45,144 L45,200` },
];

const TrainTrafficControl = () => {
    const [trains, setTrains] = useState([]);
    const [blockOccupancy, setBlockOccupancy] = useState({});
    const [stationPlatforms, setStationPlatforms] = useState({});
    const [simulationTime, setSimulationTime] = useState(0);
    const [isRunning, setIsRunning] = useState(false);
    const [hoveredTrain, setHoveredTrain] = useState(null);
    const [selectedTrain, setSelectedTrain] = useState(null);
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
    const [currentTime, setCurrentTime] = useState(new Date());
    const [signals, setSignals] = useState({});

    // Initialize system state
    useEffect(() => {
        // Initialize block occupancy
        const initialBlocks = {};
        const initialStations = {};
        const initialSignals = {};
        
        TRACK_SECTIONS.forEach(section => {
            if (section.type === 'block' || section.type === 'junction' || section.type === 'crossover') {
                initialBlocks[section.id] = null;
            } else if (section.type === 'station') {
                initialStations[section.id] = {};
                for (let i = 1; i <= section.platforms; i++) {
                    initialStations[section.id][i] = null;
                }
            } else if (section.type === 'signal') {
                initialSignals[section.id] = 'GREEN'; // GREEN, YELLOW, RED
            }
        });

        setBlockOccupancy(initialBlocks);
        setStationPlatforms(initialStations);
        setSignals(initialSignals);

        // Add some demo trains
        setTimeout(() => {
            setTrains([
                {
                    id: 'T1', number: '12301', name: 'Express Alpha', section: 'STN_A',
                    speed: 120, statusType: 'running', status: 'En route', waitingForBlock: false,
                    route: ['STN_A', 'BLOCK_A1', 'JCN_MAIN_1', 'BLOCK_A2', 'STN_B', 'BLOCK_B1', 'STN_C']
                },
                {
                    id: 'T2', number: '12302', name: 'Freight Beta', section: 'STN_D',
                    speed: 60, statusType: 'running', status: 'En route', waitingForBlock: false,
                    route: ['STN_D', 'BLOCK_D1', 'BLOCK_D2', 'BLOCK_D3', 'JCN_NORTH', 'BLOCK_D4']
                },
                {
                    id: 'T3', number: '12303', name: 'Commuter Gamma', section: 'STN_F',
                    speed: 80, statusType: 'running', status: 'En route', waitingForBlock: false,
                    route: ['STN_F', 'BLOCK_F3', 'CROSSOVER_2', 'BLOCK_F4', 'STN_G']
                },
                {
                    id: 'T4', number: '12304', name: 'Mountain Express', section: 'STN_E',
                    speed: 100, statusType: 'running', status: 'En route', waitingForBlock: false,
                    route: ['STN_E', 'BLOCK_E1', 'STN_E_NORTH']
                },
                {
                    id: 'T5', number: '12305', name: 'Eastern Shuttle', section: 'STN_C_EXT',
                    speed: 90, statusType: 'running', status: 'En route', waitingForBlock: false,
                    route: ['STN_C_EXT', 'BLOCK_EAST_1', 'STN_EAST']
                }
            ]);

            // Occupy some sections for demonstration
            setBlockOccupancy(prev => ({
                ...prev,
                'BLOCK_A1': 'T1',
                'BLOCK_D1': 'T2',
                'BLOCK_F3': 'T3',
                'BLOCK_E1': 'T4',
                'BLOCK_EAST_1': 'T5'
            }));

            setStationPlatforms(prev => ({
                ...prev,
                'STN_A': { ...prev['STN_A'], 1: 'T1' },
                'STN_D': { ...prev['STN_D'], 1: 'T2' },
                'STN_F': { ...prev['STN_F'], 1: 'T3' },
                'STN_E': { ...prev['STN_E'], 1: 'T4' },
                'STN_C_EXT': { ...prev['STN_C_EXT'], 1: 'T5' }
            }));

            // Set some signals to different states
            setSignals(prev => ({
                ...prev,
                'SIGNAL_A1': 'GREEN',
                'SIGNAL_B1': 'YELLOW',
                'SIGNAL_C1': 'RED',
                'SIGNAL_D1': 'GREEN',
                'SIGNAL_F1': 'GREEN',
                'SIGNAL_JCN_1': 'YELLOW'
            }));
        }, 1000);
    }, []);

    // Clock update
    useEffect(() => {
        const clock = setInterval(() => setCurrentTime(new Date()), 1000);
        return () => clearInterval(clock);
    }, []);

    // Utility functions
    const getSectionState = (sectionId) => {
        const section = TRACK_SECTIONS.find(s => s.id === sectionId);
        if (!section) return 'free';
        
        if (section.type === 'block' || section.type === 'junction' || section.type === 'crossover') {
            return blockOccupancy[sectionId] ? 'occupied' : 'free';
        }
        
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

    const freeBlocksCount = () => {
        let totalSlots = 0;
        let occupiedSlots = 0;
        
        // Count blocks
        totalSlots += Object.keys(blockOccupancy).length;
        occupiedSlots += Object.keys(blockOccupancy).filter(id => blockOccupancy[id] !== null).length;
        
        // Count station platforms
        Object.values(stationPlatforms).forEach(platformMap => {
            totalSlots += Object.keys(platformMap).length;
            occupiedSlots += Object.values(platformMap).filter(occupant => occupant !== null).length;
        });
        
        return totalSlots - occupiedSlots;
    };

    return (
        <div style={{ 
            width: '100vw', 
            height: '100vh', 
            background: 'linear-gradient(135deg, #1a2332 0%, #2a3441 100%)',
            fontFamily: 'JetBrains Mono, monospace',
            overflow: 'hidden',
            position: 'relative'
        }} onMouseMove={handleMouseMove}>
            {/* Header */}
            <div style={{
                position: 'absolute',
                top: '15px',
                left: '20px',
                right: '20px',
                height: '80px',
                background: 'rgba(20, 30, 45, 0.95)',
                border: '1px solid #3a4a5a',
                borderRadius: '6px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '0 25px',
                boxShadow: 'inset 0 2px 6px rgba(0,0,0,0.3)'
            }}>
                <div>
                    <div style={{ color: '#ff6b6b', fontSize: '18px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '1.2px' }}>
                        EXPANDED RAILWAY CONTROL SYSTEM
                    </div>
                    <div style={{ color: '#9aa5b1', fontSize: '12px', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.8px' }}>
                        INTELLIGENT SIGNALING & TRAFFIC MANAGEMENT
                    </div>
                </div>
                
                <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
                        <div style={{ 
                            background: '#000', color: '#44ff44', padding: '6px 12px', border: '2px solid #333',
                            borderRadius: '4px', fontSize: '20px', fontWeight: 700, textAlign: 'center', minWidth: '70px',
                            textShadow: '0 0 10px #44ff44', boxShadow: 'inset 0 0 10px rgba(68, 255, 68, 0.2)'
                        }}>
                            {freeBlocksCount()}
                        </div>
                        <div style={{ color: '#9aa5b1', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                            FREE BLOCKS
                        </div>
                    </div>
                    
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
                        <div style={{ 
                            background: '#000', color: '#4488ff', padding: '6px 12px', border: '2px solid #333',
                            borderRadius: '4px', fontSize: '20px', fontWeight: 700, textAlign: 'center', minWidth: '70px',
                            textShadow: '0 0 10px #4488ff', boxShadow: 'inset 0 0 10px rgba(68, 136, 255, 0.2)'
                        }}>
                            {String(trains.filter(t => t.statusType === 'running').length).padStart(2, '0')}
                        </div>
                        <div style={{ color: '#9aa5b1', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                            ACTIVE
                        </div>
                    </div>
                </div>
                
                <div style={{
                    background: '#000', color: '#44ff44', padding: '8px 16px', border: '2px solid #333',
                    borderRadius: '4px', fontSize: '16px', fontWeight: 700, textShadow: '0 0 8px #44ff44',
                    boxShadow: 'inset 0 0 8px rgba(68, 255, 68, 0.2)', textAlign: 'center', minWidth: '90px'
                }}>
                    {currentTime.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                </div>
            </div>

            {/* Main Track Display */}
            <div style={{
                position: 'absolute',
                top: '110px',
                left: '20px',
                right: '20px',
                bottom: '20px',
                background: 'rgba(20, 30, 45, 0.95)',
                border: '1px solid #3a4a5a',
                borderRadius: '6px',
                overflow: 'hidden',
                boxShadow: 'inset 0 2px 6px rgba(0,0,0,0.3)'
            }}>
                <svg viewBox="0 0 1000 350" style={{ width: '100%', height: '100%' }}>
                    {/* Connection Lines */}
                    {CONNECTIONS.map((conn, index) => (
                        <path key={index} d={conn.path} style={{
                            stroke: conn.type === 'main' ? '#4a7c4a' : conn.type === 'branch' ? '#7c4a4a' : '#4a6a7c',
                            strokeWidth: '4',
                            fill: 'none',
                            filter: 'drop-shadow(0 0 3px currentColor)'
                        }} />
                    ))}

                    {/* Track Sections */}
                    {TRACK_SECTIONS.map(section => {
                        const state = getSectionState(section.id);
                        const trainsInSection = getTrainsInSection(section.id);
                        const isSelected = selectedTrain && trainsInSection.some(t => t.id === selectedTrain.id);
                        
                        let sectionColor = '#2d5a2d';
                        let strokeColor = '#4a7c4a';
                        
                        if (section.type === 'station') {
                            sectionColor = '#2d4a7c';
                            strokeColor = '#4a6a9c';
                        } else if (section.type === 'junction') {
                            sectionColor = '#5a2d5a';
                            strokeColor = '#7c4a7c';
                        } else if (section.type === 'crossover') {
                            sectionColor = '#5a5a2d';
                            strokeColor = '#7c7c4a';
                        } else if (section.type === 'signal') {
                            const signalState = signals[section.id] || 'GREEN';
                            sectionColor = signalState === 'GREEN' ? '#2d5a2d' : signalState === 'YELLOW' ? '#5a5a2d' : '#5a2d2d';
                            strokeColor = signalState === 'GREEN' ? '#4aff4a' : signalState === 'YELLOW' ? '#ffff4a' : '#ff4a4a';
                        }

                        if (state === 'occupied') {
                            sectionColor = '#cc3333';
                            strokeColor = '#ff4444';
                        } else if (state === 'partial') {
                            sectionColor = '#cc8833';
                            strokeColor = '#ffaa44';
                        }

                        if (isSelected) {
                            sectionColor = '#cc9933';
                            strokeColor = '#ffdd44';
                        }

                        return (
                            <g key={section.id}>
                                <rect 
                                    x={section.x} 
                                    y={section.y} 
                                    width={section.width} 
                                    height={section.height}
                                    fill={sectionColor}
                                    stroke={strokeColor}
                                    strokeWidth="2"
                                    rx="4"
                                    style={{
                                        transition: 'all 0.3s ease',
                                        filter: state === 'occupied' ? 'drop-shadow(0 0 12px #cc3333)' : 
                                               state === 'partial' ? 'drop-shadow(0 0 8px #cc8833)' :
                                               isSelected ? 'drop-shadow(0 0 16px #cc9933)' : 'none',
                                        opacity: state === 'free' ? 0.7 : 1
                                    }}
                                />
                                
                                {/* Section Labels */}
                                <text 
                                    x={section.x + section.width / 2} 
                                    y={section.y - 8} 
                                    fill="#00ff00" 
                                    fontSize="10" 
                                    fontWeight="700" 
                                    textAnchor="middle" 
                                    fontFamily="JetBrains Mono, monospace"
                                    style={{ textShadow: '1px 1px 2px rgba(0,0,0,0.8)' }}
                                >
                                    {section.id}
                                </text>

                                {/* Station-specific elements */}
                                {section.type === 'station' && (
                                    <>
                                        <text 
                                            x={section.x + section.width / 2} 
                                            y={section.y + 25} 
                                            fill="#ffffff" 
                                            fontSize="11" 
                                            fontWeight="700" 
                                            textAnchor="middle" 
                                            fontFamily="JetBrains Mono, monospace"
                                        >
                                            {section.name}
                                        </text>
                                        <text 
                                            x={section.x + section.width / 2} 
                                            y={section.y + 38} 
                                            fill="#ffff00" 
                                            fontSize="9" 
                                            fontWeight="700" 
                                            textAnchor="middle" 
                                            fontFamily="JetBrains Mono, monospace"
                                        >
                                            {section.platforms}P
                                        </text>
                                        
                                        {/* Platform indicators */}
                                        <g>
                                            {Object.entries(stationPlatforms[section.id] || {}).map(([platformNum, occupant], idx) => (
                                                <g key={platformNum}>
                                                    <circle 
                                                        cx={section.x + 15 + (idx * 15)} 
                                                        cy={section.y + 50} 
                                                        r="5" 
                                                        fill={occupant ? '#ff4444' : '#44ff44'}
                                                        style={{ 
                                                            filter: occupant ? 'drop-shadow(0 0 6px #ff4444)' : 'drop-shadow(0 0 4px #44ff44)'
                                                        }}
                                                    />
                                                    <text 
                                                        x={section.x + 15 + (idx * 15)} 
                                                        y={section.y + 54} 
                                                        fill="#000000" 
                                                        fontSize="6" 
                                                        textAnchor="middle" 
                                                        fontFamily="JetBrains Mono, monospace"
                                                    >
                                                        {platformNum}
                                                    </text>
                                                </g>
                                            ))}
                                        </g>
                                    </>
                                )}

                                {/* Signal-specific elements */}
                                {section.type === 'signal' && (
                                    <circle 
                                        cx={section.x + section.width / 2} 
                                        cy={section.y + section.height / 2} 
                                        r="6" 
                                        fill={signals[section.id] === 'GREEN' ? '#44ff44' : 
                                              signals[section.id] === 'YELLOW' ? '#ffff44' : '#ff4444'}
                                        stroke="#ffffff"
                                        strokeWidth="1"
                                        style={{
                                            filter: `drop-shadow(0 0 8px ${signals[section.id] === 'GREEN' ? '#44ff44' : 
                                                                          signals[section.id] === 'YELLOW' ? '#ffff44' : '#ff4444'})`
                                        }}
                                    />
                                )}

                                {/* Junction indicators */}
                                {section.type === 'junction' && (
                                    <polygon 
                                        points={`${section.x + section.width/2},${section.y - 2} ${section.x + section.width - 2},${section.y + section.height/2} ${section.x + section.width/2},${section.y + section.height + 2} ${section.x + 2},${section.y + section.height/2}`}
                                        fill="#ffaa44"
                                        stroke="#ffdd44"
                                        strokeWidth="1"
                                        style={{ filter: 'drop-shadow(0 0 4px #ffaa44)' }}
                                    />
                                )}

                                {/* Crossover indicators */}
                                {section.type === 'crossover' && (
                                    <>
                                        <line 
                                            x1={section.x + 5} 
                                            y1={section.y + 2} 
                                            x2={section.x + section.width - 5} 
                                            y2={section.y + section.height - 2} 
                                            stroke="#ffaa44" 
                                            strokeWidth="2"
                                        />
                                        <line 
                                            x1={section.x + 5} 
                                            y1={section.y + section.height - 2} 
                                            x2={section.x + section.width - 5} 
                                            y2={section.y + 2} 
                                            stroke="#ffaa44" 
                                            strokeWidth="2"
                                        />
                                    </>
                                )}

                                {/* Trains in this section */}
                                {trainsInSection.map((train, trainIndex) => {
                                    const center = getSectionCenter(section);
                                    let offsetY = 0, offsetX = 0;
                                    
                                    if (section.type === 'station') {
                                        offsetY = (trainIndex * 18) - ((trainsInSection.length - 1) * 9);
                                        offsetX = (trainIndex * 10) - ((trainsInSection.length - 1) * 5);
                                    } else {
                                        offsetX = (trainIndex * 15) - ((trainsInSection.length - 1) * 7.5);
                                    }
                                    
                                    const isTrainSelected = selectedTrain?.id === train.id;
                                    
                                    return (
                                        <g 
                                            key={train.id} 
                                            style={{ cursor: 'pointer' }}
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
                                                fill={train.statusType === 'running' ? '#00aa00' : 
                                                      train.statusType === 'completed' ? '#0066cc' : '#666666'}
                                                stroke={isTrainSelected ? '#ffdd44' : '#ffffff'}
                                                strokeWidth={isTrainSelected ? "3" : "2"}
                                                style={{
                                                    filter: isTrainSelected ? 'drop-shadow(0 0 16px #ffaa00)' :
                                                           train.statusType === 'running' ? 'drop-shadow(0 0 8px #00aa00)' :
                                                           train.statusType === 'completed' ? 'drop-shadow(0 0 6px #0066cc)' : 'none',
                                                    transition: 'all 0.3s ease'
                                                }}
                                            />
                                            <text 
                                                x={center.x + offsetX} 
                                                y={center.y + offsetY + 3} 
                                                fill="#ffffff" 
                                                fontSize="10" 
                                                fontWeight="700" 
                                                textAnchor="middle" 
                                                fontFamily="JetBrains Mono, monospace"
                                                style={{ textShadow: '1px 1px 2px rgba(0,0,0,0.8)', pointerEvents: 'none' }}
                                            >
                                                {train.number}
                                            </text>
                                            
                                            {/* Waiting indicator */}
                                            {train.waitingForBlock && (
                                                <circle 
                                                    cx={center.x + 25 + offsetX} 
                                                    cy={center.y - 5 + offsetY} 
                                                    r="4" 
                                                    fill="#ffaa00"
                                                    stroke="#ffdd44"
                                                    strokeWidth="1"
                                                    style={{ 
                                                        opacity: 0.9,
                                                        animation: 'blink 1s ease-in-out infinite'
                                                    }}
                                                />
                                            )}
                                        </g>
                                    );
                                })}
                            </g>
                        );
                    })}
                </svg>

                {/* Control Panel */}
                <div style={{
                    position: 'absolute',
                    bottom: '20px',
                    left: '20px',
                    right: '20px',
                    height: '120px',
                    background: 'rgba(20, 30, 45, 0.95)',
                    border: '1px solid #3a4a5a',
                    borderRadius: '6px',
                    padding: '15px',
                    display: 'flex',
                    gap: '20px'
                }}>
                    {/* Train List */}
                    <div style={{ flex: 1 }}>
                        <div style={{ color: '#ff6b6b', fontSize: '12px', fontWeight: 700, marginBottom: '10px' }}>
                            ACTIVE TRAINS ({trains.length})
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '8px', maxHeight: '80px', overflowY: 'auto' }}>
                            {trains.map(train => {
                                const isSelected = selectedTrain?.id === train.id;
                                return (
                                    <div 
                                        key={train.id} 
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '8px',
                                            padding: '6px 10px',
                                            background: isSelected ? 'rgba(255, 170, 0, 0.2)' : 'rgba(255, 255, 255, 0.05)',
                                            border: `1px solid ${isSelected ? '#ffaa00' : 'rgba(255, 255, 255, 0.1)'}`,
                                            borderRadius: '4px',
                                            cursor: 'pointer',
                                            fontSize: '10px',
                                            transition: 'all 0.3s ease'
                                        }}
                                        onClick={() => setSelectedTrain(isSelected ? null : train)}
                                    >
                                        <div style={{
                                            width: '8px',
                                            height: '8px',
                                            borderRadius: '50%',
                                            background: train.statusType === 'running' ? '#44ff44' : 
                                                       train.statusType === 'completed' ? '#4488ff' : '#666666',
                                            boxShadow: '0 0 4px currentColor'
                                        }} />
                                        <div style={{ flex: 1, color: '#ffffff' }}>
                                            <div style={{ fontWeight: 700 }}>{train.name}</div>
                                            <div style={{ fontSize: '9px', color: '#9aa5b1' }}>
                                                {train.number} | {train.speed} km/h | {TRACK_SECTIONS.find(s => s.id === train.section)?.name || train.section}
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* Signal Status */}
                    <div style={{ width: '200px' }}>
                        <div style={{ color: '#ff6b6b', fontSize: '12px', fontWeight: 700, marginBottom: '10px' }}>
                            SIGNAL STATUS
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px', maxHeight: '80px', overflowY: 'auto' }}>
                            {Object.entries(signals).map(([signalId, state]) => (
                                <div key={signalId} style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '6px',
                                    padding: '4px 6px',
                                    background: 'rgba(255, 255, 255, 0.05)',
                                    borderRadius: '3px',
                                    fontSize: '9px'
                                }}>
                                    <div style={{
                                        width: '8px',
                                        height: '8px',
                                        borderRadius: '50%',
                                        background: state === 'GREEN' ? '#44ff44' : state === 'YELLOW' ? '#ffff44' : '#ff4444',
                                        boxShadow: `0 0 4px ${state === 'GREEN' ? '#44ff44' : state === 'YELLOW' ? '#ffff44' : '#ff4444'}`
                                    }} />
                                    <span style={{ color: '#ffffff' }}>{signalId.replace('SIGNAL_', '')}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Network Stats */}
                    <div style={{ width: '160px' }}>
                        <div style={{ color: '#ff6b6b', fontSize: '12px', fontWeight: 700, marginBottom: '10px' }}>
                            NETWORK STATS
                        </div>
                        <div style={{ fontSize: '10px', color: '#9aa5b1', lineHeight: '1.4' }}>
                            <div>Stations: {TRACK_SECTIONS.filter(s => s.type === 'station').length}</div>
                            <div>Blocks: {TRACK_SECTIONS.filter(s => s.type === 'block').length}</div>
                            <div>Junctions: {TRACK_SECTIONS.filter(s => s.type === 'junction').length}</div>
                            <div>Crossovers: {TRACK_SECTIONS.filter(s => s.type === 'crossover').length}</div>
                            <div>Signals: {TRACK_SECTIONS.filter(s => s.type === 'signal').length}</div>
                            <div style={{ color: '#44ff44', marginTop: '5px' }}>
                                Free Capacity: {Math.round((freeBlocksCount() / (Object.keys(blockOccupancy).length + Object.values(stationPlatforms).reduce((acc, platforms) => acc + Object.keys(platforms).length, 0))) * 100)}%
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Train Tooltip */}
            {hoveredTrain && (
                <div style={{
                    position: 'fixed',
                    left: Math.min(mousePos.x + 20, window.innerWidth - 350),
                    top: Math.max(mousePos.y - 180, 10),
                    background: 'linear-gradient(135deg, rgba(20, 30, 45, 0.98) 0%, rgba(30, 40, 55, 0.98) 100%)',
                    border: '2px solid #4a5a6a',
                    borderRadius: '8px',
                    padding: '15px',
                    minWidth: '300px',
                    zIndex: 1000,
                    pointerEvents: 'none',
                    boxShadow: '0 12px 40px rgba(0, 0, 0, 0.6), inset 0 2px 6px rgba(255, 255, 255, 0.1)',
                    backdropFilter: 'blur(15px)',
                    fontFamily: 'JetBrains Mono, monospace'
                }}>
                    <div style={{ 
                        color: '#ff6b6b', fontSize: '14px', fontWeight: 700, marginBottom: '12px',
                        textTransform: 'uppercase', letterSpacing: '1px', textShadow: '0 0 8px rgba(255, 107, 107, 0.5)',
                        borderBottom: '1px solid rgba(255, 107, 107, 0.3)', paddingBottom: '6px', textAlign: 'center'
                    }}>
                        {hoveredTrain.name}
                    </div>
                    <div style={{ color: '#ffffff', fontSize: '11px', lineHeight: '1.5' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                            <span style={{ color: '#9aa5b1' }}>Train Number:</span>
                            <span style={{ fontWeight: 700 }}>{hoveredTrain.number}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                            <span style={{ color: '#9aa5b1' }}>Current Speed:</span>
                            <span style={{ color: '#44ff44', textShadow: '0 0 6px rgba(68, 255, 68, 0.4)', fontWeight: 700 }}>{hoveredTrain.speed} km/h</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                            <span style={{ color: '#9aa5b1' }}>Current Location:</span>
                            <span style={{ color: '#ffdd44', textShadow: '0 0 6px rgba(255, 221, 68, 0.4)', fontWeight: 700 }}>
                                {TRACK_SECTIONS.find(s => s.id === hoveredTrain.section)?.name || hoveredTrain.section}
                            </span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                            <span style={{ color: '#9aa5b1' }}>Status:</span>
                            <span style={{ 
                                color: hoveredTrain.statusType === 'running' ? '#44ff44' : 
                                       hoveredTrain.statusType === 'completed' ? '#4488ff' : '#cccccc',
                                textShadow: hoveredTrain.statusType === 'running' ? '0 0 6px rgba(68, 255, 68, 0.4)' :
                                           hoveredTrain.statusType === 'completed' ? '0 0 6px rgba(68, 136, 255, 0.4)' : 'none',
                                fontWeight: 700
                            }}>
                                {hoveredTrain.status}
                            </span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ color: '#9aa5b1' }}>Route Progress:</span>
                            <span style={{ fontWeight: 700 }}>
                                {hoveredTrain.route ? `${hoveredTrain.route.indexOf(hoveredTrain.section) + 1} of ${hoveredTrain.route.length}` : 'N/A'}
                            </span>
                        </div>
                    </div>
                </div>
            )}

            {/* CSS Animation */}
            <style jsx>{`
                @keyframes blink {
                    0%, 100% { opacity: 0.5; }
                    50% { opacity: 1; }
                }
            `}</style>
        </div>
    );
};

export default TrainTrafficControl;