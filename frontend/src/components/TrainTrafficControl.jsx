// src/components/TrainTrafficControl.jsx
import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Zap, Train, Clock, MapPin, AlertTriangle, TrendingUp, Settings } from 'lucide-react';

/**
 TrainTrafficControl.jsx
 - Simulates trains starting at a common junction and moving down parallel tracks.
 - Realistic movement: speed (km/h) -> km moved per tick using tickSeconds.
 - Simple conflict detection + local reroute/hold logic (decision-support).
 - KPIs: throughput, punctuality, avgDelay, conflicts resolved, efficiency.
 - Meant as a demo/starter; plug in optimization/AI module for production decisions.
*/

const TrainTrafficControl = () => {
  const canvasRef = useRef(null);

  // sections = logical segments; distance in km
  const sections = [
    { id: 'Delhi-Mumbai', name: 'Delhi - Mumbai Central', distanceKm: 1384, maxSpeed: 130 },
    { id: 'Chennai-Bangalore', name: 'Chennai - Bangalore', distanceKm: 362, maxSpeed: 110 },
    { id: 'Mumbai-Pune', name: 'Mumbai - Pune', distanceKm: 192, maxSpeed: 100 }
  ];

  // simulation/control state
  const [isRunning, setIsRunning] = useState(false);
  const [simulationSpeed, setSimulationSpeed] = useState(1); // multiplier (1x)
  const [tickSec, setTickSec] = useState(0.5); // seconds per tick (real time interval)
  const [selectedSectionId, setSelectedSectionId] = useState(sections[0].id);

  // tracks count (parallel ways)
  const trackCount = 3;

  // KPI state
  const [kpis, setKpis] = useState({
    throughput: 0,
    punctualityPct: 100,
    avgDelayMin: 0,
    conflictsResolved: 0,
    efficiencyPct: 100
  });

  // internal simulation state
  const [trains, setTrains] = useState([]);
  const statsRef = useRef({ passed: 0, delays: [], conflictsResolved: 0 });

  // realistic initial trains but starting at junction position = 0
  const initialTrains = [
    { id: 'T12001', name: 'Shatabdi', type: 'Express', priority: 1, positionKm: 0, speedKmph: 85, plannedSpeedKmph: 85, destination: 'End', status: 'Waiting', color: '#FF6B6B' },
    { id: 'T12951', name: 'Mumbai Rajdhani', type: 'Rajdhani', priority: 1, positionKm: 0, speedKmph: 95, plannedSpeedKmph: 95, destination: 'End', status: 'Waiting', color: '#4ECDC4' },
    { id: 'T19019', name: 'Dehradun Express', type: 'Express', priority: 2, positionKm: 0, speedKmph: 70, plannedSpeedKmph: 70, destination: 'End', status: 'Waiting', color: '#45B7D1' },
    { id: 'F40251', name: 'Freight 251', type: 'Freight', priority: 3, positionKm: 0, speedKmph: 45, plannedSpeedKmph: 45, destination: 'Yard', status: 'Waiting', color: '#96CEB4' },
    { id: 'L15713', name: 'Local', type: 'Local', priority: 2, positionKm: 0, speedKmph: 55, plannedSpeedKmph: 55, destination: 'Suburb', status: 'Waiting', color: '#FECA57' }
  ];

  // On mount: initialize trains with track assignment and schedule expectation
  useEffect(() => {
    const section = sections.find(s => s.id === selectedSectionId);
    const seeded = initialTrains.map((t, idx) => {
      const track = idx % trackCount; // initial distribution
      // expected time (hours) to cross the section at plannedSpeed
      const expectedHours = Math.max(0.001, section.distanceKm / t.plannedSpeedKmph);
      const expectedSeconds = expectedHours * 3600;
      return {
        ...t,
        startedAt: null,            // timestamp when it actually begins moving
        track,
        lastTick: null,            // last update time
        expectedSeconds,
        actualSecondsUsed: 0,
        delaySec: 0,
        finished: false
      };
    });
    setTrains(seeded);
    // reset stats
    statsRef.current = { passed: 0, delays: [], conflictsResolved: 0 };
    setKpis(prev => ({ ...prev, throughput: 0, conflictsResolved: 0, avgDelayMin: 0, punctualityPct: 100 }));
  }, [selectedSectionId]);

  // Simulation loop
  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(() => {
      stepSimulation();
    }, tickSec * 1000 / simulationSpeed); // if simulationSpeed > 1, ticks run faster
    return () => clearInterval(interval);
  }, [isRunning, simulationSpeed, trains, tickSec]);

  // One simulation step: move trains, detect conflicts, update KPIs
  const stepSimulation = () => {
    const section = sections.find(s => s.id === selectedSectionId);
    const safetyDistanceKm = 0.2; // minimum separation on same track (200m)
    const newTrains = [...trains];
    const now = Date.now();

    // move each train if not finished
    newTrains.forEach((train, idx) => {
      if (train.finished) return;

      // if train hasn't started yet, consider it now started on first movement
      if (!train.startedAt) train.startedAt = now;

      // compute distance delta for this tick (km)
      const deltaKm = (train.speedKmph * (tickSec / 3600)) * simulationSpeed;
      train.positionKm += deltaKm;
      train.actualSecondsUsed += tickSec / simulationSpeed; // scaled by sim speed so KPI sensible
      train.lastTick = now;

      // update provisional delay (real vs expected)
      train.delaySec = train.actualSecondsUsed - train.expectedSeconds;

      // status update
      train.status = train.delaySec > 300 ? 'Delayed' : train.delaySec < -120 ? 'Ahead' : 'On Time';

      // check crossing end of section => mark finished
      if (train.positionKm >= section.distanceKm) {
        train.finished = true;
        train.positionKm = section.distanceKm;
        statsRef.current.passed += 1;
        statsRef.current.delays.push(train.delaySec / 60); // minutes
      }
    });

    // conflict detection per track: if two trains same track and separation < safetyDistance => conflict
    // naive: sort by positionKm descending (lead to tail)
    let conflictsThisStep = 0;
    for (let t = 0; t < trackCount; t++) {
      const trackTrains = newTrains.filter(tr => !tr.finished && tr.track === t).sort((a, b) => b.positionKm - a.positionKm);
      for (let i = 0; i < trackTrains.length - 1; i++) {
        const lead = trackTrains[i];
        const follow = trackTrains[i + 1];
        const gap = lead.positionKm - follow.positionKm;
        if (gap < safetyDistanceKm) {
          // Conflict detected: attempt reroute lower-priority (follow) to a free track
          let rerouted = false;
          for (let alt = 0; alt < trackCount; alt++) {
            if (alt === follow.track) continue;
            const altBusy = newTrains.some(tr => !tr.finished && tr.track === alt && Math.abs(tr.positionKm - follow.positionKm) < safetyDistanceKm);
            if (!altBusy) {
              // move follow to alt track
              follow.track = alt;
              follow.status = 'Rerouted';
              rerouted = true;
              statsRef.current.conflictsResolved += 1;
              conflictsThisStep += 1;
              break;
            }
          }
          if (!rerouted) {
            // hold the following train (stop it) to restore safety buffer
            follow.speedKmph = 0;
            follow.status = 'Holding';
            statsRef.current.conflictsResolved += 1;
            conflictsThisStep += 1;
          }
        } else {
          // if separation ok and train was holding, restore speed to planned
          if (follow.speedKmph === 0 && follow.status === 'Holding') {
            follow.speedKmph = follow.plannedSpeedKmph;
            follow.status = 'Resumed';
          }
        }
      }
    }

    // small heuristic: slow down freight trains slightly in bad weather (simulate disruption)
    if (Math.random() < 0.005) {
      const freight = newTrains.find(tr => tr.type === 'Freight' && !tr.finished);
      if (freight) {
        freight.speedKmph = Math.max(20, freight.speedKmph - 10);
        freight.status = 'Speed Restricted';
      }
    }

    // calculate KPIs
    const totalPassed = statsRef.current.passed;
    const delays = statsRef.current.delays;
    const avgDelay = delays.length ? delays.reduce((a, b) => a + b, 0) / delays.length : 0;
    const punctualCount = delays.length ? delays.filter(d => d <= 5).length : 0; // <=5 min considered punctual
    const punctualityPct = delays.length ? Math.round((punctualCount / delays.length) * 100) : 100;
    const efficiency = Math.max(40, Math.round(90 - avgDelay)); // crude composite

    // update global state
    setTrains(newTrains);
    setKpis({
      throughput: totalPassed,
      punctualityPct,
      avgDelayMin: Math.round(avgDelay * 10) / 10,
      conflictsResolved: statsRef.current.conflictsResolved,
      efficiencyPct: efficiency
    });
  };

  // reset simulation to initial conditions
  const resetSimulation = () => {
    setIsRunning(false);
    setSimulationSpeed(1);
    setTrains(prev => {
      // reinitialize trains keeping same array structure
      return initialTrains.map((t, idx) => ({
        ...t,
        positionKm: 0,
        startedAt: null,
        track: idx % trackCount,
        lastTick: null,
        expectedSeconds: Math.max(1, sections.find(s => s.id === selectedSectionId).distanceKm / t.plannedSpeedKmph) * 3600,
        actualSecondsUsed: 0,
        delaySec: 0,
        finished: false,
        speedKmph: t.plannedSpeedKmph,
        status: 'Waiting'
      }));
    });
    statsRef.current = { passed: 0, delays: [], conflictsResolved: 0 };
    setKpis({ throughput: 0, punctualityPct: 100, avgDelayMin: 0, conflictsResolved: 0, efficiencyPct: 100 });
  };

  // Canvas drawing
  useEffect(() => {
    drawCanvas();
  }, [trains, selectedSectionId]);

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const DPR = window.devicePixelRatio || 1;
    const widthCSS = canvas.clientWidth;
    const heightCSS = canvas.clientHeight;
    canvas.width = Math.floor(widthCSS * DPR);
    canvas.height = Math.floor(heightCSS * DPR);
    ctx.scale(DPR, DPR);

    // clear
    ctx.fillStyle = '#071024';
    ctx.fillRect(0, 0, widthCSS, heightCSS);

    // draw tracks
    const marginY = 30;
    const availableHeight = heightCSS - marginY * 2;
    const gap = availableHeight / (trackCount - 1 || 1);
    const section = sections.find(s => s.id === selectedSectionId);
    const pxPerKm = (widthCSS - 160) / section.distanceKm; // leave padding L/R

    for (let t = 0; t < trackCount; t++) {
      const y = marginY + t * gap;
      // track line
      ctx.strokeStyle = '#5b708a';
      ctx.lineWidth = 6;
      roundLine(ctx, 60, y, widthCSS - 60, y, 6);
      // station/junction at x=60
      ctx.fillStyle = '#cbd5e1';
      ctx.fillRect(52, y - 12, 16, 24);
      ctx.font = '12px sans-serif';
      ctx.fillStyle = '#e2e8f0';
      ctx.fillText('Junction', 20, y + 4);
      // end marker
      ctx.fillStyle = '#cbd5e1';
      ctx.fillRect(widthCSS - 68, y - 12, 16, 24);
      ctx.fillText('End', widthCSS - 40, y + 4);
    }
  const drawNetwork = (ctx, widthCSS, heightCSS) => {
  // Background
  ctx.fillStyle = '#071024';
  ctx.fillRect(0, 0, widthCSS, heightCSS);

  // Example: Two parallel mainlines
  const mainY1 = 120, mainY2 = 200;
  const startX = 80, endX = widthCSS - 100;

  // Draw main tracks
  ctx.strokeStyle = '#5b708a';
  ctx.lineWidth = 6;
  roundLine(ctx, startX, mainY1, endX, mainY1, 6);
  roundLine(ctx, startX, mainY2, endX, mainY2, 6);

  // Draw crossover (diamond)
  ctx.beginPath();
  ctx.moveTo(startX + 200, mainY1);
  ctx.lineTo(startX + 240, mainY2);
  ctx.moveTo(startX + 200, mainY2);
  ctx.lineTo(startX + 240, mainY1);
  ctx.strokeStyle = '#8b9cb7';
  ctx.lineWidth = 4;
  ctx.stroke();

  // Draw station (rectangle across both tracks)
  ctx.fillStyle = '#2f3e53';
  ctx.fillRect(startX + 400, mainY1 - 40, 140, 140);
  ctx.fillStyle = '#e2e8f0';
  ctx.font = '14px sans-serif';
  ctx.fillText('STATION A', startX + 470, mainY1 + 60);

  // Draw signals (before station)
  const sigX = startX + 350;
  drawSignal(ctx, sigX, mainY1 - 20, 'green');
  drawSignal(ctx, sigX, mainY2 - 20, 'red');
};

const drawSignal = (ctx, x, y, color) => {
  ctx.beginPath();
  ctx.arc(x, y, 8, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 1;
  ctx.stroke();
};

 // draw trains
trains.forEach((train) => {
  const trackY = marginY + train.track * gap;
  const x = 60 + Math.min(section.distanceKm, train.positionKm) * pxPerKm;

  // 🚂 Train body (locomotive + 2 carriages)
  ctx.fillStyle = train.color || '#ef4444';
  ctx.fillRect(x - 20, trackY - 12, 40, 24); // loco
  ctx.fillRect(x + 22, trackY - 10, 28, 20); // carriage 1
  ctx.fillRect(x + 52, trackY - 10, 28, 20); // carriage 2

  // outline
  ctx.strokeStyle = '#0f172a';
  ctx.lineWidth = 2;
  ctx.strokeRect(x - 20, trackY - 12, 40, 24);
  ctx.strokeRect(x + 22, trackY - 10, 28, 20);
  ctx.strokeRect(x + 52, trackY - 10, 28, 20);

  // 🚂 Wheels (2 per carriage/loco)
  ctx.fillStyle = '#222';
  const wheelY = trackY + 12;
  [x - 12, x + 12, x + 28, x + 46, x + 58, x + 76].forEach((wx) => {
    ctx.beginPath();
    ctx.arc(wx, wheelY, 4, 0, 2 * Math.PI);
    ctx.fill();
  });

  // 🔤 Train ID above
  ctx.fillStyle = '#f8fafc';
  ctx.font = '10px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(train.id, x + 30, trackY - 20);

  // Speed label
  ctx.font = '9px sans-serif';
  ctx.fillStyle = train.speedKmph > 80 ? '#ffbaba' : '#f1f5f9';
  ctx.fillText(`${train.speedKmph} km/h`, x + 30, trackY + 36);

  // Progress bar above track
  const progressW = 120;
  const pxStart = x - progressW / 2;
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(pxStart, trackY - 34, progressW, 5);
  const percent = Math.min(1, train.positionKm / section.distanceKm);
  ctx.fillStyle = train.color;
  ctx.fillRect(pxStart, trackY - 34, progressW * percent, 5);

  // Status below
  ctx.fillStyle = '#cbd5e1';
  ctx.font = '9px sans-serif';
  ctx.fillText(train.status, x + 30, trackY + 50);
});

  };

  // helper to draw rounded thick line
  const roundLine = (ctx, x1, y1, x2, y2, width) => {
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.lineWidth = width;
    ctx.stroke();
  };

  // utility for status color classes
  const statusColor = (s) => {
    if (s === 'On Time' || s === 'Resumed') return 'text-green-400';
    if (s === 'Delayed' || s === 'Holding') return 'text-red-400';
    if (s === 'Rerouted') return 'text-yellow-400';
    return 'text-slate-300';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl">
            <Train className="h-8 w-8 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">AI Train Traffic Control — Section Simulator</h1>
            <p className="text-slate-300 text-sm">Realistic movement, multi-track, conflict detection & simple reroute</p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="bg-slate-800/50 backdrop-blur rounded-xl p-2 px-3">
            <div className="text-sm text-slate-300">Section</div>
            <select value={selectedSectionId} onChange={(e) => setSelectedSectionId(e.target.value)} className="bg-transparent text-white border-none">
              {sections.map(s => <option className="text-black" key={s.id} value={s.id}>{s.name}</option>)}
            </select>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-3">
            <div className="text-xs text-slate-300">Throughput</div>
            <div className="text-white font-bold">{kpis.throughput}</div>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-3">
            <div className="text-xs text-slate-300">Punctuality</div>
            <div className="text-white font-bold">{kpis.punctualityPct}%</div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700">
          <div className="flex gap-2">
            <button onClick={() => setIsRunning(r => !r)} className={`flex-1 py-2 rounded-lg font-medium ${isRunning ? 'bg-red-500' : 'bg-green-500'} text-white`}>
              {isRunning ? <><Pause className="inline-block mr-2" /> Pause</> : <><Play className="inline-block mr-2" /> Start</>}
            </button>
            <button onClick={resetSimulation} className="px-4 py-2 bg-slate-600 rounded-lg text-white"><RotateCcw /></button>
          </div>

          <div className="mt-3">
            <label className="text-slate-300 text-sm">Sim speed: {simulationSpeed}x</label>
            <input type="range" min="0.5" max="4" step="0.5" value={simulationSpeed} onChange={(e) => setSimulationSpeed(parseFloat(e.target.value))} className="w-full mt-2" />
          </div>

          <div className="mt-3">
            <label className="text-slate-300 text-sm">Tick sec (internal)</label>
            <input type="range" min="0.1" max="1" step="0.1" value={tickSec} onChange={(e) => setTickSec(parseFloat(e.target.value))} className="w-full mt-2" />
            <div className="text-xs text-slate-400 mt-1">Faster ticks produce smoother motion</div>
          </div>
        </div>

        <div className="md:col-span-3 bg-slate-800/50 p-4 rounded-xl border border-slate-700">
          <h3 className="text-white font-semibold mb-2">AI Metrics (placeholder)</h3>
          <div className="grid grid-cols-5 gap-3 text-sm">
            <div className="text-center">
              <div className="text-white font-bold">94%</div>
              <div className="text-slate-400">Throughput Opt</div>
            </div>
            <div className="text-center">
              <div className="text-white font-bold">78%</div>
              <div className="text-slate-400">Delay Reduction</div>
            </div>
            <div className="text-center">
              <div className="text-white font-bold">{kpis.conflictsResolved}</div>
              <div className="text-slate-400">Conflicts Resolved</div>
            </div>
            <div className="text-center">
              <div className="text-white font-bold">{kpis.avgDelayMin}m</div>
              <div className="text-slate-400">Avg Delay</div>
            </div>
            <div className="text-center">
              <div className="text-white font-bold">{kpis.efficiencyPct}%</div>
              <div className="text-slate-400">Efficiency</div>
            </div>
          </div>
        </div>
      </div>

      {/* Canvas */}
      <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700 mb-6">
        <div className="mb-3 flex justify-between items-center">
          <h4 className="text-white font-semibold">Live Track Visualization</h4>
          <div className="text-sm text-slate-400">Tracks: {trackCount} | Section distance: {sections.find(s => s.id === selectedSectionId).distanceKm} km</div>
        </div>
        <div style={{ height: 320 }}>
          <canvas ref={canvasRef} style={{ width: '100%', height: '100%' }} />
        </div>
      </div>

      {/* Train Status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700">
          <h4 className="text-white font-semibold mb-3">Active Trains ({trains.filter(t => !t.finished).length})</h4>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {trains.map(tr => (
              <div key={tr.id} className="bg-slate-700/40 p-3 rounded-lg border border-slate-600">
                <div className="flex justify-between">
                  <div>
                    <div className="text-white font-semibold">{tr.name} <span className="text-sm text-slate-400">({tr.id})</span></div>
                    <div className="text-slate-400 text-sm">{tr.type} • Track {tr.track + 1}</div>
                  </div>
                  <div className={`text-sm ${statusColor(tr.status)}`}>{tr.status}</div>
                </div>

                <div className="grid grid-cols-4 gap-2 text-sm mt-2">
                  <div><div className="text-slate-400">Speed</div><div className="text-white">{tr.speedKmph} km/h</div></div>
                  <div><div className="text-slate-400">Position</div><div className="text-white">{Math.min(sections.find(s => s.id === selectedSectionId).distanceKm, tr.positionKm).toFixed(2)} km</div></div>
                  <div><div className="text-slate-400">ETA vs Plan</div><div className="text-white">{(tr.delaySec / 60).toFixed(1)} min</div></div>
                  <div><div className="text-slate-400">Progress</div><div className="text-white">{Math.min(100, (tr.positionKm / sections.find(s => s.id === selectedSectionId).distanceKm * 100)).toFixed(1)}%</div></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700">
            <h4 className="text-white font-semibold mb-2">System Alerts & Recommendations</h4>
            <div className="text-slate-300 text-sm space-y-2">
              <div>• If a conflict persists, system tries to reroute a lower priority train to a free parallel track.</div>
              <div>• If reroute not possible, the train will be held until safe separation is restored.</div>
              <div>• Replace this policy with an OR optimizer or RL policy for production-grade decisions.</div>
            </div>
          </div>

          <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700">
            <h4 className="text-white font-semibold mb-2">Quick Statistics</h4>
            <div className="grid grid-cols-2 gap-3">
              <div className="text-center">
                <div className="text-2xl text-white font-bold">{kpis.throughput}</div>
                <div className="text-slate-400 text-xs">Trains Passed</div>
              </div>
              <div className="text-center">
                <div className="text-2xl text-white font-bold">{kpis.punctualityPct}%</div>
                <div className="text-slate-400 text-xs">Punctuality</div>
              </div>
              <div className="text-center">
                <div className="text-2xl text-white font-bold">{kpis.avgDelayMin}m</div>
                <div className="text-slate-400 text-xs">Avg Delay</div>
              </div>
              <div className="text-center">
                <div className="text-2xl text-white font-bold">{kpis.conflictsResolved}</div>
                <div className="text-slate-400 text-xs">Conflicts Resolved</div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default TrainTrafficControl;
