import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, RotateCcw, Train } from 'lucide-react';

/**
 * TrainTrafficControl-Interactive.jsx
 * - Enhanced visualization + interactivity for the section simulator.
 * - Smooth animation (requestAnimationFrame), clickable trains, hover tooltips.
 * - Realistic tracks, intermediate stations, semaphore-style signals, and Indian Railways inspired palette.
 * - Controller quick actions: Hold / Resume / Prioritise / Swap track.
 *
 * Drop-in replacement for the original component. Styling uses Tailwind classes already in the project.
 */
// Utility: darken/lighten a hex color
function shadeColor(color, percent) {
  let R = parseInt(color.substring(1, 3), 16);
  let G = parseInt(color.substring(3, 5), 16);
  let B = parseInt(color.substring(5, 7), 16);

  R = parseInt(R * (100 + percent) / 100);
  G = parseInt(G * (100 + percent) / 100);
  B = parseInt(B * (100 + percent) / 100);

  R = (R < 255) ? R : 255;
  G = (G < 255) ? G : 255;
  B = (B < 255) ? B : 255;

  const RR = (R.toString(16).length === 1 ? "0" + R.toString(16) : R.toString(16));
  const GG = (G.toString(16).length === 1 ? "0" + G.toString(16) : G.toString(16));
  const BB = (B.toString(16).length === 1 ? "0" + B.toString(16) : B.toString(16));

  return "#" + RR + GG + BB;
}

// Utility: return pill background color based on status
function pillColor(status) {
  switch (status) {
    case 'Holding': return '#facc15'; // yellow
    case 'Rerouted': return '#f97316'; // orange
    case 'Arrived': return '#22c55e'; // green
    default: return '#60a5fa'; // blue for running
  }
}

const INDIAN_RAIL_BG = 'linear-gradient(135deg,#012a4a 0%,#014f86 40%,#0a9396 100%)';
const PANEL_BG = 'rgba(2,6,23,0.55)';

const TrainTrafficControl = () => {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [isRunning, setIsRunning] = useState(false);
  const [simulationSpeed, setSimulationSpeed] = useState(1);
  const [tickSec, setTickSec] = useState(0.5);
  const [selectedSectionId, setSelectedSectionId] = useState('Delhi-Mumbai');
  const trackCount = 3;

  const sections = [
    { id: 'Delhi-Mumbai', name: 'Delhi - Mumbai Central', distanceKm: 1384, maxSpeed: 130 },
    { id: 'Chennai-Bangalore', name: 'Chennai - Bangalore', distanceKm: 362, maxSpeed: 110 },
    { id: 'Mumbai-Pune', name: 'Mumbai - Pune', distanceKm: 192, maxSpeed: 100 }
  ];

  // stations along the section (sample intermediate stations)
  const stationsBySection = {
    'Delhi-Mumbai': [
      { name: 'Junction', km: 0 },
      { name: 'Station A', km: 220 },
      { name: 'Station B', km: 520 },
      { name: 'Station C', km: 900 },
      { name: 'Mumbai Central', km: 1384 }
    ],
    'Chennai-Bangalore': [
      { name: 'Junction', km: 0 }, { name: 'Katpadi', km: 120 }, { name: 'Arakkonam', km: 220 }, { name: 'Bangalore', km: 362 }
    ],
    'Mumbai-Pune': [
      { name: 'Junction', km: 0 }, { name: 'Lonavala', km: 90 }, { name: 'Pune', km: 192 }
    ]
  };
const junctionsBySection = {
  'Delhi-Mumbai': [
    { km: 520, branches: 2 }, // Y-junction at 520 km
    { km: 900, branches: 2 }
  ],
  'Chennai-Bangalore': [
    { km: 120, branches: 2 }
  ],
  'Mumbai-Pune': []
};
  // initial trains
  const baseTrains = [
    { id: 'T12001', name: 'Shatabdi', type: 'Express', priority: 1, positionKm: 0, speedKmph: 85, plannedSpeedKmph: 85, destination: 'End', color: '#FF6B6B' },
    { id: 'T12951', name: 'Mumbai Rajdhani', type: 'Rajdhani', priority: 1, positionKm: 0, speedKmph: 95, plannedSpeedKmph: 95, destination: 'End', color: '#4ECDC4' },
    { id: 'T19019', name: 'Dehradun', type: 'Express', priority: 2, positionKm: 0, speedKmph: 70, plannedSpeedKmph: 70, destination: 'End', color: '#45B7D1' },
    { id: 'F40251', name: 'Freight 251', type: 'Freight', priority: 3, positionKm: 0, speedKmph: 45, plannedSpeedKmph: 45, destination: 'Yard', color: '#96CEB4' },
    { id: 'L15713', name: 'Local', type: 'Local', priority: 2, positionKm: 0, speedKmph: 55, plannedSpeedKmph: 55, destination: 'Suburb', color: '#FECA57' }
  ];

  const [trains, setTrains] = useState([]);
  const statsRef = useRef({ passed: 0, delays: [], conflictsResolved: 0 });
  const [kpis, setKpis] = useState({ throughput: 0, punctualityPct: 100, avgDelayMin: 0, conflictsResolved: 0, efficiencyPct: 100 });
  const rafRef = useRef(null);
  const lastTimeRef = useRef(null);
  const [hoverInfo, setHoverInfo] = useState(null); // {x,y,train}
  const [selectedTrainId, setSelectedTrainId] = useState(null);

  // initialize trains for chosen section
  useEffect(() => {
    const section = sections.find(s => s.id === selectedSectionId);
    const seeded = baseTrains.map((t, idx) => {
      const track = idx % trackCount;
      const expectedHours = Math.max(0.001, section.distanceKm / t.plannedSpeedKmph);
      const expectedSeconds = expectedHours * 3600;
      return {
        ...t,
        startedAt: null,
        track,
        lastTick: null,
        expectedSeconds,
        actualSecondsUsed: 0,
        delaySec: 0,
        finished: false,
        speedKmph: t.plannedSpeedKmph,
        status: 'Waiting'
      };
    });
    setTrains(seeded);
    statsRef.current = { passed: 0, delays: [], conflictsResolved: 0 };
    setKpis(prev => ({ ...prev, throughput: 0, conflictsResolved: 0, avgDelayMin: 0, punctualityPct: 100 }));
  }, [selectedSectionId]);

  // reset simulation
  const resetSimulation = () => {
    setIsRunning(false);
    setSimulationSpeed(1);
    // re-initialize trains quickly
    const section = sections.find(s => s.id === selectedSectionId);
    setTrains(baseTrains.map((t, idx) => ({
      ...t,
      positionKm: 0,
      startedAt: null,
      track: idx % trackCount,
      lastTick: null,
      expectedSeconds: Math.max(1, section.distanceKm / t.plannedSpeedKmph) * 3600,
      actualSecondsUsed: 0,
      delaySec: 0,
      finished: false,
      speedKmph: t.plannedSpeedKmph,
      status: 'Waiting'
    })));
    statsRef.current = { passed: 0, delays: [], conflictsResolved: 0 };
    setKpis({ throughput: 0, punctualityPct: 100, avgDelayMin: 0, conflictsResolved: 0, efficiencyPct: 100 });
  };

  // core simulation tick using requestAnimationFrame for smoother visuals
  const simulate = useCallback((timestamp) => {
    if (!lastTimeRef.current) lastTimeRef.current = timestamp;
    const deltaMs = timestamp - lastTimeRef.current;
    const secondsElapsed = (deltaMs / 1000) * simulationSpeed; // scaled
    // only advance if isRunning
    if (isRunning) {
      stepBy(secondsElapsed);
    }
    lastTimeRef.current = timestamp;
    rafRef.current = requestAnimationFrame(simulate);
  }, [isRunning, simulationSpeed, trains]);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(simulate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [simulate]);

  // advance simulation by secondsElapsed seconds (simulated)
  const stepBy = (secondsElapsed) => {
    const section = sections.find(s => s.id === selectedSectionId);
    if (!section) return;
    const safetyDistanceKm = 0.2;
    const newTrains = trains.map(t => ({ ...t }));

    // move trains
    newTrains.forEach(train => {
      if (train.finished) return;
      if (!train.startedAt) train.startedAt = Date.now();

      const deltaKm = (train.speedKmph * (secondsElapsed / 3600));
      train.positionKm += deltaKm;
      train.actualSecondsUsed += secondsElapsed;
      train.lastTick = Date.now();
      train.delaySec = train.actualSecondsUsed - train.expectedSeconds;
      train.status = train.delaySec > 300 ? 'Delayed' : train.delaySec < -120 ? 'Ahead' : 'On Time';
      if (train.positionKm >= section.distanceKm) {
        train.finished = true;
        train.positionKm = section.distanceKm;
        statsRef.current.passed += 1;
        statsRef.current.delays.push(train.delaySec / 60);
      }
    });

    // conflict detection + naive reroute/hold
    let conflictsThisStep = 0;
    for (let t = 0; t < trackCount; t++) {
      const trackTrains = newTrains.filter(tr => !tr.finished && tr.track === t).sort((a, b) => b.positionKm - a.positionKm);
      for (let i = 0; i < trackTrains.length - 1; i++) {
        const lead = trackTrains[i];
        const follow = trackTrains[i + 1];
        const gap = lead.positionKm - follow.positionKm;
        if (gap < safetyDistanceKm) {
          // try to reroute follow
          let rerouted = false;
          for (let alt = 0; alt < trackCount; alt++) {
            if (alt === follow.track) continue;
            const altBusy = newTrains.some(tr => !tr.finished && tr.track === alt && Math.abs(tr.positionKm - follow.positionKm) < safetyDistanceKm);
            if (!altBusy) {
              follow.track = alt;
              follow.status = 'Rerouted';
              rerouted = true;
              statsRef.current.conflictsResolved += 1;
              conflictsThisStep += 1;
              break;
            }
          }
          if (!rerouted) {
            follow.speedKmph = 0;
            follow.status = 'Holding';
            statsRef.current.conflictsResolved += 1;
            conflictsThisStep += 1;
          }
        } else {
          if (follow.speedKmph === 0 && follow.status === 'Holding') {
            follow.speedKmph = follow.plannedSpeedKmph;
            follow.status = 'Resumed';
          }
        }
      }
    }

    // occasional speed restriction simulation
    if (Math.random() < 0.006) {
      const freight = newTrains.find(tr => tr.type === 'Freight' && !tr.finished);
      if (freight) {
        freight.speedKmph = Math.max(20, freight.speedKmph - 10);
        freight.status = 'Speed Restricted';
      }
    }

    // update KPIs
    const totalPassed = statsRef.current.passed;
    const delays = statsRef.current.delays;
    const avgDelay = delays.length ? delays.reduce((a, b) => a + b, 0) / delays.length : 0;
    const punctualCount = delays.length ? delays.filter(d => d <= 5).length : 0;
    const punctualityPct = delays.length ? Math.round((punctualCount / delays.length) * 100) : 100;
    const efficiency = Math.max(40, Math.round(90 - avgDelay));

    setTrains(newTrains);
    setKpis({ throughput: totalPassed, punctualityPct, avgDelayMin: Math.round(avgDelay * 10) / 10, conflictsResolved: statsRef.current.conflictsResolved, efficiencyPct: efficiency });
  };

  // canvas draw
  useEffect(() => {
    draw();
  }, [trains, selectedSectionId]);

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const DPR = window.devicePixelRatio || 1;
    const widthCSS = canvas.clientWidth;
    const heightCSS = canvas.clientHeight;
    canvas.width = Math.floor(widthCSS * DPR);
    canvas.height = Math.floor(heightCSS * DPR);
    ctx.scale(DPR, DPR);

    // background inspired by Indian Railways (deep indigo to teal)
    const grd = ctx.createLinearGradient(0, 0, widthCSS, heightCSS);
    grd.addColorStop(0, '#001f3f');
    grd.addColorStop(0.5, '#05386B');
    grd.addColorStop(1, '#0A6A67');
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, widthCSS, heightCSS);

    // subtle texture: repeating stripes
    for (let i = 0; i < widthCSS; i += 80) {
      ctx.fillStyle = 'rgba(255,255,255,0.02)';
      ctx.fillRect(i, 0, 40, heightCSS);
    }

    // draw realistic tracks
    const marginX = 80;
    const marginY = 60;
    const trackGap = (heightCSS - marginY * 2) / (trackCount - 1 || 1);
    const section = sections.find(s => s.id === selectedSectionId);
    const pxPerKm = (widthCSS - marginX * 2) / section.distanceKm;

    // sleepers pattern and rails
    for (let t = 0; t < trackCount; t++) {
      const y = marginY + t * trackGap;
      // sleepers
      ctx.strokeStyle = 'rgba(80,80,80,0.6)';
      for (let x = marginX; x < widthCSS - marginX; x += 24) {
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, y - 10);
        ctx.lineTo(x + 14, y + 10);
        ctx.stroke();
      }
      // rails (two rails)
      ctx.strokeStyle = '#b39c82';
      ctx.lineWidth = 6;
      roundLine(ctx, marginX, y - 8, widthCSS - marginX, y - 8, 6);
      roundLine(ctx, marginX, y + 8, widthCSS - marginX, y + 8, 6);
    }

    // draw intermediate stations and signals
    const stations = stationsBySection[selectedSectionId] || [];
    stations.forEach(st => {
      const x = marginX + Math.max(0, Math.min(section.distanceKm, st.km)) * pxPerKm;
      // platform across tracks (raised rectangle)
      ctx.fillStyle = 'rgba(18,18,18,0.6)';
      ctx.fillRect(x - 28, marginY - 40, 56, (trackGap * (trackCount - 1)) + 80);

      // station name
      ctx.fillStyle = '#f8fafc';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(st.name, x, marginY - 46);

      // signals before station for each track
      for (let t = 0; t < trackCount; t++) {
        const sy = marginY + t * trackGap - 18;
        const sigX = x - 60;
        const signalState = getSignalStateForLocation(st.km, t); // green/red amber
        drawSemaphoreSignal(ctx, sigX, sy, signalState);
      }
    });
  // draw junctions
const junctions = junctionsBySection[selectedSectionId] || [];
junctions.forEach(j => {
  const x = marginX + j.km * pxPerKm;
  for (let t = 0; t < j.branches; t++) {
    const y = marginY + t * trackGap;
    ctx.strokeStyle = t === 0 ? '#FFD700' : '#aaa'; // main vs branch
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 40, y - (t*15)); // branching angle
    ctx.stroke();

    // signal at branch
    drawSemaphoreSignal(ctx, x + 10, y - 10, getSignalStateForLocation(j.km, t));
  }
});
    // draw trains (improved visuals: SVG-like shapes, shadows, heading)
trains.forEach(train => {
  const trackY = marginY + train.track * trackGap;
  const x = marginX + Math.min(section.distanceKm, train.positionKm) * pxPerKm;

  // draw shadow
  ctx.fillStyle = 'rgba(0,0,0,0.35)';
  ctx.beginPath();
  ctx.ellipse(x + 32, trackY + 18, 50, 12, 0, 0, Math.PI * 2);
  ctx.fill();

  // locomotive (rounded, with gradient)
  const locoW = 48, locoH = 28;
  const locoGrad = ctx.createLinearGradient(x - locoW / 2, trackY - locoH / 2, x + locoW / 2, trackY + locoH / 2);
  locoGrad.addColorStop(0, shadeColor(train.color, 20));
  locoGrad.addColorStop(1, train.color);
  roundedRect(ctx, x - locoW / 2, trackY - locoH / 2, locoW, locoH, 6);
  ctx.fillStyle = locoGrad;
  ctx.fill();
  ctx.strokeStyle = '#071024';
  ctx.lineWidth = 2;
  ctx.stroke();

      // headlight
      ctx.beginPath();
      ctx.arc(x + locoW / 2, trackY, 6, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,200,0.7)';
      ctx.fill();

      // pantograph (for electric trains)
      if (train.type === 'Rajdhani' || train.type === 'Express') {
        ctx.strokeStyle = '#aaa';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, trackY - locoH / 2);
        ctx.lineTo(x, trackY - locoH / 2 - 12);
        ctx.lineTo(x + 12, trackY - locoH / 2 - 8);
        ctx.stroke();
      }

      // carriages behind loco
      const carriageW = 38, carriageH = 22;
      for (let c = 0; c < 2; c++) {
        const cx = x + 30 + c * (carriageW + 6);
        roundedRect(ctx, cx - carriageW / 2, trackY - carriageH / 2, carriageW, carriageH, 4);
        // carriage color by type
        let carriageColor = train.type === 'Freight' ? '#888' : shadeColor(train.color, -8 - c * 6);
        ctx.fillStyle = carriageColor;
        ctx.fill();
        ctx.strokeStyle = '#071024';
        ctx.stroke();

        // windows
        ctx.fillStyle = '#e0e7ef';
        for (let w = 0; w < 3; w++) {
          ctx.fillRect(cx - carriageW / 2 + 6 + w * 10, trackY - 6, 8, 8);
        }

        // doors
        ctx.fillStyle = '#bfc9d1';
        ctx.fillRect(cx + carriageW / 2 - 14, trackY - 6, 6, 12);

        // wheels
        ctx.beginPath();
        ctx.arc(cx - 10, trackY + carriageH / 2, 4, 0, Math.PI * 2);
        ctx.arc(cx + 10, trackY + carriageH / 2, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#222';
        ctx.fill();
      }

      // smoke for diesel trains
      if (train.type === 'Freight' || train.type === 'Express') {
        ctx.beginPath();
        ctx.arc(x - locoW / 2, trackY - locoH / 2 - 8, 8, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(180,180,180,0.3)';
        ctx.fill();
      }

      // train ID and small status pill
      ctx.fillStyle = '#fff';
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(train.id, x + 28, trackY - 20);

      // status pill
      ctx.fillStyle = pillColor(train.status);
      roundedRect(ctx, x - 24, trackY + 18, 80, 18, 10);
      ctx.fillStyle = '#071024';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`${train.status} • ${Math.round(Math.min(section.distanceKm, train.positionKm))} km`, x + 16, trackY + 32);

      // invisible hit target (store for interactions)
      train._hitbox = { x: x - 40, y: trackY - 30, w: 160, h: 80 };
      if (selectedTrainId === train.id) {
  ctx.strokeStyle = '#FFD700'; // golden border
  ctx.lineWidth = 3;
  roundedRect(ctx, train._hitbox.x - 4, train._hitbox.y - 4, train._hitbox.w + 8, train._hitbox.h + 8, 12);
  ctx.stroke();
}
    });
  };

  const roundLine = (ctx, x1, y1, x2, y2, width) => {
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.lineWidth = width;
    ctx.stroke();
  };

  const roundedRect = (ctx, x, y, w, h, r) => {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  };

  const drawSemaphoreSignal = (ctx, x, y, state = 'green') => {
    // mast
    ctx.fillStyle = '#2d3748';
    ctx.fillRect(x - 2, y - 18, 4, 36);
    // head box
    roundedRect(ctx, x - 10, y - 26, 20, 20, 4);
    ctx.fillStyle = '#0f172a';
    ctx.fill();
    // lights
    const colors = { green: '#16a34a', red: '#ef4444', amber: '#f59e0b' };
    ctx.beginPath(); ctx.arc(x, y - 16, 4, 0, Math.PI * 2); ctx.fillStyle = state === 'green' ? colors.green : 'rgba(100,100,100,0.25)'; ctx.fill();
    ctx.beginPath(); ctx.arc(x, y - 8, 4, 0, Math.PI * 2); ctx.fillStyle = state === 'amber' ? colors.amber : 'rgba(100,100,100,0.25)'; ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fillStyle = state === 'red' ? colors.red : 'rgba(100,100,100,0.25)'; ctx.fill();
  };

  const getSignalStateForLocation = (km, track) => {
    // simplistic rule: alternate states to make visualization interesting
    const seed = Math.floor(km / 100) + track;
    if (seed % 3 === 0) return 'green';
    if (seed % 3 === 1) return 'amber';
    return 'red';
  }

  // interactivity: mouse move / click on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      // find hovered train
      const found = trains.find(tr => {
        if (!tr._hitbox) return false;
        const hb = tr._hitbox;
        return x >= hb.x && x <= hb.x + hb.w && y >= hb.y && y <= hb.y + hb.h;
      });
      if (found) {
        setHoverInfo({ x: e.clientX - rect.left + 12, y: e.clientY - rect.top + 12, train: found });
      } else setHoverInfo(null);
    };

    const handleClick = (e) => {
      if (!hoverInfo) return;
      setSelectedTrainId(hoverInfo.train.id === selectedTrainId ? null : hoverInfo.train.id);
    };

    canvas.addEventListener('mousemove', handleMove);
    canvas.addEventListener('click', handleClick);
    return () => {
      canvas.removeEventListener('mousemove', handleMove);
      canvas.removeEventListener('click', handleClick);
    };
  }, [selectedSectionId, baseTrains, sections, trains, hoverInfo, selectedTrainId]);

  // controller quick actions
  const holdResumeTrain = (id) => {
    setTrains(prev => prev.map(t => t.id === id ? { ...t, speedKmph: t.speedKmph === 0 ? t.plannedSpeedKmph : 0, status: t.speedKmph === 0 ? 'Resumed' : 'Holding' } : t));
  };
  const prioritizeTrain = (id) => {
    // bump priority (smaller number higher priority)
    setTrains(prev => prev.map(t => t.id === id ? { ...t, priority: Math.max(0, t.priority - 1), status: 'Prioritised' } : t));
  };
  const swapTrack = (id) => {
    setTrains(prev => prev.map(t => t.id === id ? { ...t, track: (t.track + 1) % trackCount, status: 'Track Swap' } : t));
  };

  return (
    <div ref={containerRef} className="min-h-screen p-6" style={{ background: INDIAN_RAIL_BG }}>
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="p-3 rounded-xl bg-gradient-to-r from-yellow-400 via-orange-500 to-red-600">
            <Train className="h-8 w-8 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">AI Train Traffic Control — Interactive Section</h1>
            <p className="text-slate-200 text-sm">Enhanced visualization · click trains to open quick controls · realistic tracks, signals & stations</p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <div className="bg-black/40 text-white rounded-xl px-3 py-2">
            <div className="text-xs text-slate-200">Section</div>
            <select value={selectedSectionId} onChange={(e) => setSelectedSectionId(e.target.value)} className="bg-transparent text-white border-none">
              {sections.map(s => <option className="text-black" key={s.id} value={s.id}>{s.name}</option>)}
            </select>
          </div>
          <div className="bg-black/40 rounded-xl p-3 text-white">
            <div className="text-xs text-slate-300">Throughput</div>
            <div className="text-white font-bold">{kpis.throughput}</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 mb-6">
        <div className="bg-black/40 p-4 rounded-xl">
          <div className="flex gap-2">
            <button onClick={() => setIsRunning(r => !r)} className={`flex-1 py-2 rounded-lg font-medium ${isRunning ? 'bg-red-500' : 'bg-green-500'} text-white`}>
              {isRunning ? <><Pause className="inline-block mr-2" /> Pause</> : <><Play className="inline-block mr-2" /> Start</>}
            </button>
            <button onClick={resetSimulation} className="px-4 py-2 bg-slate-600 rounded-lg text-white"><RotateCcw /></button>
          </div>

          <div className="mt-3">
            <label className="text-slate-200 text-sm">Sim speed: {simulationSpeed}x</label>
            <input type="range" min="0.5" max="4" step="0.5" value={simulationSpeed} onChange={(e) => setSimulationSpeed(parseFloat(e.target.value))} className="w-full mt-2" />
          </div>

          <div className="mt-3">
            <label className="text-slate-200 text-sm">Tick sec (internal)</label>
            <input type="range" min="0.1" max="1" step="0.1" value={tickSec} onChange={(e) => setTickSec(parseFloat(e.target.value))} className="w-full mt-2" />
            <div className="text-xs text-slate-300 mt-1">Faster ticks produce smoother motion</div>
          </div>
        </div>

        <div className="lg:col-span-3 bg-black/40 p-4 rounded-xl">
          <h3 className="text-white font-semibold mb-2">AI Metrics & Quick KPIs</h3>
          <div className="grid grid-cols-5 gap-3 text-sm">
            <div className="text-center">
              <div className="text-white font-bold">94%</div>
              <div className="text-slate-300">Throughput Opt</div>
            </div>
            <div className="text-center">
              <div className="text-white font-bold">78%</div>
              <div className="text-slate-300">Delay Reduction</div>
            </div>
            <div className="text-center">
              <div className="text-white font-bold">{kpis.conflictsResolved}</div>
              <div className="text-slate-300">Conflicts Resolved</div>
            </div>
            <div className="text-center">
              <div className="text-white font-bold">{kpis.avgDelayMin}m</div>
              <div className="text-slate-300">Avg Delay</div>
            </div>
            <div className="text-center">
              <div className="text-white font-bold">{kpis.efficiencyPct}%</div>
              <div className="text-slate-300">Efficiency</div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-black/40 p-4 rounded-xl mb-6">
        <div className="mb-3 flex justify-between items-center">
          <h4 className="text-white font-semibold">Live Track Visualization</h4>
          <div className="text-sm text-slate-200">Tracks: {trackCount} | Section distance: {sections.find(s => s.id === selectedSectionId).distanceKm} km</div>
        </div>
        <div style={{ height: 420 }} className="relative">
          <canvas ref={canvasRef} style={{ width: '100%', height: '100%', borderRadius: 12 }} />

          {/* hover tooltip */}
          {hoverInfo && (
            <div style={{ position: 'absolute', left: hoverInfo.x + 6, top: hoverInfo.y + 6, background: PANEL_BG, color: '#fff', padding: 8, borderRadius: 8, zIndex: 40, pointerEvents: 'none', minWidth: 220 }}>
              <div className="font-semibold">{hoverInfo.train.name} <span className="text-xs text-slate-300">({hoverInfo.train.id})</span></div>
              <div className="text-xs mt-1">Type: {hoverInfo.train.type} • Track {hoverInfo.train.track + 1}</div>
              <div className="text-xs">Speed: {hoverInfo.train.speedKmph} km/h • Status: {hoverInfo.train.status}</div>
            </div>
          )}

        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-black/40 p-4 rounded-xl">
          <h4 className="text-white font-semibold mb-3">Active Trains ({trains.filter(t => !t.finished).length})</h4>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {trains.map(tr => (
              <div key={tr.id} className={`p-3 rounded-lg border ${selectedTrainId === tr.id ? 'border-yellow-400' : 'border-slate-700'}`} style={{ background: 'rgba(255,255,255,0.02)' }}>
                <div className="flex justify-between">
                  <div>
                    <div className="text-white font-semibold">{tr.name} <span className="text-sm text-slate-300">({tr.id})</span></div>
                    <div className="text-slate-300 text-sm">{tr.type} • Track {tr.track + 1}</div>
                  </div>
                  <div className="text-sm text-slate-200">{tr.status}</div>
                </div>

                <div className="grid grid-cols-4 gap-2 text-sm mt-2">
                  <div><div className="text-slate-300">Speed</div><div className="text-white">{tr.speedKmph} km/h</div></div>
                  <div><div className="text-slate-300">Position</div><div className="text-white">{Math.min(sections.find(s => s.id === selectedSectionId).distanceKm, tr.positionKm).toFixed(2)} km</div></div>
                  <div><div className="text-slate-300">ETA vs Plan</div><div className="text-white">{(tr.delaySec / 60).toFixed(1)} min</div></div>
                  <div><div className="text-slate-300">Progress</div><div className="text-white">{Math.min(100, (tr.positionKm / sections.find(s => s.id === selectedSectionId).distanceKm * 100)).toFixed(1)}%</div></div>
                </div>

                <div className="mt-3 flex gap-2">
                  <button onClick={() => holdResumeTrain(tr.id)} className="px-3 py-1 bg-yellow-500 rounded text-black text-sm">Hold/Resume</button>
                  <button onClick={() => prioritizeTrain(tr.id)} className="px-3 py-1 bg-green-600 rounded text-white text-sm">Prioritise</button>
                  <button onClick={() => swapTrack(tr.id)} className="px-3 py-1 bg-slate-600 rounded text-white text-sm">Swap Track</button>
                </div>

              </div>
            ))}
          </div>
        </div>

        <div className="col-span-2 space-y-4">
          <div className="bg-black/40 p-4 rounded-xl">
            <h4 className="text-white font-semibold mb-2">System Alerts & Recommendations</h4>
            <div className="text-slate-200 text-sm space-y-2">
              <div>• Click a train on canvas or in the panel to perform quick actions (Hold / Prioritise / Swap track).</div>
              <div>• Conflicts: system will first try reroute, otherwise hold the lower-priority train.</div>
              <div>• For production: replace heuristics with OR solver (CP / MILP) or RL policy for optimized decisions & explainability.</div>
            </div>
          </div>

          <div className="bg-black/40 p-4 rounded-xl">
            <h4 className="text-white font-semibold mb-2">Quick Statistics</h4>
            <div className="grid grid-cols-4 gap-3">
              <div className="text-center">
                <div className="text-2xl text-white font-bold">{kpis.throughput}</div>
                <div className="text-slate-300 text-xs">Trains Passed</div>
              </div>
              <div className="text-center">
                <div className="text-2xl text-white font-bold">{kpis.punctualityPct}%</div>
                <div className="text-slate-300 text-xs">Punctuality</div>
              </div>
              <div className="text-center">
                <div className="text-2xl text-white font-bold">{kpis.avgDelayMin}m</div>
                <div className="text-slate-300 text-xs">Avg Delay</div>
              </div>
              <div className="text-center">
                <div className="text-2xl text-white font-bold">{kpis.conflictsResolved}</div>
                <div className="text-slate-300 text-xs">Conflicts Resolved</div>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default TrainTrafficControl;
