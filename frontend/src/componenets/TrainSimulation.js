import React, { useEffect, useRef, useState } from 'react';
import './index.css';

// ----Utilities----
const now = () => Date.now();
function formatTime(ms) {
  const d = new Date(ms);
  return d.toLocaleTimeString();
}

// Simple unique id
let _id = 1000;
function uid(prefix = 'id') { return `${prefix}-${_id++}`; }

// ---- Default network & trains ----
const DEFAULT_STATIONS = [
  { id: 'S1', name: 'STA' },
  { id: 'S2', name: 'A' },
  { id: 'S3', name: 'B' },
  { id: 'S4', name: 'C' },
  { id: 'S5', name: 'STB' },
];

const makeDefaultTrains = () => ([
  {
    id: 'T1', name: 'Express 101', type: 'Express', priority: 1,
    route: DEFAULT_STATIONS.map(s => s.id),
    speed: 18, // km/h abstract units
    positionSeg: 0, progress: 0, status: 'running', // progress 0..1 between segments
    scheduled: [0, 8, 20, 35, 50], // scheduled minutes from t0 (example)
    actualDelay: 0,
    events: [],
  },
  {
    id: 'T2', name: 'Local 502', type: 'Local', priority: 2,
    route: DEFAULT_STATIONS.map(s => s.id),
    speed: 12,
    positionSeg: 0, progress: 0, status: 'running',
    scheduled: [0, 12, 28, 44, 60],
    actualDelay: 0,
    events: [],
  },
  {
    id: 'T3', name: 'Freight 300', type: 'Freight', priority: 3,
    route: DEFAULT_STATIONS.map(s => s.id),
    speed: 8,
    positionSeg: 0, progress: 0, status: 'running',
    scheduled: [0, 20, 50, 80, 110],
    actualDelay: 0,
    events: [],
  }
]);

// --- Main App ---
export default function App() {
  // simulation clock (minutes) starting at 0
  const [simTime, setSimTime] = useState(0); // minutes
  const [running, setRunning] = useState(true);
  const [stations] = useState(DEFAULT_STATIONS);
  const [trains, setTrains] = useState(makeDefaultTrains);
  const [log, setLog] = useState([]);
  const intervalRef = useRef(null);
  const t0 = useRef(Date.now());

  // KPI counters
  const [arrivals, setArrivals] = useState(0);

  // Start simulation timer
  useEffect(() => {
    if (running) {
      intervalRef.current = setInterval(() => {
        setSimTime(prev => prev + 0.5); // advance by 0.5 minutes per tick (~30 seconds real-time)
      }, 1000);
    }
    return () => clearInterval(intervalRef.current);
  }, [running]);

  // Advance trains each simTime tick
  useEffect(() => {
    // update trains based on simTime
    setTrains(prev => prev.map(train => {
      if (train.status === 'stopped' || train.status === 'arrived') return train;

      // compute progress increment: based on speed; map speed -> progress per minute
      const segCount = train.route.length - 1;
      const segTime = 10 * (20 / train.speed); // heuristic: faster speed => less time per segment
      const progressInc = (0.5 / segTime); // since simTime increment is 0.5 min per tick

      let progress = train.progress + progressInc;
      let seg = train.positionSeg;
      let status = train.status;
      let actualDelay = train.actualDelay;

      if (progress >= 1) {
        // arrive at next station
        progress = 0;
        seg = Math.min(seg + 1, segCount);
        logAction(`${train.name} reached ${stations[seg].name}`);
        // if final station
        if (seg === segCount) {
          status = 'arrived';
          setArrivals(a => a + 1);
        }
      }

      // random small chance of incident causing hold
      if (Math.random() < 0.002) {
        status = 'held';
        logAction(`${train.name} held due to simulated incident`);
      }

      return { ...train, progress, positionSeg: seg, status, actualDelay };
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [simTime]);

  // Helpers
  function logAction(text) {
    setLog(prev => [{ t: new Date(), text, id: uid('log') }, ...prev].slice(0, 200));
  }

  function applyOverride(trainId, action, payload = {}) {
    setTrains(prev => prev.map(train => {
      if (train.id !== trainId) return train;
      let t = { ...train };
      if (action === 'hold') { t.status = 'held'; logAction(`${t.name} held by controller`); }
      if (action === 'resume') { t.status = 'running'; logAction(`${t.name} resumed`); }
      if (action === 'stop') { t.status = 'stopped'; logAction(`${t.name} stopped`); }
      if (action === 'advance') { t.positionSeg = Math.min(t.positionSeg + 1, t.route.length - 1); t.progress = 0; logAction(`${t.name} advanced to next segment`); }
      if (action === 'priority') { t.priority = payload.priority; logAction(`${t.name} priority set to ${payload.priority}`); }
      if (action === 'reroute') { if (Array.isArray(payload.route)) { t.route = payload.route; t.positionSeg = 0; t.progress = 0; logAction(`${t.name} rerouted`); } }
      return t;
    }));
  }

  // What-if: a very simple heuristic optimizer that suggests precedence order for trains in section
  // Strategy: sort by priority, then by estimated remaining time (faster first)
  function suggestSchedule() {
    // estimate remaining time per train: remaining segments * baseTime / speed
    const suggestions = trains
      .filter(t => t.status !== 'arrived')
      .map(t => {
        const remSeg = (t.route.length - 1) - t.positionSeg - (t.progress > 0.001 ? 0 : 1);
        const est = Math.max(1, remSeg) * (20 / t.speed);
        return { id: t.id, name: t.name, priority: t.priority, est };
      })
      .sort((a,b) => { if (a.priority !== b.priority) return a.priority - b.priority; return a.est - b.est; });

    logAction('Optimizer suggested order: ' + suggestions.map(s => s.name).join(' -> '));
    return suggestions;
  }

  // Apply suggestion by forcing lower priority trains to hold for a short time
  function applySuggestion() {
    const suggestion = suggestSchedule();
    // let top two proceed, others held for 1 minute
    const allow = new Set(suggestion.slice(0,2).map(s => s.id));
    setTrains(prev => prev.map(t => {
      if (allow.has(t.id)) return { ...t, status: 'running' };
      return { ...t, status: 'held' };
    }));
    // schedule resume after 1 minute sim-time -> translate to real time: 1 minute sim = 2 seconds per 0.5 tick => 4 seconds
    setTimeout(() => {
      setTrains(prev => prev.map(t => ({ ...t, status: t.status === 'held' ? 'running' : t.status })));
      logAction('Optimizer applied temporary holds released');
    }, 4000);
  }

  // UI actions
  function pauseSim() { setRunning(false); logAction('Simulation paused'); }
  function resumeSim() { setRunning(true); logAction('Simulation resumed'); }
  function resetSim() {
    setTrains(makeDefaultTrains()); setSimTime(0); setLog([]); setArrivals(0); logAction('Simulation reset');
  }

  // KPI calculations
  const avgDelay = trains.length ? (trains.reduce((s,t) => s + (t.actualDelay||0), 0) / trains.length).toFixed(1) : 0;

  // render
  return (
    <div className="app-root">
      <header className="topbar">
        <h1>Train Traffic — Section Simulator</h1>
        <div className="top-controls">
          <div>Sim time: {simTime.toFixed(1)} min</div>
          <button onClick={running ? pauseSim : resumeSim}>{running ? 'Pause' : 'Resume'}</button>
          <button onClick={resetSim}>Reset</button>
          <button onClick={() => { const s=suggestSchedule(); alert('Suggested order:\n'+s.map(x=>x.name).join('\n')); }}>Suggest</button>
          <button onClick={applySuggestion}>Apply Suggestion</button>
        </div>
      </header>

      <main className="main-grid">
        <section className="viz">
          <Schematic stations={stations} trains={trains} onOverride={applyOverride} />
        </section>

        <aside className="side">
          <KPI avgDelay={avgDelay} arrivals={arrivals} active={trains.filter(t=>t.status==='running').length} />

          <div className="panel">
            <h3>Trains</h3>
            <div className="train-list">
              {trains.map(t => (
                <div key={t.id} className="train-row">
                  <div className="train-info">
                    <strong>{t.name}</strong>
                    <div>{t.type} • Pri {t.priority} • {t.status}</div>
                    <div>Next: {t.route[Math.min(t.positionSeg+1, t.route.length-1)]}</div>
                  </div>
                  <div className="train-actions">
                    <button onClick={() => applyOverride(t.id, 'hold')}>Hold</button>
                    <button onClick={() => applyOverride(t.id, 'resume')}>Resume</button>
                    <button onClick={() => applyOverride(t.id, 'advance')}>Advance</button>
                    <button onClick={() => applyOverride(t.id, 'priority', { priority: 1 })}>High</button>
                    <button onClick={() => applyOverride(t.id, 'stop')}>Stop</button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="panel">
            <h3>Timeline (schedule vs progress)</h3>
            <Timeline trains={trains} t0={t0.current} />
          </div>

          <div className="panel">
            <h3>Audit Log</h3>
            <div className="log">
              {log.map(entry => (
                <div key={entry.id} className="log-row">
                  <div className="log-time">{formatTime(entry.t)}</div>
                  <div className="log-text">{entry.text}</div>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </main>

      <footer className="footer">Prototype — frontend-only. Use "Suggest" to get optimizer order and "Apply Suggestion" to force temporary holds.</footer>
    </div>
  );
}

// -- Components --
function Schematic({ stations, trains, onOverride }) {
  const width = 920;
  const height = 240;
  const padding = 60;
  const usable = width - padding * 2;
  const gap = usable / (stations.length - 1);

  return (
    <div className="schematic-card">
      <svg width={width} height={height}>
        {/* track */}
        <line x1={padding} y1={height/2} x2={width-padding} y2={height/2} stroke="#444" strokeWidth={6} strokeLinecap="round" />

        {/* stations */}
        {stations.map((s, i) => {
          const x = padding + i * gap;
          return (
            <g key={s.id}>
              <circle cx={x} cy={height/2} r={12} fill="#fff" stroke="#222" />
              <text x={x} y={height/2 + 36} fontSize={12} textAnchor="middle">{s.name}</text>
            </g>
          );
        })}

        {/* trains */}
        {trains.map((t, idx) => {
          const segCount = t.route.length - 1;
          const segIndex = Math.min(t.positionSeg, segCount);
          const nextIndex = Math.min(segIndex + 1, segCount);
          const xBase = padding + segIndex * gap;
          const xNext = padding + nextIndex * gap;
          const x = xBase + (xNext - xBase) * (t.progress || 0);
          const y = height/2 - 30 - (idx % 4) * 18;
          const color = t.priority === 1 ? '#d9534f' : t.priority === 2 ? '#f0ad4e' : '#5bc0de';

          return (
            <g key={t.id} className="train">
              <rect x={x-18} y={y-10} width={36} height={20} rx={6} fill={color} stroke="#222" />
              <text x={x} y={y+4} fontSize={10} textAnchor="middle" fill="#111">{t.name}</text>
              <foreignObject x={x-30} y={y-44} width={80} height={30}>
                <div className="train-popup">
                  <div style={{fontWeight:700}}>{t.name}</div>
                  <div style={{fontSize:11}}>{t.status} • Pri {t.priority}</div>
                  <div className="popup-actions">
                    <button onClick={() => onOverride(t.id, 'hold')}>Hold</button>
                    <button onClick={() => onOverride(t.id, 'resume')}>Resume</button>
                  </div>
                </div>
              </foreignObject>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function KPI({ avgDelay, arrivals, active }) {
  return (
    <div className="kpi-panel">
      <div className="kpi"> <div className="kpi-num">{avgDelay}</div> <div className="kpi-label">Avg Delay (min)</div> </div>
      <div className="kpi"> <div className="kpi-num">{arrivals}</div> <div className="kpi-label">Arrivals</div> </div>
      <div className="kpi"> <div className="kpi-num">{active}</div> <div className="kpi-label">Active Trains</div> </div>
    </div>
  );
}

function Timeline({ trains }) {
  // create simple bars using scheduled array
  const totalWidth = 300;
  const maxTime = Math.max(...trains.flatMap(t => t.scheduled || [0,60]));
  return (
    <div>
      {trains.map(t => (
        <div key={t.id} style={{marginBottom:8}}>
          <div style={{fontWeight:700}}>{t.name}</div>
          <div className="timeline-row">
            {t.scheduled.map((s,i) => {
              const left = (s / maxTime) * totalWidth;
              return <div key={i} className="sched-dot" style={{left}} />;
            })}
            {/* marker showing current pos */}
            <div className="pos-marker" style={{left: ((t.positionSeg + t.progress) / (t.route.length-1)) * totalWidth}} />
          </div>
        </div>
      ))}
    </div>
  );
}