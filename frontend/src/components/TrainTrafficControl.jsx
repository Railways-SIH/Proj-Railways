// SchematicTrainPanel.jsx
import React, { useEffect, useState } from "react";
import { UncontrolledReactSVGPanZoom } from "react-svg-pan-zoom";
import "./SchematicTrainPanel.css";

const API_BASE = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";

export default function SchematicTrainPanel() {
  const [schem, setSchem] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/schematic?osm=true`)
      .then((r) => r.json())
      .then(setSchem)
      .catch((err) => {
        console.error("schematic fetch failed", err);
        fetch(`${API_BASE}/schematic?osm=false`)
          .then((r) => r.json())
          .then(setSchem);
      });
  }, []);

  if (!schem) return <div style={{ padding: 20 }}>Loading schematic...</div>;

  // ✅ safe defaults
  const stations = schem.stations || [];
  const blocks = schem.blocks || [];
  const signals = schem.signals || [];
  const junctions = schem.junctions || [];
  const crossings = schem.crossings || [];
  const trains = schem.trains || [];

  // ✅ Scale & spacing
  const SCALE = 4; // increase this for more spacing
  const OFFSET = 80; // margin
  const project = (x, y) => ({
    X: (typeof x === "number" ? x : 0) * SCALE + OFFSET,
    Y: (typeof y === "number" ? y : 0) * SCALE + OFFSET,
  });

  // compute bounds for canvas
  const allX = [
    ...stations.map((s) => s.x || 0),
    ...blocks.map((b) => (b.x || 0) + (b.width || 0)),
  ];
  const allY = [
    ...stations.map((s) => s.y || 0),
    ...blocks.map((b) => (b.y || 0) + (b.height || 0)),
  ];
  const maxX = Math.max(...allX, 800) * SCALE + 200;
  const maxY = Math.max(...allY, 600) * SCALE + 200;

  return (
    <div style={{ height: "92vh", display: "flex" }}>
      {/* Left SVG schematic with Pan/Zoom */}
      <div style={{ flex: 1, overflow: "hidden", background: "#081021" }}>
        <UncontrolledReactSVGPanZoom
          width={Math.max(1100, maxX)}
          height={Math.max(600, maxY)}
          background="#081021"
          tool="auto"
          detectAutoPan={true}
        >
          <svg width={maxX} height={maxY}>
            <rect x={0} y={0} width="100%" height="100%" fill="#081021" />

            {/* blocks */}
            {blocks.map((block) => {
              const { X, Y } = project(block.x, block.y);
              return (
                <g key={block.id}>
                  <rect
                    x={X}
                    y={Y}
                    width={(block.width || 100) * SCALE * 0.6}
                    height={(block.height || 20) * SCALE * 0.4}
                    rx="6"
                    className={`schem-block ${
                      block.status === "occupied" ? "occupied" : "free"
                    }`}
                  />
                  <text
                    x={X + (block.width || 100) * SCALE * 0.3}
                    y={Y - 8}
                    textAnchor="middle"
                    className="schem-label"
                  >
                    {block.id}
                  </text>
                </g>
              );
            })}

            {/* stations */}
            {stations.map((st) => {
              const { X, Y } = project(st.x, st.y);
              return (
                <g key={st.id}>
                  <circle cx={X} cy={Y} r={14} className="schem-station" />
                  <text
                    x={X}
                    y={Y + 36}
                    textAnchor="middle"
                    className="schem-label"
                  >
                    {st.name} ({st.id})
                  </text>
                </g>
              );
            })}

            {/* signals */}
            {signals.map((sig) => {
              const { X, Y } = project(sig.x, sig.y);
              return (
                <g key={sig.id}>
                  <rect
                    x={X}
                    y={Y}
                    width={12}
                    height={12}
                    rx={3}
                    className="schem-signal"
                  />
                  <text
                    x={X}
                    y={Y - 6}
                    textAnchor="middle"
                    className="schem-label"
                  >
                    🚦{sig.id}
                  </text>
                </g>
              );
            })}

            {/* junctions */}
            {junctions.map((j) => {
              const { X, Y } = project(j.x, j.y);
              return (
                <g key={j.id}>
                  <rect x={X} y={Y} width={14} height={14} rx={3} className="schem-junction" />
                  <text
                    x={X}
                    y={Y - 6}
                    textAnchor="middle"
                    className="schem-label"
                  >
                    ⚡{j.id}
                  </text>
                </g>
              );
            })}

            {/* crossings */}
            {crossings.map((c) => {
              const { X, Y } = project(c.x, c.y);
              return (
                <g key={c.id}>
                  <rect x={X} y={Y} width={16} height={16} rx={4} className="schem-crossing" />
                  <text
                    x={X}
                    y={Y - 6}
                    textAnchor="middle"
                    className="schem-label"
                  >
                    🚧{c.id}
                  </text>
                </g>
              );
            })}
          </svg>
        </UncontrolledReactSVGPanZoom>
      </div>

      {/* Sidebar info */}
      <div className="schem-sidebar">
        <h4 className="schem-title">Block Status</h4>
        {blocks.map((b) => (
          <div
            key={b.id}
            className={`schem-sidebar-block ${
              b.status === "occupied" ? "occupied" : "free"
            }`}
          >
            <div>{b.id}</div>
            <div style={{ fontSize: 12 }}>{b.status}</div>
          </div>
        ))}

        <h4 className="schem-title">Stations</h4>
        {stations.map((st) => (
          <div key={st.id} className="schem-sidebar-card">
            <div style={{ fontWeight: 700 }}>
              {st.name} ({st.id})
            </div>
            <div style={{ fontSize: 12 }}>Platforms: {st.platforms}</div>
          </div>
        ))}

        <h4 className="schem-title">Trains</h4>
        {trains.map((t) => (
          <div key={t.id} className="schem-sidebar-card">
            <div style={{ fontWeight: 700 }}>
              {t.id} — {t.name}
            </div>
            <div style={{ fontSize: 12 }}>Pos: {JSON.stringify(t.position)}</div>
            <div style={{ fontSize: 12 }}>
              Route: {t.route?.join(" → ")}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
