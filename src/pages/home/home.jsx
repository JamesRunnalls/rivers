import React, { useState, useMemo } from 'react';
import { Map } from 'react-map-gl/maplibre';
import DeckGL from '@deck.gl/react';
import { PathLayer } from '@deck.gl/layers';

// Sample Swiss river data - each point has [longitude, latitude, widthFactor]
// widthFactor represents how wide the river is at that point (0 = source, 1 = mouth)
const RIVER_DATA = [
  {
    name: 'Aare',
    temperature: 12.5,
    baseFlow: 560,
    path: [
      [8.3, 46.72, 0.3], [8.28, 46.75, 0.35], [8.25, 46.78, 0.4],
      [8.22, 46.82, 0.5], [8.18, 46.85, 0.6], [8.12, 46.88, 0.7],
      [8.05, 46.92, 0.8], [7.98, 46.95, 0.85], [7.9, 46.98, 0.9],
      [7.82, 47.0, 0.95], [7.75, 47.02, 1.0]
    ]
  },
  {
    name: 'Rhône',
    temperature: 8.2,
    baseFlow: 340,
    path: [
      [8.0, 46.38, 0.25], [7.85, 46.35, 0.3], [7.7, 46.32, 0.4],
      [7.55, 46.3, 0.5], [7.4, 46.28, 0.6], [7.25, 46.25, 0.7],
      [7.1, 46.22, 0.75], [6.95, 46.2, 0.8], [6.8, 46.22, 0.85],
      [6.65, 46.28, 0.9], [6.5, 46.35, 1.0]
    ]
  },
  {
    name: 'Reuss',
    temperature: 10.8,
    baseFlow: 280,
    path: [
      [8.62, 46.65, 0.2], [8.58, 46.72, 0.3], [8.55, 46.78, 0.4],
      [8.52, 46.85, 0.5], [8.48, 46.92, 0.6], [8.45, 46.98, 0.7],
      [8.42, 47.05, 0.8], [8.38, 47.12, 0.9], [8.35, 47.18, 1.0]
    ]
  },
  {
    name: 'Limmat',
    temperature: 14.2,
    baseFlow: 180,
    path: [
      [8.54, 47.37, 0.4], [8.52, 47.38, 0.5], [8.48, 47.38, 0.6],
      [8.44, 47.39, 0.7], [8.4, 47.4, 0.8], [8.35, 47.42, 0.9],
      [8.3, 47.44, 1.0]
    ]
  },
  {
    name: 'Vorderrhein',
    temperature: 6.5,
    baseFlow: 150,
    path: [
      [8.75, 46.63, 0.15], [8.82, 46.68, 0.2], [8.88, 46.72, 0.25],
      [8.95, 46.78, 0.35], [9.02, 46.82, 0.45], [9.1, 46.85, 0.55],
      [9.18, 46.88, 0.65], [9.28, 46.9, 0.75], [9.38, 46.92, 0.85],
      [9.48, 46.95, 1.0]
    ]
  },
  {
    name: 'Thur',
    temperature: 11.5,
    baseFlow: 120,
    path: [
      [9.1, 47.25, 0.3], [9.0, 47.3, 0.4], [8.88, 47.35, 0.5],
      [8.75, 47.42, 0.65], [8.62, 47.48, 0.8], [8.55, 47.52, 1.0]
    ]
  },
  {
    name: 'Saane',
    temperature: 9.8,
    baseFlow: 95,
    path: [
      [7.35, 46.52, 0.2], [7.28, 46.58, 0.3], [7.22, 46.65, 0.45],
      [7.15, 46.72, 0.6], [7.1, 46.78, 0.75], [7.05, 46.82, 0.9],
      [7.0, 46.85, 1.0]
    ]
  },
  {
    name: 'Emme',
    temperature: 10.2,
    baseFlow: 75,
    path: [
      [7.95, 46.92, 0.25], [7.88, 46.95, 0.35], [7.82, 46.98, 0.5],
      [7.75, 47.02, 0.65], [7.68, 47.06, 0.8], [7.62, 47.1, 1.0]
    ]
  },
  {
    name: 'Ticino',
    temperature: 15.5,
    baseFlow: 200,
    path: [
      [8.72, 46.15, 0.2], [8.78, 46.12, 0.3], [8.85, 46.08, 0.4],
      [8.9, 46.02, 0.55], [8.92, 45.95, 0.7], [8.9, 45.88, 0.85],
      [8.85, 45.82, 1.0]
    ]
  },
  {
    name: 'Inn',
    temperature: 5.8,
    baseFlow: 85,
    path: [
      [10.0, 46.42, 0.2], [10.08, 46.48, 0.3], [10.18, 46.52, 0.45],
      [10.28, 46.55, 0.6], [10.38, 46.58, 0.75], [10.48, 46.6, 1.0]
    ]
  },
  {
    name: 'Linth',
    temperature: 9.2,
    baseFlow: 110,
    path: [
      [9.05, 46.92, 0.2], [9.02, 46.98, 0.3], [8.98, 47.05, 0.45],
      [8.95, 47.12, 0.6], [8.92, 47.18, 0.8], [8.88, 47.22, 1.0]
    ]
  },
  {
    name: 'Birs',
    temperature: 11.8,
    baseFlow: 65,
    path: [
      [7.42, 47.18, 0.25], [7.48, 47.22, 0.4], [7.52, 47.28, 0.55],
      [7.58, 47.35, 0.7], [7.6, 47.42, 0.85], [7.59, 47.48, 1.0]
    ]
  }
];

// Temperature to RGBA color
const getTemperatureColor = (temp, alpha = 255) => {
  const minTemp = 4;
  const maxTemp = 18;
  const normalized = Math.max(0, Math.min(1, (temp - minTemp) / (maxTemp - minTemp)));
  
  const colors = [
    [20, 60, 180],    // Deep cold blue
    [30, 144, 200],   // Cyan blue
    [50, 180, 140],   // Teal
    [120, 200, 80],   // Green
    [220, 200, 50],   // Yellow
    [255, 140, 40],   // Orange
    [255, 60, 50]     // Red
  ];
  
  const idx = normalized * (colors.length - 1);
  const lower = Math.floor(idx);
  const upper = Math.min(lower + 1, colors.length - 1);
  const t = idx - lower;
  
  return [
    Math.round(colors[lower][0] + (colors[upper][0] - colors[lower][0]) * t),
    Math.round(colors[lower][1] + (colors[upper][1] - colors[lower][1]) * t),
    Math.round(colors[lower][2] + (colors[upper][2] - colors[lower][2]) * t),
    alpha
  ];
};

// Initial view centered on Switzerland
const INITIAL_VIEW_STATE = {
  longitude: 8.2,
  latitude: 46.8,
  zoom: 7.5,
  pitch: 0,
  bearing: 0
};

// Dark map style
const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

const SwissRiversDeckGL = () => {
  const [hoveredRiver, setHoveredRiver] = useState(null);
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);

  // Process river data for Deck.gl PathLayer
  // For variable width, we need to split each river into segments
  const processedData = useMemo(() => {
    const segments = [];
    
    RIVER_DATA.forEach((river, riverIndex) => {
      const baseWidth = (river.baseFlow / 560) * 800 + 200; // Scale width by flow (in meters)
      const color = getTemperatureColor(river.temperature);
      
      // Create segments between each pair of points
      for (let i = 0; i < river.path.length - 1; i++) {
        const p1 = river.path[i];
        const p2 = river.path[i + 1];
        
        // Average width factor for this segment
        const avgWidthFactor = (p1[2] + p2[2]) / 2;
        
        segments.push({
          path: [[p1[0], p1[1]], [p2[0], p2[1]]],
          width: baseWidth * avgWidthFactor,
          color: color,
          riverIndex: riverIndex,
          riverName: river.name,
          temperature: river.temperature,
          flow: river.baseFlow,
          widthFactor: avgWidthFactor
        });
      }
    });
    
    return segments;
  }, []);

  // Create river layer (no glow, full opacity)
  const layers = useMemo(() => {
    return [
      new PathLayer({
        id: 'rivers-core',
        data: processedData,
        getPath: d => d.path,
        getWidth: d => d.width,
        getColor: d => {
          const [r, g, b] = d.color;
          return [r, g, b, 255];
        },
        widthUnits: 'meters',
        widthMinPixels: 1,
        widthMaxPixels: 30,
        capRounded: true,
        jointRounded: true,
        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 255, 80],
        onHover: (info) => {
          setHoveredRiver(info.object ? info.object.riverIndex : null);
        }
      })
    ];
  }, [processedData]);

  // Tooltip content
  const getTooltip = ({ object }) => {
    if (!object) return null;
    return {
      html: `
        <div style="
          background: rgba(10, 20, 35, 0.95);
          padding: 12px 16px;
          border-radius: 8px;
          border: 1px solid rgba(100, 150, 200, 0.3);
          font-family: 'JetBrains Mono', monospace;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        ">
          <div style="color: #e8f0ff; font-size: 14px; font-weight: 600; margin-bottom: 8px;">
            ${object.riverName}
          </div>
          <div style="color: rgba(150, 180, 220, 0.9); font-size: 11px; line-height: 1.6;">
            <div>🌡️ Temperature: <span style="color: rgb(${object.color.slice(0, 3).join(',')})">${object.temperature.toFixed(1)}°C</span></div>
            <div>💧 Flow: ${object.flow} m³/s</div>
          </div>
        </div>
      `,
      style: {
        backgroundColor: 'transparent',
        border: 'none',
        padding: 0
      }
    };
  };

  return (
    <div style={{ 
      width: '100vw', 
      height: '100vh', 
      background: '#0a0f1a',
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace"
    }}>
      <DeckGL
        viewState={viewState}
        onViewStateChange={({ viewState }) => setViewState(viewState)}
        controller={true}
        layers={layers}
        getTooltip={getTooltip}
      >
        <Map mapStyle={MAP_STYLE} />
      </DeckGL>

      {/* Header overlay */}
      <div style={{
        position: 'absolute',
        top: 20,
        left: 20,
        zIndex: 1,
        pointerEvents: 'none'
      }}>
        <h1 style={{
          fontSize: '24px',
          fontWeight: 300,
          color: '#e8f0ff',
          margin: 0,
          letterSpacing: '3px',
          textTransform: 'uppercase',
          textShadow: '0 2px 10px rgba(0,0,0,0.8)'
        }}>
          <span style={{ color: '#ff4444' }}>Swiss</span> Rivers
        </h1>
        <p style={{
          color: 'rgba(150, 180, 220, 0.8)',
          fontSize: '10px',
          marginTop: '4px',
          letterSpacing: '2px',
          textShadow: '0 1px 5px rgba(0,0,0,0.8)'
        }}>
          TEMPERATURE & FLOW • DECK.GL
        </p>
      </div>

      {/* Legend panel */}
      <div style={{
        position: 'absolute',
        bottom: 30,
        left: 20,
        zIndex: 1,
        background: 'rgba(10, 20, 35, 0.9)',
        borderRadius: '8px',
        padding: '16px',
        border: '1px solid rgba(60, 100, 150, 0.3)',
        backdropFilter: 'blur(10px)',
        minWidth: '200px'
      }}>
        <h3 style={{
          color: '#a8c8e8',
          fontSize: '10px',
          letterSpacing: '2px',
          marginBottom: '10px',
          fontWeight: 500,
          margin: 0
        }}>
          WATER TEMPERATURE
        </h3>
        <div style={{
          height: '12px',
          borderRadius: '3px',
          background: 'linear-gradient(90deg, rgb(20,60,180), rgb(30,144,200), rgb(50,180,140), rgb(120,200,80), rgb(220,200,50), rgb(255,140,40), rgb(255,60,50))',
          margin: '8px 0'
        }} />
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          color: 'rgba(150, 180, 220, 0.8)',
          fontSize: '9px'
        }}>
          <span>4°C</span>
          <span>11°C</span>
          <span>18°C</span>
        </div>

        <h3 style={{
          color: '#a8c8e8',
          fontSize: '10px',
          letterSpacing: '2px',
          marginTop: '16px',
          marginBottom: '8px',
          fontWeight: 500
        }}>
          FLOW VOLUME
        </h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' }}>
          <div style={{
            width: '40px',
            height: '3px',
            background: 'rgba(100, 160, 220, 0.8)',
            borderRadius: '2px'
          }} />
          <span style={{ color: 'rgba(150, 180, 220, 0.8)', fontSize: '9px' }}>Low</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '40px',
            height: '10px',
            background: 'rgba(100, 160, 220, 0.8)',
            borderRadius: '5px'
          }} />
          <span style={{ color: 'rgba(150, 180, 220, 0.8)', fontSize: '9px' }}>High</span>
        </div>
      </div>

      {/* River list panel */}
      <div style={{
        position: 'absolute',
        top: 80,
        right: 20,
        zIndex: 1,
        background: 'rgba(10, 20, 35, 0.9)',
        borderRadius: '8px',
        padding: '16px',
        border: '1px solid rgba(60, 100, 150, 0.3)',
        backdropFilter: 'blur(10px)',
        maxHeight: 'calc(100vh - 120px)',
        overflow: 'auto',
        width: '220px'
      }}>
        <h3 style={{
          color: '#a8c8e8',
          fontSize: '10px',
          letterSpacing: '2px',
          marginBottom: '12px',
          fontWeight: 500,
          margin: 0,
          marginBottom: '12px'
        }}>
          RIVERS
        </h3>
        {RIVER_DATA.map((river, idx) => {
          const color = getTemperatureColor(river.temperature);
          const isHovered = hoveredRiver === idx;
          return (
            <div
              key={river.name}
              onMouseEnter={() => setHoveredRiver(idx)}
              onMouseLeave={() => setHoveredRiver(null)}
              onClick={() => {
                // Fly to river center
                const midPoint = river.path[Math.floor(river.path.length / 2)];
                setViewState({
                  ...viewState,
                  longitude: midPoint[0],
                  latitude: midPoint[1],
                  zoom: 9,
                  transitionDuration: 1000
                });
              }}
              style={{
                padding: '8px 10px',
                marginBottom: '4px',
                borderRadius: '4px',
                background: isHovered ? 'rgba(60, 100, 150, 0.4)' : 'rgba(30, 50, 80, 0.3)',
                border: `1px solid ${isHovered ? `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.6)` : 'transparent'}`,
                cursor: 'pointer',
                transition: 'all 0.15s ease'
              }}
            >
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                <div style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: `rgb(${color[0]}, ${color[1]}, ${color[2]})`,
                  boxShadow: `0 0 6px rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.6)`
                }} />
                <span style={{
                  color: '#e8f0ff',
                  fontSize: '11px',
                  fontWeight: 500
                }}>
                  {river.name}
                </span>
              </div>
              <div style={{
                display: 'flex',
                gap: '12px',
                marginTop: '4px',
                marginLeft: '16px'
              }}>
                <span style={{
                  color: 'rgba(150, 180, 220, 0.7)',
                  fontSize: '9px'
                }}>
                  {river.temperature.toFixed(1)}°C
                </span>
                <span style={{
                  color: 'rgba(150, 180, 220, 0.7)',
                  fontSize: '9px'
                }}>
                  {river.baseFlow} m³/s
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Instructions */}
      <div style={{
        position: 'absolute',
        bottom: 30,
        right: 20,
        zIndex: 1,
        background: 'rgba(10, 20, 35, 0.8)',
        borderRadius: '6px',
        padding: '10px 14px',
        border: '1px solid rgba(60, 100, 150, 0.2)'
      }}>
        <p style={{
          color: 'rgba(120, 150, 190, 0.7)',
          fontSize: '9px',
          margin: 0,
          lineHeight: 1.5
        }}>
          Scroll to zoom • Drag to pan • Click river to focus
        </p>
      </div>
    </div>
  );
};

export default SwissRiversDeckGL;