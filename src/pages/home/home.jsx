import React, { useState, useEffect, useMemo } from 'react';
import { Map } from 'react-map-gl/maplibre';
import DeckGL from '@deck.gl/react';
import { PathLayer } from '@deck.gl/layers';

// Interpolate blue -> red based on t (0..1)
const lerpColor = (t) => {
  const r = Math.round(60 + t * 195);   // 60 -> 255
  const g = Math.round(140 - t * 100);  // 140 -> 40
  const b = Math.round(220 - t * 180);  // 220 -> 40
  return [r, g, b, 255];
};

// Convert GeoJSON features into PathLayer segments with gradient colors.
// Each segment is a 3-point path (prev, current, next) so that adjacent
// segments overlap at shared vertices and joints render without holes.
const processGeoJson = (geojson) => {
  const segments = [];
  for (const feature of geojson.features) {
    const lines = feature.geometry.coordinates; // MultiLineString: array of lines
    for (const line of lines) {
      const n = line.length;
      if (n < 2) continue;
      for (let i = 0; i < n - 1; i++) {
        const t = n <= 2 ? 0.5 : i / (n - 2);
        const widthT = n <= 2 ? 0.5 : i / (n - 2);
        // Build a 3-point path: include previous point (if any) so the
        // cap of this segment covers the joint with the previous one.
        const path = [];
        if (i > 0) path.push(line[i - 1]);
        path.push(line[i]);
        path.push(line[i + 1]);
        segments.push({
          path,
          color: lerpColor(t),
          width: 50 + widthT * 450,
        });
      }
    }
  }
  return segments;
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
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
  const [geojson, setGeojson] = useState(null);

  useEffect(() => {
    fetch('/geodata/rivers.geojson')
      .then(res => res.json())
      .then(setGeojson);
  }, []);

  const segments = useMemo(() => {
    if (!geojson) return [];
    return processGeoJson(geojson);
  }, [geojson]);

  const layers = useMemo(() => {
    if (!segments.length) return [];
    return [
      new PathLayer({
        id: 'rivers',
        data: segments,
        getPath: d => d.path,
        getColor: d => d.color,
        getWidth: d => d.width,
        widthUnits: 'meters',
        widthMinPixels: 1,
        widthMaxPixels: 20,
        capRounded: true,
        jointRounded: true,
        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 255, 80],
      })
    ];
  }, [segments]);

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
          RIVER NETWORK • DECK.GL
        </p>
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
          Scroll to zoom • Drag to pan
        </p>
      </div>
    </div>
  );
};

export default SwissRiversDeckGL;
