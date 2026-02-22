import React, { useState, useEffect, useMemo } from 'react';
import { Map } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import DeckGL from '@deck.gl/react';
import { PathLayer, SolidPolygonLayer } from '@deck.gl/layers';

// Interpolate blue -> red based on t (0..1)
const lerpColor = (t) => {
  const r = Math.round(60 + t * 195);   // 60 -> 255
  const g = Math.round(140 - t * 100);  // 140 -> 40
  const b = Math.round(220 - t * 180);  // 220 -> 40
  return [60, 120, 200, 180];
  //return [r, g, b, 255];
};

// Build binary attribute buffers for PathLayer.
// Each feature becomes a single path object with per-vertex gradient colors.
// Width is fixed per feature, using vertex count as a proxy for river size.
const processGeoJson = (geojson) => {
  const features = geojson.features;
  const n = features.length;

  // First pass: count total vertices per feature (concatenate all lines)
  const vertexCounts = new Array(n);
  let totalVertices = 0;
  for (let fi = 0; fi < n; fi++) {
    let count = 0;
    for (const line of features[fi].geometry.coordinates) count += line.length;
    vertexCounts[fi] = count;
    totalVertices += count;
  }

  const positions    = new Float64Array(totalVertices * 3);
  const colors       = new Uint8Array(totalVertices * 4);
  const startIndices = new Uint32Array(n + 1); // +1 for sentinel
  const widths       = new Float32Array(n);
  const names        = new Array(n);

  let vo = 0; // vertex offset into flat arrays

  for (let fi = 0; fi < n; fi++) {
    const feature = features[fi];
    startIndices[fi] = vo;
    names[fi] = feature.properties?.name ?? null;
    if (names[fi] && names[fi].includes(" |")) names[fi] = names[fi].split(" |")[0];

    const nv = vertexCounts[fi]; // total vertices for this feature
    let lv = 0; // local vertex index within the feature

    for (const line of feature.geometry.coordinates) {
      for (const coord of line) {
        const t = nv <= 1 ? 0.5 : lv / (nv - 1);
        const [r, g, b, a] = lerpColor(t);
        colors[vo * 4]     = r;
        colors[vo * 4 + 1] = g;
        colors[vo * 4 + 2] = b;
        colors[vo * 4 + 3] = a;
        positions[vo * 3]     = coord[0];
        positions[vo * 3 + 1] = coord[1];
        positions[vo * 3 + 2] = coord[2] ?? 0;
        vo++;
        lv++;
      }
    }

    widths[fi] = feature.properties?.gauge_ids?.length > 0 ? 400 : 80;
  }
  startIndices[n] = vo; // sentinel: end of last path

  return {
    length: n,
    startIndices,
    attributes: {
      getPath:  { value: positions, size: 3 },
      getColor: { value: colors,    size: 4 },
    },
    widths,
    names,
  };
};

const INITIAL_VIEW_STATE = {
  longitude: 8.2,
  latitude: 46.8,
  zoom: 7.5,
  pitch: 0,
  bearing: 0
};

const MAP_STYLE = {
  version: 8,
  sources: {
    'local-tiles': {
      type: 'raster',
      tiles: ['/geodata/tiles/{z}/{x}/{y}.png'],
      tileSize: 256,
      minzoom: 7,
      maxzoom: 12,
      bounds: [2.8125, 43.0689, 14.0625, 48.9225],
    }
  },
  layers: [
    {
      id: 'background',
      type: 'background',
      paint: { 'background-color': '#333333' }
    },
    {
      id: 'local-tiles',
      type: 'raster',
      source: 'local-tiles',
    }
  ]
};

const SwissRiversDeckGL = () => {
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
  const [geojson, setGeojson] = useState(null);
  const [lakes, setLakes] = useState(null);
  const [hoverInfo, setHoverInfo] = useState(null);

  useEffect(() => {
    fetch('/geodata/outputs/rivers.geojson')
      .then(res => res.json())
      .then(setGeojson);
    fetch('/geodata/inputs/lakes.geojson')
      .then(res => res.json())
      .then(setLakes);
  }, []);

  const riverData = useMemo(() => {
    if (!geojson) return null;
    return processGeoJson(geojson);
  }, [geojson]);

  const layers = useMemo(() => {
    const result = [];
    if (riverData) {
      result.push(new PathLayer({
        id: 'rivers',
        data: riverData,
        getWidth: (_, { index }) => riverData.widths[index],
        widthUnits: 'meters',
        widthMinPixels: 1,
        widthMaxPixels: 20,
        capRounded: true,
        jointRounded: true,
        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 255, 120],
        onHover: info => {
          if (info.index >= 0) {
            setHoverInfo({ x: info.x, y: info.y, name: riverData.names[info.index] });
          } else {
            setHoverInfo(null);
          }
        },
      }));
    }
    if (lakes) {
      result.push(new SolidPolygonLayer({
        id: 'lakes',
        data: lakes.features,
        getPolygon: d => d.geometry.coordinates,
        getFillColor: [60, 120, 200, 180],
        extruded: false,
      }));
    }
    return result;
  }, [riverData, lakes]);

  return (
    <div style={{
      position: 'relative',
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
      {hoverInfo && hoverInfo.name && (
        <div style={{
          position: 'absolute',
          left: hoverInfo.x + 12,
          top: hoverInfo.y + 12,
          background: 'rgba(0,0,0,0.75)',
          color: '#fff',
          padding: '4px 10px',
          borderRadius: 4,
          fontSize: 13,
          pointerEvents: 'none',
        }}>
          {hoverInfo.name}
        </div>
      )}
    </div>
  );
};

export default SwissRiversDeckGL;
