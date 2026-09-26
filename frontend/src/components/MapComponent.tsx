import React, { useEffect, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import type { ResilienceResponse, HealResponse, GeoJSONLineString } from '../api';

try { delete (L.Icon.Default.prototype as any)._getIconUrl; } catch(e) {}
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Map bounds fitter strictly adhering to Step 8
const MapFitter: React.FC<{ bounds: L.LatLngBounds | null; fallbackStart: [number, number]; fallbackEnd: [number, number] }> = ({
  bounds,
  fallbackStart,
  fallbackEnd
}) => {
  const map = useMap();
  useEffect(() => {
    if (bounds && bounds.isValid()) {
      map.fitBounds(bounds, { padding: [35, 35], maxZoom: 16 });
    } else if (fallbackStart && fallbackEnd) {
      map.fitBounds([fallbackStart, fallbackEnd], { padding: [35, 35] });
    }
  }, [bounds, fallbackStart, fallbackEnd, map]);
  return null;
};

interface MapProps {
  start: [number, number];
  end: [number, number];
  isDisrupted: boolean;
  data: ResilienceResponse | null;
  healResult?: HealResponse | null;
  uploadedGeometry?: GeoJSONLineString[] | null;
  graphVersion: string;
}

// Convert GeoJSON [longitude, latitude] to Leaflet Polyline [latitude, longitude] (Step 5)
const toLatLngs = (lines?: GeoJSONLineString[]): [number, number][][] => {
  if (!lines || lines.length === 0) return [];
  return lines.map(line =>
    line.coordinates.map(coord => [coord[1], coord[0]] as [number, number])
  );
};

export const MapComponent: React.FC<MapProps> = ({
  start,
  end,
  isDisrupted,
  data,
  healResult,
  uploadedGeometry,
  graphVersion
}) => {
  // Step 3: Logging data flow at MAP PROPS
  useEffect(() => {
    if (healResult) {
      console.log(`[MAP PROPS] graphVersion = ${graphVersion}`);
      console.log(`[MAP PROPS] original geometry count = ${healResult.original_graph_geometry?.length ?? 0}`);
      console.log(`[MAP PROPS] damaged geometry count = ${healResult.damaged_graph_geometry?.length ?? 0}`);
      console.log(`[MAP PROPS] removed geometry count = ${healResult.removed_edge_geometry?.length ?? 0}`);
      console.log(`[MAP PROPS] accepted geometry count = ${healResult.accepted_healed_geometry?.length ?? 0}`);
      console.log(`[MAP PROPS] unrecovered geometry count = ${healResult.unrecovered_removed_geometry?.length ?? 0}`);
      console.log(`[MAP PROPS] healed geometry count = ${healResult.healed_graph_geometry?.length ?? 0}`);
    }
  }, [healResult, graphVersion]);

  // Convert geometries with [lat, lon] coordinate ordering
  const origLatLngs = useMemo(() => toLatLngs(healResult?.original_graph_geometry), [healResult?.original_graph_geometry]);
  const damagedLatLngs = useMemo(() => toLatLngs(healResult?.damaged_graph_geometry), [healResult?.damaged_graph_geometry]);
  const removedLatLngs = useMemo(() => toLatLngs(healResult?.removed_edge_geometry), [healResult?.removed_edge_geometry]);
  const acceptedLatLngs = useMemo(() => toLatLngs(healResult?.accepted_healed_geometry), [healResult?.accepted_healed_geometry]);
  const unrecoveredLatLngs = useMemo(() => toLatLngs(healResult?.unrecovered_removed_geometry), [healResult?.unrecovered_removed_geometry]);
  const healedLatLngs = useMemo(() => toLatLngs(healResult?.healed_graph_geometry), [healResult?.healed_graph_geometry]);
  const uploadedLatLngs = useMemo(() => toLatLngs(uploadedGeometry ?? undefined), [uploadedGeometry]);
  // Route geometries
  const normLatLngs = useMemo(() => {
    if (!data?.normal?.geometry) return [];
    return data.normal.geometry.coordinates.map(c => [c[1], c[0]] as [number, number]);
  }, [data?.normal?.geometry]);

  const disLatLngs = useMemo(() => {
    if (!data?.disrupted?.geometry) return [];
    return data.disrupted.geometry.coordinates.map(c => [c[1], c[0]] as [number, number]);
  }, [data?.disrupted?.geometry]);

  const disruptionsLatLngs = useMemo(() => {
    if (!data?.disruptions) return [];
    return data.disruptions.map(d => d.geometry.coordinates.map(c => [c[1], c[0]] as [number, number]));
  }, [data?.disruptions]);

  // Step 8: Calculate bounds based on active graph geometry or route
  const mapBounds = useMemo(() => {
    const b = L.latLngBounds([]);
    if (isDisrupted && disLatLngs.length > 0) {
      disLatLngs.forEach(pt => b.extend(pt));
      normLatLngs.forEach(pt => b.extend(pt));
      return b;
    }
    if (normLatLngs.length > 0) {
      normLatLngs.forEach(pt => b.extend(pt));
      return b;
    }
    // Fit to graph geometry if no route active
    const activeLines = graphVersion === 'original'
      ? origLatLngs
      : graphVersion === 'damaged'
      ? damagedLatLngs
      : healedLatLngs;

    if (activeLines.length > 0) {
      activeLines.forEach(line => line.forEach(pt => b.extend(pt)));
      return b.isValid() ? b : null;
    }

    if (uploadedLatLngs.length > 0) {
      uploadedLatLngs.forEach(line => line.forEach(pt => b.extend(pt)));
      return b.isValid() ? b : null;
    }

    return null;
  }, [isDisrupted, disLatLngs, normLatLngs, graphVersion, origLatLngs, damagedLatLngs, healedLatLngs, uploadedLatLngs]);

  // Counts for diagnostic display
  const apiOrigCount = healResult?.original_graph_geometry?.length ?? healResult?.diagnostics?.original_geometry_count ?? uploadedLatLngs.length;
  const apiDamagedCount = healResult?.damaged_graph_geometry?.length ?? healResult?.diagnostics?.damaged_geometry_count ?? 0;
  const apiRemovedCount = healResult?.removed_edge_geometry?.length ?? healResult?.diagnostics?.removed_geometry_count ?? 0;
  const apiAcceptedCount = healResult?.accepted_healed_geometry?.length ?? healResult?.diagnostics?.accepted_candidate_geometry_count ?? 0;
  const apiHealedCount = healResult?.healed_graph_geometry?.length ?? healResult?.diagnostics?.healed_geometry_count ?? 0;

  // Rendered counts based on active mode
  const renderedOriginal = graphVersion === 'original' ? (origLatLngs.length > 0 ? origLatLngs.length : uploadedLatLngs.length) : 0;
  const renderedDamaged = graphVersion === 'damaged' ? damagedLatLngs.length : (graphVersion === 'healed' ? damagedLatLngs.length : 0);
  const renderedRemoved = graphVersion === 'damaged' ? removedLatLngs.length : (graphVersion === 'healed' ? unrecoveredLatLngs.length : 0);
  const renderedHealed = graphVersion === 'healed' ? acceptedLatLngs.length : 0;

  return (
    <div className="relative h-full w-full">
      <MapContainer center={start} zoom={14} className="h-full w-full min-h-[500px]">
        {/* Base Map Layer */}
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution="&copy; OpenStreetMap" />

        {/* Origin / Destination Markers */}
        <Marker position={start}>
          <Popup>
            <div className="text-xs">
              <strong>Origin</strong>
              <div>{start[0].toFixed(5)}, {start[1].toFixed(5)}</div>
            </div>
          </Popup>
        </Marker>

        <Marker position={end}>
          <Popup>
            <div className="text-xs">
              <strong>Destination</strong>
              <div>{end[0].toFixed(5)}, {end[1].toFixed(5)}</div>
            </div>
          </Popup>
        </Marker>

        <MapFitter bounds={mapBounds} fallbackStart={start} fallbackEnd={end} />

        {/* ========================================================================= */}
        {/* RENDER GRAPH STATES                                                       */}
        {/* ========================================================================= */}

        {/* 1. ORIGINAL VIEW: Render original graph edges or uploaded graphML edges */}
        {graphVersion === 'original' && (
          <>
            {origLatLngs.length > 0 && (
              <Polyline
                positions={origLatLngs}
                pathOptions={{
                  color: '#2563eb',
                  weight: 3.5,
                  opacity: 0.85
                }}
              />
            )}
            {!healResult && uploadedLatLngs.length > 0 && (
              <Polyline
                positions={uploadedLatLngs}
                pathOptions={{
                  color: '#2563eb',
                  weight: 3.5,
                  opacity: 0.85
                }}
              />
            )}
          </>
        )}

        {/* 2. DAMAGED VIEW: Render surviving roads + removed roads */}
        {graphVersion === 'damaged' && (
          <>
            {damagedLatLngs.length > 0 && (
              <Polyline
                positions={damagedLatLngs}
                pathOptions={{
                  color: '#475569',
                  weight: 3.0,
                  opacity: 0.80
                }}
              />
            )}
            {removedLatLngs.length > 0 && (
              <Polyline
                positions={removedLatLngs}
                pathOptions={{
                  color: '#ef4444',
                  weight: 5.0,
                  dashArray: '8, 8',
                  opacity: 1.0
                }}
              />
            )}
          </>
        )}

        {/* 3. HEALED VIEW: Surviving roads + accepted healed candidates + unrecovered */}
        {graphVersion === 'healed' && (
          <>
            {damagedLatLngs.length > 0 && (
              <Polyline
                positions={damagedLatLngs}
                pathOptions={{
                  color: '#475569',
                  weight: 3.0,
                  opacity: 0.75
                }}
              />
            )}
            {unrecoveredLatLngs.length > 0 && (
              <Polyline
                positions={unrecoveredLatLngs}
                pathOptions={{
                  color: '#ef4444',
                  weight: 4.0,
                  dashArray: '6, 6',
                  opacity: 0.85
                }}
              />
            )}
            {acceptedLatLngs.length > 0 && (
              <Polyline
                positions={acceptedLatLngs}
                pathOptions={{
                  color: '#10b981',
                  weight: 6.0,
                  opacity: 1.0
                }}
              />
            )}
          </>
        )}

        {/* ========================================================================= */}
        {/* ROUTE OVERLAYS                                                            */}
        {/* ========================================================================= */}
        {normLatLngs.length > 0 && (
          <Polyline
            positions={normLatLngs}
            pathOptions={{
              color: isDisrupted ? '#94a3b8' : '#1d4ed8',
              weight: isDisrupted ? 4 : 6,
              opacity: isDisrupted ? 0.6 : 0.95
            }}
          />
        )}


        {isDisrupted && disLatLngs.length > 0 && (
          <Polyline
            positions={disLatLngs}
            pathOptions={{
              color: '#9333ea',
              weight: 6.5,
              opacity: 0.95
            }}
          />
        )}

        {isDisrupted && disruptionsLatLngs.length > 0 && disruptionsLatLngs.map((pts, i) => (
          <Polyline
            key={`disr-${i}`}
            positions={pts}
            pathOptions={{
              color: '#dc2626',
              weight: 8,
              dashArray: '10, 10',
              opacity: 1
            }}
          />
        ))}

        {/* ========================================================================= */}
        {/* DIAGNOSTIC & DEBUG OVERLAY PANEL                                          */}
        {/* ========================================================================= */}
        {(healResult || uploadedLatLngs.length > 0) && (
          <div className="absolute top-3 right-3 z-[1000] bg-slate-900/90 text-white p-3 rounded-md shadow-lg text-[11px] font-mono border border-slate-700 max-w-xs space-y-1">
            <div className="font-bold text-amber-400 text-xs border-b border-slate-700 pb-1">DIAGNOSTICS</div>
            <div className="text-gray-300 font-semibold pt-0.5">API Geometry:</div>
            <div className="flex justify-between pl-2 text-gray-300"><span>Original:</span> <span className="text-blue-400 font-bold">{apiOrigCount}</span></div>
            <div className="flex justify-between pl-2 text-gray-300"><span>Damaged:</span> <span className="text-slate-300 font-bold">{apiDamagedCount}</span></div>
            <div className="flex justify-between pl-2 text-gray-300"><span>Removed:</span> <span className="text-red-400 font-bold">{apiRemovedCount}</span></div>
            <div className="flex justify-between pl-2 text-gray-300"><span>Healed candidates:</span> <span className="text-emerald-400 font-bold">{apiAcceptedCount}</span></div>
            <div className="flex justify-between pl-2 text-gray-300"><span>Healed graph:</span> <span className="text-emerald-300 font-bold">{apiHealedCount}</span></div>

            <div className="text-gray-300 font-semibold pt-1 border-t border-slate-800">Rendered (Mode: {graphVersion.toUpperCase()}):</div>
            {graphVersion === 'original' && (
              <div className="flex justify-between pl-2 text-blue-300"><span>Original roads:</span> <span className="font-bold">{renderedOriginal}</span></div>
            )}
            {graphVersion === 'damaged' && (
              <>
                <div className="flex justify-between pl-2 text-slate-300"><span>Surviving roads:</span> <span className="font-bold">{renderedDamaged}</span></div>
                <div className="flex justify-between pl-2 text-red-400"><span>Removed roads:</span> <span className="font-bold">{renderedRemoved}</span></div>
              </>
            )}
            {graphVersion === 'healed' && (
              <>
                <div className="flex justify-between pl-2 text-slate-300"><span>Surviving roads:</span> <span className="font-bold">{renderedDamaged}</span></div>
                <div className="flex justify-between pl-2 text-emerald-400"><span>Healed roads:</span> <span className="font-bold">{renderedHealed}</span></div>
                <div className="flex justify-between pl-2 text-red-400"><span>Unrecovered missing:</span> <span className="font-bold">{unrecoveredLatLngs.length}</span></div>
                <div className="flex justify-between pl-2 text-amber-300 font-bold pt-0.5 border-t border-slate-800"><span>Total visible network:</span> <span>{renderedDamaged + renderedHealed}</span></div>
              </>
            )}
          </div>
        )}

        {/* ========================================================================= */}
        {/* MAP LEGEND                                                                */}
        {/* ========================================================================= */}
        <div className="absolute bottom-6 left-2 z-[1000] bg-white/95 p-3 text-xs text-gray-800 rounded shadow-md border border-gray-200 max-w-sm">
          <div className="space-y-3 font-medium">
            <div>
              <div className="font-bold text-gray-900 mb-1 border-b pb-1 text-xs">GRAPH TOPOLOGY ({graphVersion.toUpperCase()})</div>
              {graphVersion === 'original' && (
                <div className="flex items-center gap-2 mt-1">
                  <div className="w-4 h-1 bg-blue-600"></div>
                  <span>ORIGINAL GRAPH ({apiOrigCount} edges)</span>
                </div>
              )}
              {graphVersion === 'damaged' && (
                <>
                  <div className="flex items-center gap-2 mt-1">
                    <div className="w-4 h-1 bg-slate-600"></div>
                    <span>DAMAGED GRAPH (surviving: {apiDamagedCount} edges)</span>
                  </div>
                  <div className="flex items-center gap-2 mt-1">
                    <div className="w-4 h-1 border-t-2 border-dashed border-red-500"></div>
                    <span>REMOVED ROAD (controlled: {apiRemovedCount} edges)</span>
                  </div>
                </>
              )}
              {graphVersion === 'healed' && (
                <>
                  <div className="flex items-center gap-2 mt-1">
                    <div className="w-4 h-1 bg-slate-600"></div>
                    <span>SURVIVING ORIGINAL ROADS ({apiDamagedCount} edges)</span>
                  </div>
                  <div className="flex items-center gap-2 mt-1">
                    <div className="w-4 h-1 bg-emerald-500"></div>
                    <span>HEALED TOPOLOGY (accepted: {apiAcceptedCount} edges)</span>
                  </div>
                  <div className="flex items-center gap-2 mt-1">
                    <div className="w-4 h-1 border-t-2 border-dashed border-red-500"></div>
                    <span>UNRECOVERED MISSING ROAD ({unrecoveredLatLngs.length} edges)</span>
                  </div>
                </>
              )}
            </div>

            <div>
              <div className="font-bold text-gray-900 mb-1 border-b pb-1 text-xs">ROUTE &amp; DISRUPTIONS</div>
              <div className="flex items-center gap-2 mt-1">
                <div className="w-2.5 h-2.5 rounded-full bg-blue-600"></div>
                <span>Origin ({start[0].toFixed(3)}, {start[1].toFixed(3)})</span>
              </div>
              <div className="flex items-center gap-2 mt-1">
                <div className="w-2.5 h-2.5 rounded-full bg-red-600"></div>
                <span>Destination ({end[0].toFixed(3)}, {end[1].toFixed(3)})</span>
              </div>
              {data && (
                <>
                  <div className="flex items-center gap-2 mt-1"><div className="w-4 h-1 bg-blue-700"></div> Normal route</div>
                  {isDisrupted && <div className="flex items-center gap-2 mt-1"><div className="w-4 h-1 bg-purple-600"></div> Alternative route</div>}
                  {isDisrupted && <div className="flex items-center gap-2 mt-1"><div className="w-4 h-1 border-t-2 border-dashed border-red-600"></div> Flooded segment</div>}
                </>
              )}
            </div>
          </div>
        </div>
      </MapContainer>
    </div>
  );
};
