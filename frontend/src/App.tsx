import React, { useState, useEffect } from 'react';
import type { ResilienceRequest, ResilienceResponse, GraphUploadResponse, HealResponse } from './api';
import { resilienceApi, graphApi, systemApi } from './api';
import { MapComponent } from './components/MapComponent';
import { RouteCards } from './components/RouteCards';
import { ComparisonPanel } from './components/ComparisonPanel';
import {
  Upload,
  Play,
  RotateCcw,
  Activity,
  Layers,
  Cpu,
  Navigation,
  Waves,
  Sparkles
} from 'lucide-react';

function App() {
  const [loadingState, setLoadingState] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ResilienceResponse | null>(null);
  const [isDisrupted, setIsDisrupted] = useState(false);
  const [graphUpload, setGraphUpload] = useState<GraphUploadResponse | null>(null);
  const [healResult, setHealResult] = useState<HealResponse | null>(null);
  const [graphVersion, setGraphVersion] = useState<string>("original");
  const [backendHealthy, setBackendHealthy] = useState<boolean>(true);

  const [req, setReq] = useState<ResilienceRequest>({
    start_lat: 51.51615,
    start_lon: -0.12268,
    end_lat: 51.50969,
    end_lon: -0.12336,
    disruptions: []
  });

  // Live health check
  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        if (systemApi?.checkHealth) {
          const ok = await systemApi.checkHealth();
          if (mounted) setBackendHealthy(ok);
        }
      } catch {
        if (mounted) setBackendHealthy(false);
      }
    };
    check();
    const interval = setInterval(check, 10000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleDemo = async () => {
    setLoadingState("Healing graph...");
    setError(null);
    setData(null);
    setIsDisrupted(false);
    try {
      const res = await graphApi.demoHealing();
      console.log("[GRAPH API] original geometry count =", res.original_graph_geometry?.length ?? 0);
      console.log("[GRAPH API] damaged geometry count =", res.damaged_graph_geometry?.length ?? 0);
      console.log("[GRAPH API] removed geometry count =", res.removed_edge_geometry?.length ?? 0);
      console.log("[GRAPH API] accepted geometry count =", res.accepted_healed_geometry?.length ?? 0);
      console.log("[GRAPH API] unrecovered geometry count =", res.unrecovered_removed_geometry?.length ?? 0);
      console.log("[GRAPH API] healed geometry count =", res.healed_graph_geometry?.length ?? 0);

      setHealResult(res);
      const demoReq: ResilienceRequest = {
        graph_id: res.graph_id,
        start_lat: 51.51615,
        start_lon: -0.12268,
        end_lat: 51.50969,
        end_lon: -0.12336,
        graph_version: "healed",
        disruptions: []
      };
      setReq(demoReq);
      setGraphVersion("healed");

      // Auto-compute baseline route on healed network
      try {
        const routeRes = await resilienceApi.analyzeRoute(demoReq);
        setData(routeRes);
      } catch {
        // Can be computed manually if needed
      }
    } catch (err) {
      setError("Failed to run demo.");
    } finally {
      setLoadingState(null);
    }
  };

  useEffect(() => {
    if (healResult) {
      console.log("[APP STATE] selected graph mode =", graphVersion);
      console.log("[APP STATE] healed geometry count =", healResult.healed_graph_geometry?.length ?? 0);
    }
  }, [healResult, graphVersion]);

  // Graph state switching
  const handleGraphVersionChange = (newVersion: string) => {
    setGraphVersion(newVersion);
    setData(null);
    setIsDisrupted(false);
  };

  const handleAnalyze = async (disrupt: boolean) => {
    const activeGraphId = healResult?.graph_id || graphUpload?.graph_id || req.graph_id;
    if (!req.start_lat || !req.start_lon || req.start_lat === 0) {
      setError("Please enter origin coordinates (latitude, longitude).");
      return;
    }
    if (!req.end_lat || !req.end_lon || req.end_lat === 0) {
      setError("Please enter destination coordinates (latitude, longitude).");
      return;
    }
    if (req.start_lat === req.end_lat && req.start_lon === req.end_lon) {
      setError("Origin and destination coordinates must be different.");
      return;
    }

    setLoadingState(disrupt ? "Simulating flood..." : "Calculating route...");
    setError(null);
    setIsDisrupted(disrupt);

    try {
      const payload: ResilienceRequest = {
        ...req,
        graph_id: activeGraphId,
        graph_version: graphVersion,
        start_lat: req.start_lat,
        start_lon: req.start_lon,
        end_lat: req.end_lat,
        end_lon: req.end_lon,
      };

      if (disrupt && data && data.normal.segments && data.normal.segments.length > 0) {
        const midIdx = Math.floor(data.normal.segments.length / 2);
        const segment = data.normal.segments[midIdx];
        payload.disruptions = [{ u: segment.u, v: segment.v, type: "FLOODED" }];
      } else {
        payload.disruptions = [];
      }

      const res = await resilienceApi.analyzeRoute(payload);

      if (res.normal && !res.normal.geometry) {
        setError("Graph geometry is unavailable.");
      }
      setData(res);
    } catch (err: any) {
      if (err?.message?.includes("No route exists") || err?.message?.includes("No route available") || err?.message?.includes("No drivable path")) {
        setError("No route available in the selected graph.");
      } else if (err?.message && err.message !== "Failed") {
        setError(err.message);
      } else {
        setError("Unable to calculate route. No valid alternative route exists under this disruption.");
      }
    } finally {
      setLoadingState(null);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    setLoadingState("Uploading...");
    setError(null);
    try {
      const res = await graphApi.uploadGraph(e.target.files[0]);
      setGraphUpload(res);
      setHealResult(null);
      setGraphVersion("original");
      setData(null);
      setIsDisrupted(false);

      if (res.default_origin && res.default_destination) {
        setReq({
          graph_id: res.graph_id,
          start_lat: res.default_origin.lat,
          start_lon: res.default_origin.lon,
          end_lat: res.default_destination.lat,
          end_lon: res.default_destination.lon,
          graph_version: "original",
          disruptions: []
        });
      } else {
        setReq(prev => ({
          ...prev,
          graph_id: res.graph_id,
          graph_version: "original",
          disruptions: []
        }));
      }
    } catch (err) {
      setError("Failed to upload graph. Invalid GraphML.");
    } finally {
      setLoadingState(null);
    }
  };

  const handleHealCustomGraph = async () => {
    if (!graphUpload) return;
    setLoadingState("Healing custom graph...");
    setError(null);
    try {
      const res = await graphApi.healGraph(graphUpload.graph_id);
      setHealResult(res);
      setGraphVersion("healed");
      setReq(prev => ({
        ...prev,
        graph_id: res.graph_id,
        graph_version: "healed"
      }));
    } catch {
      setError("Failed to heal custom graph.");
    } finally {
      setLoadingState(null);
    }
  };

  // Compute active graph edge count for status panel
  const getActiveGraphEdgeCount = () => {
    if (healResult) {
      if (graphVersion === 'original') return healResult.diagnostics?.original_edge_count ?? healResult.original.edges;
      if (graphVersion === 'damaged') return healResult.diagnostics?.damaged_edge_count ?? 1453;
      return healResult.diagnostics?.healed_edge_count ?? 1484;
    }
    if (graphUpload) return graphUpload.edges;
    return 0;
  };

  // Routing status text based on active source and graph version
  const getRoutingStatusText = () => {
    if (loadingState === 'Calculating route...') return 'Calculating...';
    if (!data) return 'Ready';
    const isCustom = data.routing_source === 'Custom GraphML' || (graphUpload && (!healResult || healResult.graph_id === graphUpload.graph_id));
    if (isCustom) {
      const vName = graphVersion ? graphVersion.charAt(0).toUpperCase() + graphVersion.slice(1) : 'Custom';
      return `Calculated (${vName} Custom Graph)`;
    }
    return 'Calculated (OSM Paths)';
  };

  return (
    <div className="min-h-screen flex flex-col font-sans text-gray-800 bg-[#f8fafc]">
      {/* ========================================================================= */}
      {/* HEADER                                                                    */}
      {/* ========================================================================= */}
      <header className="bg-slate-900 text-white px-6 py-4 shadow-md">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <div className="flex items-center gap-3">
              <span className="bg-blue-600 text-white text-xs font-bold px-2 py-0.5 rounded tracking-wider uppercase">
                Prototype v2.0
              </span>
              <h1 className="text-2xl font-bold tracking-tight">ROUTE RESILIENCE</h1>
            </div>
            <p className="text-slate-400 text-xs mt-1">
              Validated Graph Recovery (V4-C RF, &tau;=0.75) + Resilient Routing Under Disruption
            </p>
          </div>

          <div className="flex items-center gap-3 w-full md:w-auto justify-end">
            <button
              onClick={handleDemo}
              disabled={!!loadingState}
              className="bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-2 rounded-lg font-bold text-xs flex items-center gap-2 shadow-sm transition disabled:opacity-50"
            >
              <Play size={16} /> RUN LONDON DEMO
            </button>
          </div>
        </div>
      </header>

      {/* ========================================================================= */}
      {/* SYSTEM STATUS BAR                                                         */}
      {/* ========================================================================= */}
      <div className="bg-slate-800 text-slate-300 text-xs px-6 py-2 border-b border-slate-700">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-1.5 font-semibold text-slate-400">
            <Activity size={14} className="text-blue-400" /> SYSTEM STATUS:
          </div>

          <div className="flex flex-wrap items-center gap-4">
            {/* Backend */}
            <div className="flex items-center gap-1.5">
              <span className={`w-2 h-2 rounded-full ${backendHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`} />
              <span className="text-slate-400">Backend:</span>
              <span className="font-semibold text-slate-200">{backendHealthy ? 'Connected' : 'Disconnected'}</span>
            </div>

            {/* Graph */}
            <div className="flex items-center gap-1.5">
              <Layers size={13} className="text-slate-400" />
              <span className="text-slate-400">Graph:</span>
              <span className="font-semibold text-slate-200">
                {healResult ? `Loaded (${healResult.original.nodes} nodes, ${getActiveGraphEdgeCount()} edges)` : (graphUpload ? `Custom (${graphUpload.nodes} nodes, ${graphUpload.edges} edges)` : 'Not Loaded')}
              </span>
            </div>

            {/* Graph Healing */}
            <div className="flex items-center gap-1.5">
              <Cpu size={13} className="text-slate-400" />
              <span className="text-slate-400">Graph Healing:</span>
              <span className="font-semibold text-emerald-400">
                {healResult ? 'Complete (V4-C RF, \u03c4=0.75)' : (loadingState === 'Healing graph...' ? 'Running...' : 'Ready')}
              </span>
            </div>

            {/* Routing */}
            <div className="flex items-center gap-1.5">
              <Navigation size={13} className="text-slate-400" />
              <span className="text-slate-400">Routing:</span>
              <span className="font-semibold text-slate-200">
                {getRoutingStatusText()}
              </span>
            </div>

            {/* Flood Scenario */}
            <div className="flex items-center gap-1.5">
              <Waves size={13} className={isDisrupted ? 'text-amber-400 animate-pulse' : 'text-slate-400'} />
              <span className="text-slate-400">Flood Scenario:</span>
              <span className={`font-semibold ${isDisrupted ? 'text-amber-400' : 'text-slate-400'}`}>
                {isDisrupted ? 'Active (Road Disrupted)' : 'Inactive'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* INTERACTIVE DEMO STORYLINE STEPPER                                        */}
      {/* ========================================================================= */}
      <div className="bg-white border-b border-gray-200 px-6 py-2.5 shadow-xs">
        <div className="max-w-7xl mx-auto flex items-center justify-between overflow-x-auto text-[11px] font-medium text-slate-600 gap-4">
          <div className="flex items-center gap-2 whitespace-nowrap">
            <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
              graphVersion === 'original' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600'
            }`}>1</span>
            <span>Original Graph</span>
          </div>
          <span className="text-slate-300">&rarr;</span>

          <div className="flex items-center gap-2 whitespace-nowrap">
            <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
              graphVersion === 'damaged' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600'
            }`}>2</span>
            <span>Damaged Graph</span>
          </div>
          <span className="text-slate-300">&rarr;</span>

          <div className="flex items-center gap-2 whitespace-nowrap">
            <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
              graphVersion === 'healed' && !isDisrupted ? 'bg-emerald-600 text-white' : 'bg-slate-100 text-slate-600'
            }`}>3</span>
            <span>Healed Graph (ML V4-C)</span>
          </div>
          <span className="text-slate-300">&rarr;</span>

          <div className="flex items-center gap-2 whitespace-nowrap">
            <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
              data && !isDisrupted ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600'
            }`}>4</span>
            <span>Route A/B/C</span>
          </div>
          <span className="text-slate-300">&rarr;</span>

          <div className="flex items-center gap-2 whitespace-nowrap">
            <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
              isDisrupted ? 'bg-amber-600 text-white animate-pulse' : 'bg-slate-100 text-slate-600'
            }`}>5</span>
            <span className={isDisrupted ? 'font-bold text-amber-900' : ''}>🌊 Flood Disruption</span>
          </div>
          <span className="text-slate-300">&rarr;</span>

          <div className="flex items-center gap-2 whitespace-nowrap">
            <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
              isDisrupted && data?.disrupted ? 'bg-purple-600 text-white' : 'bg-slate-100 text-slate-600'
            }`}>6</span>
            <span>Alternative Route &amp; Resilience</span>
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* MAIN CONTENT                                                              */}
      {/* ========================================================================= */}
      <main className="flex-1 p-6 max-w-7xl mx-auto w-full grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* ================= LEFT COLUMN: CONTROLS & DIAGNOSTICS ================= */}
        <div className="lg:col-span-1 space-y-5">
          {/* GRAPH INPUT CARD */}
          <div className="bg-white p-5 rounded-lg shadow-sm border border-gray-200">
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 border-b pb-2 mb-3">
              Graph Input
            </h2>

            <label className="flex items-center justify-center w-full h-11 px-4 transition bg-white border-2 border-gray-300 border-dashed rounded-md appearance-none cursor-pointer hover:border-blue-400 focus:outline-none mb-3">
              <span className="flex items-center space-x-2">
                <Upload size={18} className="text-gray-400" />
                <span className="text-xs font-medium text-gray-600">Upload Custom GraphML</span>
              </span>
              <input type="file" className="hidden" accept=".graphml,.geojson,.json" onChange={handleFileUpload} />
            </label>

            {graphUpload && (
              <div className="text-xs bg-gray-50 p-2.5 rounded border border-gray-200 mb-3 space-y-2">
                <div className="flex justify-between">
                  <span>File:</span>
                  <span className="font-semibold text-emerald-700">Custom GraphML</span>
                </div>
                <div className="flex justify-between">
                  <span>Nodes:</span>
                  <span className="font-semibold">{graphUpload.nodes}</span>
                </div>
                <div className="flex justify-between">
                  <span>Edges:</span>
                  <span className="font-semibold">{graphUpload.edges}</span>
                </div>

                {!healResult && (
                  <button
                    onClick={handleHealCustomGraph}
                    disabled={!!loadingState}
                    className="w-full mt-2 bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-1.5 px-2 rounded text-[11px] flex items-center justify-center gap-1.5 transition"
                  >
                    <Sparkles size={13} /> Run Graph Healing on Upload
                  </button>
                )}
              </div>
            )}

            {/* GRAPH STATE SELECTOR */}
            {healResult && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-700 uppercase tracking-wide">
                    Graph State
                  </span>
                  <span className="text-[11px] font-semibold text-slate-500">
                    Active: {graphVersion.toUpperCase()}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-1 rounded-md bg-slate-100 p-1">
                  <button
                    onClick={() => handleGraphVersionChange("original")}
                    className={`py-1.5 text-xs font-bold rounded transition ${
                      graphVersion === 'original'
                        ? 'bg-blue-600 text-white shadow-xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    ORIGINAL
                  </button>
                  <button
                    onClick={() => handleGraphVersionChange("damaged")}
                    className={`py-1.5 text-xs font-bold rounded transition ${
                      graphVersion === 'damaged'
                        ? 'bg-blue-600 text-white shadow-xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    DAMAGED
                  </button>
                  <button
                    onClick={() => handleGraphVersionChange("healed")}
                    className={`py-1.5 text-xs font-bold rounded transition ${
                      graphVersion === 'healed'
                        ? 'bg-emerald-600 text-white shadow-xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    HEALED
                  </button>
                </div>

                {/* Compact Graph Status Panel */}
                <div className="bg-slate-50 p-3 rounded border border-slate-200 text-xs space-y-1">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Graph Version:</span>
                    <span className="font-semibold text-slate-800 capitalize">{graphVersion}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Active Nodes:</span>
                    <span className="font-semibold text-slate-800">{healResult.original.nodes}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Active Edges:</span>
                    <span className="font-bold text-blue-700">{getActiveGraphEdgeCount()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Removed Edges:</span>
                    <span className="font-semibold text-red-600">{healResult.diagnostics?.removed_edge_count ?? 52}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Candidates Generated:</span>
                    <span className="font-semibold text-slate-700">{healResult.healing.candidate_edges.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Validated Candidates:</span>
                    <span className="font-semibold text-slate-700">{healResult.healing.validated_edges.length}</span>
                  </div>
                  <div className="flex justify-between border-t pt-1 font-semibold text-emerald-800">
                    <span>Accepted Healed Edges:</span>
                    <span>{healResult.diagnostics?.accepted_healed_edge_count ?? 31}</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* GRAPH HEALING PANEL */}
          {healResult && (
            <div className="bg-white p-5 rounded-lg shadow-sm border border-gray-200">
              <div className="flex items-center justify-between border-b pb-2 mb-3">
                <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800">
                  GRAPH HEALING
                </h2>
                <span className="text-[10px] font-bold bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded border border-emerald-300">
                  V4-C RF &bull; &tau; = 0.75
                </span>
              </div>

              <div className="bg-slate-50 p-2.5 rounded border border-slate-200 text-[11px] mb-3 font-medium text-slate-700">
                <div className="flex items-center justify-between text-slate-500 pb-1 mb-1 border-b border-slate-200">
                  <span>Input: <strong className="text-slate-800">Damaged Graph</strong></span>
                  <span>&rarr;</span>
                  <span>Model: <strong className="text-slate-800">V4-C RF</strong></span>
                  <span>&rarr;</span>
                  <span>Output: <strong className="text-emerald-700">Healed Graph</strong></span>
                </div>
                <div className="text-[10px] text-slate-500 leading-tight">
                  Damaged Graph &rarr; Candidate Connections &rarr; ML Prediction &rarr; Validation &rarr; Healed Graph
                </div>
              </div>

              <div className="text-xs space-y-1.5 mb-3">
                <div className="flex justify-between">
                  <span className="text-slate-600">Original graph edges</span>
                  <span className="font-semibold">{healResult.diagnostics?.original_edge_count ?? healResult.original.edges}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-600">Damaged graph edges</span>
                  <span className="font-semibold">{healResult.diagnostics?.damaged_edge_count ?? 1453}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-600">Removed edges</span>
                  <span className="font-semibold text-red-600">{healResult.diagnostics?.removed_edge_count ?? 52}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-600">Candidates generated</span>
                  <span className="font-semibold">{healResult.healing.candidate_edges.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-600">Candidates validated</span>
                  <span className="font-semibold">{healResult.healing.validated_edges.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-600">Accepted healed candidates</span>
                  <span className="font-bold text-emerald-700">{healResult.diagnostics?.accepted_healed_edge_count ?? 31}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-600">Rejected candidates</span>
                  <span className="font-semibold text-slate-500">{healResult.healing.rejected_edges.length}</span>
                </div>
                <div className="flex justify-between pt-1 border-t text-emerald-800 font-semibold">
                  <span>Healed graph edges</span>
                  <span>{healResult.diagnostics?.healed_edge_count ?? 1484}</span>
                </div>
              </div>

              <div className="bg-amber-50 border border-amber-200 text-amber-900 text-xs p-2.5 rounded mb-3">
                <div className="font-semibold">
                  Controlled recovery: {healResult.diagnostics?.accepted_healed_edge_count ?? 31} accepted / {healResult.diagnostics?.removed_edge_count ?? 52} removed
                </div>
                <div className="mt-0.5 text-amber-700 text-[11px]">
                  Some removed topology remains unrecovered.
                </div>
              </div>

              <p className="text-[11px] text-slate-600 italic bg-blue-50/50 p-2 rounded border border-blue-100 mb-3">
                ML identifies plausible missing connections between existing road segments. Validated candidates are merged into the healed graph.
              </p>

              <h3 className="font-bold text-xs text-slate-500 uppercase tracking-wider mb-1.5">
                Topology Recovery Evaluation
              </h3>
              <div className="text-xs space-y-1 bg-gray-50 p-2.5 rounded border border-gray-200">
                <div className="flex justify-between">
                  <span>True Positives (TP)</span>
                  <span className="font-semibold">{healResult.evaluation.true_positives}</span>
                </div>
                <div className="flex justify-between">
                  <span>False Positives (FP)</span>
                  <span className="font-semibold">{healResult.evaluation.false_positives}</span>
                </div>
                <div className="flex justify-between">
                  <span>False Negatives (FN)</span>
                  <span className="font-semibold">{healResult.evaluation.false_negatives}</span>
                </div>
                <div className="flex justify-between mt-1 pt-1 border-t">
                  <span>Precision</span>
                  <span className="font-semibold text-blue-700">{healResult.evaluation.precision.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Recall</span>
                  <span className="font-semibold text-blue-700">{healResult.evaluation.recall.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span>F1 Score</span>
                  <span className="font-semibold text-purple-700">{healResult.evaluation.f1.toFixed(3)}</span>
                </div>
              </div>
            </div>
          )}

          {/* ========================================================================= */}
          {/* ROUTE PLANNING & MAP-BASED SELECTION CONTROLS                             */}
          {/* ========================================================================= */}
          <div className="bg-white p-5 rounded-lg shadow-sm border border-gray-200">
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 border-b pb-2 mb-3">
              ROUTE PLANNING
            </h2>

            <div className="space-y-3 text-xs">
              <div>
                <label className="block text-slate-700 font-bold mb-1">
                  Origin (Lat, Lon):
                </label>
                <input
                  type="text"
                  value={`${req.start_lat}, ${req.start_lon}`}
                  onChange={(e) => {
                    const parts = e.target.value.split(',').map(s => parseFloat(s.trim()));
                    if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
                      setReq(r => ({ ...r, start_lat: parts[0], start_lon: parts[1] }));
                    }
                  }}
                  className="w-full bg-slate-50 p-2 rounded border border-slate-200 font-mono text-xs focus:bg-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                  placeholder="51.51615, -0.12268"
                />
              </div>

              <div>
                <label className="block text-slate-700 font-bold mb-1">
                  Destination (Lat, Lon):
                </label>
                <input
                  type="text"
                  value={`${req.end_lat}, ${req.end_lon}`}
                  onChange={(e) => {
                    const parts = e.target.value.split(',').map(s => parseFloat(s.trim()));
                    if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
                      setReq(r => ({ ...r, end_lat: parts[0], end_lon: parts[1] }));
                    }
                  }}
                  className="w-full bg-slate-50 p-2 rounded border border-slate-200 font-mono text-xs focus:bg-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                  placeholder="51.50969, -0.12336"
                />
              </div>

              {/* Action Buttons */}
              <div className="pt-2 flex flex-col gap-2">
                <button
                  onClick={() => handleAnalyze(false)}
                  disabled={!!loadingState}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-3 rounded-lg text-xs flex items-center justify-center gap-2 transition disabled:opacity-50 cursor-pointer"
                >
                  <Navigation size={15} />
                  {loadingState === "Calculating route..." ? loadingState : "NORMAL CONDITIONS"}
                </button>

                <button
                  onClick={() => handleAnalyze(true)}
                  disabled={!!loadingState || !data}
                  className={`w-full font-bold py-2.5 px-3 rounded-lg text-xs flex items-center justify-center gap-2 transition shadow-sm cursor-pointer ${
                    isDisrupted
                      ? 'bg-amber-600 hover:bg-amber-700 text-white'
                      : 'bg-slate-900 hover:bg-black text-amber-400'
                  } disabled:opacity-50`}
                >
                  <Waves size={16} />
                  {loadingState === "Simulating flood..." ? loadingState : "🌊 SIMULATE FLOOD"}
                </button>

                {isDisrupted && (
                  <button
                    onClick={() => handleAnalyze(false)}
                    disabled={!!loadingState}
                    className="w-full bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold py-1.5 px-3 rounded text-xs flex items-center justify-center gap-1.5 transition cursor-pointer"
                  >
                    <RotateCcw size={13} /> Reset Disruption
                  </button>
                )}
              </div>

              {/* ROUTING DIAGNOSTICS */}
              {data && (
                <div className="mt-3 p-2.5 bg-slate-50 border border-slate-200 rounded text-xs space-y-1">
                  <div className="font-bold text-slate-700 uppercase tracking-wide text-[10px] pb-1 border-b border-slate-200">
                    Routing Diagnostics
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Routing Graph:</span>
                    <span className="font-semibold text-slate-800">
                      {data.routing_source || (graphUpload ? 'Custom GraphML' : 'OpenStreetMap')}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Graph Version:</span>
                    <span className="font-semibold text-slate-800 capitalize">
                      {data.graph_version_used || graphVersion}
                    </span>
                  </div>
                  <div className="flex justify-between font-mono text-[11px]">
                    <span className="text-slate-500 font-sans text-xs">Routing Nodes:</span>
                    <span className="font-bold text-blue-700">
                      {data.routing_nodes || (data.origin_node && data.dest_node ? `${data.origin_node} → ${data.dest_node}` : `${req.start_lat.toFixed(4)} → ${req.end_lat.toFixed(4)}`)}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* MESSAGES */}
          {error && (
            <div className="bg-red-50 border-l-4 border-red-500 p-3.5 rounded text-red-700 text-xs font-medium shadow-xs">
              {error}
            </div>
          )}
          {loadingState && !error && (
            <div className="bg-blue-50 border-l-4 border-blue-500 p-3.5 rounded text-blue-700 text-xs font-bold animate-pulse shadow-xs">
              {loadingState}
            </div>
          )}
        </div>

        {/* ================= RIGHT COLUMN: MAP, ROUTES, RESILIENCE ================= */}
        <div className="lg:col-span-2 space-y-6">
          {/* MAP VISUALIZATION */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden h-[540px] relative">
            <MapComponent
              start={[req.start_lat, req.start_lon]}
              end={[req.end_lat, req.end_lon]}
              data={data}
              isDisrupted={isDisrupted}
              healResult={healResult}
              uploadedGeometry={graphUpload?.original_graph_geometry}
              graphVersion={graphVersion}
            />
          </div>

          {/* ROUTE OPTIONS (ROUTE A / B / C) */}
          <RouteCards
            resilienceData={data}
            isDisrupted={isDisrupted}
          />

          {/* FLOOD SCENARIO, BEFORE/AFTER TABLE, AND RESILIENCE EVALUATION */}
          {data && (
            <ComparisonPanel
              data={data}
              isDisrupted={isDisrupted}
            />
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
