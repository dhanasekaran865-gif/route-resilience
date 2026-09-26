export interface DisruptionEdge {
  u: number | string;
  v: number | string;
  type: string;
}

export interface ResilienceRequest {
  start_lat: number;
  start_lon: number;
  end_lat: number;
  end_lon: number;
  origin_node?: string | number;
  dest_node?: string | number;
  disruptions?: DisruptionEdge[];
  graph_id?: string;
  graph_version?: string;
}

export interface GraphUploadResponse {
  graph_id: string;
  format?: string;
  nodes: number;
  edges: number;
  crs?: string;
  has_geometry?: boolean;
  status?: string;
  warnings?: string[];
  original_graph_geometry?: GeoJSONLineString[];
  default_origin?: {
    lat: number;
    lon: number;
    node_id?: string;
  } | null;
  default_destination?: {
    lat: number;
    lon: number;
    node_id?: string;
  } | null;
}

export type GeoJSONLineString = {
  type: 'LineString';
  coordinates: [number, number][];
};

export interface GraphStats {
  nodes: number;
  edges: number;
}

export interface RemovedEdge {
  u: number;
  v: number;
  key: number;
  data: Record<string, unknown>;
  geometry: GeoJSONLineString;
}

export interface HealingCandidate {
  u: number;
  v: number;
  distance: number;
}

export interface PredictedEdge {
  u: number;
  v: number;
  probability: number;
  predicted_exists: boolean;
  key: number;
  data: Record<string, unknown>;
  geometry: GeoJSONLineString;
}

export interface RejectedEdge {
  u: number;
  v: number;
  reason: string;
}

export interface GraphHealingMetrics {
  candidate_edges: HealingCandidate[];
  predicted_edges: PredictedEdge[];
  validated_edges: PredictedEdge[];
  rejected_edges: RejectedEdge[];
}

export interface EvaluationMetrics {
  precision: number;
  recall: number;
  f1: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  total_truth: number;
}

export interface GraphDiagnostics {
  original_edge_count: number;
  damaged_edge_count: number;
  removed_edge_count: number;
  accepted_healed_edge_count: number;
  healed_edge_count: number;
  original_geometry_count: number;
  damaged_geometry_count: number;
  removed_geometry_count: number;
  accepted_candidate_geometry_count: number;
  healed_geometry_count: number;
}

export interface AutoODResponse {
  success: boolean;
  origin_node?: string;
  origin_lat?: number;
  origin_lon?: number;
  destination_node?: string;
  destination_lat?: number;
  destination_lon?: number;
  healed_edge?: {
    u: string;
    v: string;
    key: number | string;
  };
  healed_edge_in_route?: boolean;
  damaged_route_exists?: boolean;
  selection_reason?: string;
  healed_edge_geometry?: GeoJSONLineString;
  route_nodes_count?: number;
  edges_before?: number;
  edges_after?: number;
  message?: string;
}

export interface HealResponse {
  graph_id: string;
  original: GraphStats;
  damage: { removed_edges: RemovedEdge[] };
  healing: GraphHealingMetrics;
  evaluation: EvaluationMetrics;
  healed: GraphStats;
  diagnostics?: GraphDiagnostics;
  original_graph_geometry?: GeoJSONLineString[];
  damaged_graph_geometry?: GeoJSONLineString[];
  healed_graph_geometry?: GeoJSONLineString[];
  removed_edge_geometry?: GeoJSONLineString[];
  accepted_healed_geometry?: GeoJSONLineString[];
  unrecovered_removed_geometry?: GeoJSONLineString[];
  auto_od?: AutoODResponse;
}

export interface RouteMetrics {
  distance_km: number;
  travel_time_min: number;
  average_congestion: number;
  max_congestion: number;
  safety_risk_exposure: number;
  weather_risk_exposure: number;
  disrupted_edges: number;
  alternative_routes: number;
  resilience_score: number;
  geometry?: GeoJSONLineString;
  segments?: Array<{ u: number | string; v: number | string; length_m: number }>;
}

export interface DisruptionGeometry {
  segment_id: string;
  type: string;
  geometry: GeoJSONLineString;
}

export interface ResilienceResponse {
  route_id: string;
  routing_source?: string;
  routing_nodes?: string;
  origin_node?: string;
  dest_node?: string;
  graph_version_used?: string;
  normal: RouteMetrics;
  disrupted?: RouteMetrics;
  disruptions?: DisruptionGeometry[];
  resilience: {
    reachable: boolean;
    detour_distance_km: number;
    travel_time_increase_min: number;
    alternative_routes_remaining: number;
  };
  disruption_impact: {
    affected_original_edges: number;
    final_route_disrupted_edges: number;
  };
  comparison: {
    scenario: string;
    selected_route: string;
    reason: string;
  };
}

const API_BASE = 'http://localhost:8000';

export const graphApi = {
  uploadGraph: async (file: File): Promise<GraphUploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/graph/upload`, { method: 'POST', body: formData });
    if (!res.ok) throw new Error('Failed to upload graph.');
    return res.json();
  },
  healGraph: async (graphId: string): Promise<HealResponse> => {
    const res = await fetch(`${API_BASE}/graph/${graphId}/heal`, { method: 'POST' });
    if (!res.ok) throw new Error('Graph healing failed.');
    return res.json();
  },
  demoHealing: async (): Promise<HealResponse> => {
    const res = await fetch(`${API_BASE}/graph/demo/healing`, { method: 'POST' });
    if (!res.ok) throw new Error('Backend unavailable or demo failed.');
    return res.json();
  },
  autoSelectOD: async (graphId: string): Promise<AutoODResponse> => {
    const res = await fetch(`${API_BASE}/graph/${graphId}/auto-select-od`);
    if (!res.ok) throw new Error('Failed to auto-select route.');
    return res.json();
  }
};

export const resilienceApi = {
  analyzeRoute: async (req: ResilienceRequest): Promise<ResilienceResponse> => {
    const response = await fetch(`${API_BASE}/predict/resilience`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req)
    });
    if (!response.ok) {
      let errorMsg = 'Unable to calculate route.';
      try {
        const errData = await response.json();
        if (errData?.detail) errorMsg = errData.detail;
      } catch {}
      throw new Error(errorMsg);
    }
    return await response.json();
  }
};

export const systemApi = {
  checkHealth: async (): Promise<boolean> => {
    try {
      const res = await fetch(`${API_BASE}/health`, { method: 'GET' });
      return res.ok;
    } catch {
      return false;
    }
  }
};

