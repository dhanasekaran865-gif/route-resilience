import React from 'react';
import type { ResilienceResponse } from '../api';
import { Waves, CheckCircle2, AlertTriangle, ShieldCheck, HelpCircle } from 'lucide-react';

interface ComparisonProps {
  data: ResilienceResponse;
  isDisrupted?: boolean;
}

export const ComparisonPanel: React.FC<ComparisonProps> = ({ data, isDisrupted = false }) => {
  const norm = data?.normal;
  const dis = data?.disrupted;
  const resilience = data?.resilience;
  const impact = data?.disruption_impact;
  const disruptionGeom = data?.disruptions && data.disruptions.length > 0 ? data.disruptions[0] : null;

  return (
    <div className="space-y-6">
      {/* ========================================================================= */}
      {/* PHASE 8: FLOOD SCENARIO PANEL                                             */}
      {/* ========================================================================= */}
      <div className={`p-4 rounded-lg border transition-all ${
        isDisrupted
          ? 'bg-amber-50 border-amber-300 shadow-sm'
          : 'bg-slate-50 border-slate-200'
      }`}>
        <div className="flex items-center justify-between mb-3 border-b border-amber-200/60 pb-2">
          <div className="flex items-center gap-2">
            <Waves className={isDisrupted ? 'text-amber-700 animate-pulse' : 'text-slate-400'} size={20} />
            <h3 className="font-bold text-sm tracking-wide uppercase text-slate-800">
              Flood Scenario (Phase 8)
            </h3>
          </div>
          <span className={`text-xs font-bold px-2.5 py-0.5 rounded-full border ${
            isDisrupted
              ? 'bg-red-100 text-red-800 border-red-300'
              : 'bg-slate-100 text-slate-600 border-slate-300'
          }`}>
            {isDisrupted ? 'Road disruption simulated' : 'Scenario Inactive'}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          <div>
            <span className="text-slate-500 block">Disruption Type:</span>
            <span className="font-semibold text-slate-800">
              {isDisrupted ? 'Simulated flood-induced road disruption' : 'None (Normal Conditions)'}
            </span>
          </div>
          <div>
            <span className="text-slate-500 block">Affected OSM Segment:</span>
            <span className="font-mono font-semibold text-slate-800">
              {isDisrupted && disruptionGeom ? disruptionGeom.segment_id : (isDisrupted ? 'Midpoint Route Segment' : 'None')}
            </span>
          </div>
          <div>
            <span className="text-slate-500 block">Network Response:</span>
            <span className="font-semibold text-slate-800">
              {isDisrupted
                ? (resilience?.reachable ? 'Alternative route rerouted successfully' : 'Unreachable')
                : 'Primary path active'}
            </span>
          </div>
        </div>

        <div className="mt-2.5 pt-2 border-t border-amber-200/50 text-[11px] text-amber-900/80 italic">
          Important Scientific Limitation: This is a simulated flood-induced road disruption on an actual OSM road segment. It demonstrates topological rerouting and route resilience under network failure, not physical flood depth or hydrology.
        </div>
      </div>

      {/* ========================================================================= */}
      {/* PHASE 10: BEFORE / AFTER FLOOD COMPARISON TABLE                           */}
      {/* ========================================================================= */}
      <div className="bg-white p-5 rounded-lg shadow-sm border border-gray-200">
        <div className="flex items-center justify-between mb-4 border-b pb-2">
          <h2 className="text-base font-bold text-slate-800 uppercase tracking-wide">
            Before vs After Flood Comparison (Phase 10)
          </h2>
          <span className="text-xs text-slate-500">
            Authoritative Backend Metrics
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="bg-slate-50 text-slate-700 border-b border-slate-200">
                <th className="py-2.5 px-3 font-semibold">METRIC</th>
                <th className="py-2.5 px-3 font-semibold text-blue-700">BEFORE FLOOD (NORMAL)</th>
                <th className="py-2.5 px-3 font-semibold text-purple-700">AFTER FLOOD (SIMULATED)</th>
                <th className="py-2.5 px-3 font-semibold text-slate-600">CHANGE / IMPACT</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              <tr>
                <td className="py-2.5 px-3 font-medium text-slate-600">Route Status</td>
                <td className="py-2.5 px-3 font-bold text-emerald-700">
                  <span className="inline-flex items-center gap-1">
                    <CheckCircle2 size={13} /> Open
                  </span>
                </td>
                <td className="py-2.5 px-3 font-bold text-red-700">
                  {isDisrupted ? (
                    <span className="inline-flex items-center gap-1">
                      <AlertTriangle size={13} /> Affected (Bypassed)
                    </span>
                  ) : (
                    <span className="text-slate-400 font-normal">N/A (Simulation Inactive)</span>
                  )}
                </td>
                <td className="py-2.5 px-3 font-semibold text-slate-700">
                  {isDisrupted ? 'Dynamic Reroute' : 'None'}
                </td>
              </tr>

              <tr>
                <td className="py-2.5 px-3 font-medium text-slate-600">Travel Time</td>
                <td className="py-2.5 px-3 font-semibold text-slate-800">
                  {norm ? `${norm.travel_time_min?.toFixed(0) ?? 'N/A'} min` : 'N/A'}
                </td>
                <td className="py-2.5 px-3 font-semibold text-slate-800">
                  {dis ? `${dis.travel_time_min?.toFixed(0) ?? 'N/A'} min` : 'N/A'}
                </td>
                <td className="py-2.5 px-3 font-bold text-red-600">
                  {isDisrupted && dis && resilience ? `+${resilience.travel_time_increase_min?.toFixed(0) ?? 0} min` : '0 min'}
                </td>
              </tr>

              <tr>
                <td className="py-2.5 px-3 font-medium text-slate-600">Distance</td>
                <td className="py-2.5 px-3 font-semibold text-slate-800">
                  {norm ? `${norm.distance_km?.toFixed(1) ?? 'N/A'} km` : 'N/A'}
                </td>
                <td className="py-2.5 px-3 font-semibold text-slate-800">
                  {dis ? `${dis.distance_km?.toFixed(1) ?? 'N/A'} km` : 'N/A'}
                </td>
                <td className="py-2.5 px-3 font-bold text-amber-700">
                  {isDisrupted && dis && resilience ? `+${resilience.detour_distance_km?.toFixed(1) ?? 0} km` : '0.0 km'}
                </td>
              </tr>

              <tr>
                <td className="py-2.5 px-3 font-medium text-slate-600">Detour</td>
                <td className="py-2.5 px-3 font-semibold text-slate-800">0.0 km</td>
                <td className="py-2.5 px-3 font-bold text-amber-700">
                  {isDisrupted && dis && resilience ? `+${resilience.detour_distance_km?.toFixed(1) ?? 0} km` : 'N/A'}
                </td>
                <td className="py-2.5 px-3 font-semibold text-slate-700">
                  {isDisrupted && dis && resilience ? `+${resilience.detour_distance_km?.toFixed(1) ?? 0} km detour` : 'No detour'}
                </td>
              </tr>

              <tr>
                <td className="py-2.5 px-3 font-medium text-slate-600">Disrupted Edges</td>
                <td className="py-2.5 px-3 font-semibold text-slate-800">0</td>
                <td className="py-2.5 px-3 font-bold text-red-600">
                  {isDisrupted ? (impact?.affected_original_edges ?? 1) : 0}
                </td>
                <td className="py-2.5 px-3 font-semibold text-slate-700">
                  {isDisrupted ? `${impact?.affected_original_edges ?? 1} segment(s) closed` : '0 segments closed'}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* PHASE 12: ROUTE RESILIENCE PANEL                                          */}
      {/* ========================================================================= */}
      <div className="bg-white p-5 rounded-lg shadow-sm border border-gray-200">
        <div className="flex items-center justify-between mb-3 border-b pb-2">
          <div className="flex items-center gap-2">
            <ShieldCheck size={20} className="text-indigo-600" />
            <h2 className="text-base font-bold text-slate-800 uppercase tracking-wide">
              Route Resilience Evaluation (Phase 12)
            </h2>
          </div>
          <div className="text-right">
            <span
              className="text-sm font-bold text-indigo-700 cursor-help border-b border-dashed border-indigo-400"
              title="This is a heuristic prototype score based on configured network metrics; it is not a probability of survival or safety."
            >
              Prototype Resilience Score: {dis ? dis.resilience_score : (norm?.resilience_score ?? 100)}/100
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs mb-4">
          <div className="bg-slate-50 p-2.5 rounded border border-slate-200">
            <span className="text-slate-500 block">Detour Distance</span>
            <span className="font-bold text-slate-800 text-sm">
              {isDisrupted && dis && resilience ? `+${resilience.detour_distance_km?.toFixed(1) ?? 0} km` : '0.0 km'}
            </span>
          </div>
          <div className="bg-slate-50 p-2.5 rounded border border-slate-200">
            <span className="text-slate-500 block">Travel Time Increase</span>
            <span className="font-bold text-slate-800 text-sm">
              {isDisrupted && dis && resilience ? `+${resilience.travel_time_increase_min?.toFixed(0) ?? 0} min` : '0 min'}
            </span>
          </div>
          <div className="bg-slate-50 p-2.5 rounded border border-slate-200">
            <span className="text-slate-500 block">Destination Reachable</span>
            <span className="font-bold text-emerald-700 text-sm">
              {resilience ? (resilience.reachable ? 'Yes (Viable alternative)' : 'No') : 'Yes'}
            </span>
          </div>
          <div className="bg-slate-50 p-2.5 rounded border border-slate-200">
            <span className="text-slate-500 block">Alternative Routes Left</span>
            <span className="font-bold text-indigo-700 text-sm">
              {resilience?.alternative_routes_remaining ?? (norm?.alternative_routes ?? 1)} available
            </span>
          </div>
        </div>

        <div className="bg-indigo-50/70 border border-indigo-100 p-3 rounded text-xs text-indigo-950 space-y-1">
          <div className="flex items-start gap-1.5 font-medium">
            <HelpCircle size={15} className="text-indigo-600 shrink-0 mt-0.5" />
            <span>
              Resilience indicates how well the route/network continues to provide viable travel options when conditions change or roads are disrupted.
            </span>
          </div>
          <p className="text-[11px] text-indigo-800/80 pl-5">
            Note: The prototype resilience score is a deterministic heuristic computed from relative distance, detour ratio, congestion exposure, and intersection safety. It does not represent a probability of disaster survival or absolute safety guarantee.
          </p>
        </div>
      </div>
    </div>
  );
};
