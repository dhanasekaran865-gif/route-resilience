import React from 'react';
import type { RouteMetrics, ResilienceResponse } from '../api';
import { Info, ShieldAlert, CloudRain, Clock, Route, AlertOctagon, CheckCircle2, CornerDownRight } from 'lucide-react';

interface RouteCardItemProps {
  name: string;
  subtitle: string;
  statusBadge: {
    label: string;
    variant: 'green' | 'amber' | 'red' | 'purple' | 'slate';
  };
  metrics?: RouteMetrics | null;
  detourKm?: number | null;
  disruptedEdgesCount?: number | null;
  altRoutesRemaining?: number | null;
  isActive?: boolean;
}

const badgeStyles: Record<string, string> = {
  green: 'bg-emerald-100 text-emerald-800 border-emerald-300',
  amber: 'bg-amber-100 text-amber-800 border-amber-300',
  red: 'bg-red-100 text-red-800 border-red-300 animate-pulse',
  purple: 'bg-purple-100 text-purple-800 border-purple-300',
  slate: 'bg-slate-100 text-slate-700 border-slate-300',
};

export const RouteCardItem: React.FC<RouteCardItemProps> = ({
  name,
  subtitle,
  statusBadge,
  metrics,
  detourKm,
  disruptedEdgesCount,
  altRoutesRemaining,
  isActive
}) => {
  const formatVal = (val: number | undefined | null, suffix = '') =>
    val !== undefined && val !== null && !isNaN(val) ? `${val.toFixed(1)}${suffix}` : 'N/A';

  const formatInt = (val: number | undefined | null) =>
    val !== undefined && val !== null && !isNaN(val) ? `${Math.round(val)}` : 'N/A';

  return (
    <div className={`p-4 rounded-lg border transition-all ${
      isActive
        ? 'bg-white shadow-md border-indigo-500 ring-2 ring-indigo-200'
        : 'bg-white shadow-sm border-gray-200 opacity-95'
    }`}>
      {/* Header */}
      <div className="flex justify-between items-start mb-3 border-b pb-2">
        <div>
          <h3 className="font-bold text-base text-slate-800">{name}</h3>
          <p className="text-xs text-slate-500">{subtitle}</p>
        </div>
        <span className={`text-[11px] font-semibold px-2 py-0.5 rounded border ${badgeStyles[statusBadge.variant]}`}>
          {statusBadge.label}
        </span>
      </div>

      {/* Metrics List */}
      <div className="space-y-2 text-xs">
        <div className="flex justify-between items-center py-0.5">
          <span className="text-gray-500 flex items-center gap-1.5"><Clock size={13} className="text-slate-400" /> Travel Time</span>
          <span className="font-bold text-gray-800">{metrics ? `${formatInt(metrics.travel_time_min)} min` : 'N/A'}</span>
        </div>

        <div className="flex justify-between items-center py-0.5">
          <span className="text-gray-500 flex items-center gap-1.5"><Route size={13} className="text-slate-400" /> Distance</span>
          <span className="font-semibold text-gray-800">{metrics ? `${formatVal(metrics.distance_km)} km` : 'N/A'}</span>
        </div>

        <div className="flex justify-between items-center py-0.5">
          <span className="text-gray-500 flex items-center gap-1.5"><Clock size={13} className="text-slate-400" /> Average Congestion</span>
          <span className="font-semibold text-gray-800">{metrics ? `${(metrics.average_congestion * 100).toFixed(0)}%` : 'N/A'}</span>
        </div>

        <div className="flex justify-between items-center py-0.5 group relative">
          <span className="text-gray-500 flex items-center gap-1.5">
            <ShieldAlert size={13} className="text-amber-500" /> Safety Risk Exposure
            <Info size={11} className="text-gray-400 cursor-pointer" />
          </span>
          <span className="font-semibold text-blue-700">{metrics ? `${(metrics.safety_risk_exposure * 100).toFixed(1)}` : 'N/A'}</span>
          <div className="hidden group-hover:block absolute bottom-full left-0 mb-1 w-56 bg-slate-800 text-white text-[10px] p-2 rounded shadow-lg z-20">
            Based on historical segment features and intersection density. Does not guarantee absolute safety.
          </div>
        </div>

        <div className="flex justify-between items-center py-0.5">
          <span className="text-gray-500 flex items-center gap-1.5"><CloudRain size={13} className="text-sky-500" /> Weather Exposure</span>
          <span className="font-semibold text-blue-700">{metrics ? `${(metrics.weather_risk_exposure * 100).toFixed(1)}` : 'N/A'}</span>
        </div>

        <div className="flex justify-between items-center py-0.5 border-t pt-1.5 mt-1">
          <span className="text-gray-500 flex items-center gap-1.5"><AlertOctagon size={13} className="text-red-500" /> Disrupted Edges</span>
          <span className={`font-bold ${disruptedEdgesCount && disruptedEdgesCount > 0 ? 'text-red-600' : 'text-gray-700'}`}>
            {disruptedEdgesCount !== undefined && disruptedEdgesCount !== null ? disruptedEdgesCount : 'N/A'}
          </span>
        </div>

        <div className="flex justify-between items-center py-0.5">
          <span className="text-gray-500 flex items-center gap-1.5"><CornerDownRight size={13} className="text-slate-400" /> Detour Distance</span>
          <span className={`font-bold ${detourKm && detourKm > 0 ? 'text-amber-700' : 'text-gray-700'}`}>
            {detourKm !== undefined && detourKm !== null ? (detourKm > 0 ? `+${detourKm.toFixed(1)} km` : '0.0 km') : 'N/A'}
          </span>
        </div>

        <div className="flex justify-between items-center py-0.5">
          <span className="text-gray-500 flex items-center gap-1.5"><CheckCircle2 size={13} className="text-emerald-500" /> Alternative Routes</span>
          <span className="font-semibold text-gray-800">
            {altRoutesRemaining !== undefined && altRoutesRemaining !== null ? `${altRoutesRemaining} available` : 'N/A'}
          </span>
        </div>
      </div>
    </div>
  );
};

export interface RouteCardsProps {
  metrics?: RouteMetrics;
  title?: string;
  resilienceData?: ResilienceResponse | null;
  isDisrupted?: boolean;
}

export const RouteCards: React.FC<RouteCardsProps> = ({
  metrics,
  title,
  resilienceData,
  isDisrupted = false
}) => {
  // If legacy single card format used
  if (metrics && title && !resilienceData) {
    return (
      <div className="bg-white p-5 rounded-lg shadow-sm border border-gray-100">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-bold text-slate-800">{title}</h2>
          <span className="text-sm bg-slate-100 text-slate-600 px-2 py-1 rounded font-medium">
            {metrics.travel_time_min.toFixed(0)} min &bull; {metrics.distance_km.toFixed(1)} km
          </span>
        </div>
        <div className="space-y-4">
          <div className="flex justify-between items-center border-b pb-2">
            <span className="text-gray-500 flex items-center gap-2"><Clock size={16}/> Congestion</span>
            <span className="font-semibold text-gray-800">{(metrics.average_congestion * 100).toFixed(0)}%</span>
          </div>
          <div className="flex justify-between items-center border-b pb-2">
            <span className="text-gray-500 flex items-center gap-2"><ShieldAlert size={16}/> Safety Risk</span>
            <span className="font-semibold text-gray-800">{(metrics.safety_risk_exposure * 100).toFixed(1)}</span>
          </div>
          <div className="flex justify-between items-center border-b pb-2">
            <span className="text-gray-500 flex items-center gap-2"><CloudRain size={16}/> Weather Risk</span>
            <span className="font-semibold text-gray-800">{(metrics.weather_risk_exposure * 100).toFixed(1)}</span>
          </div>
          <div className="flex justify-between items-center pb-2">
            <span className="text-gray-500 flex items-center gap-2"><Route size={16}/> Alternative Routes</span>
            <span className="font-semibold text-gray-800">{metrics.alternative_routes}</span>
          </div>
        </div>
      </div>
    );
  }

  const normal = resilienceData?.normal;
  const disrupted = resilienceData?.disrupted;
  const resilience = resilienceData?.resilience;
  const impact = resilienceData?.disruption_impact;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-wider text-slate-700">
          Route Alternatives (Phase 5)
        </h2>
        <span className="text-xs text-slate-500">
          Backend Evaluated Paths
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Route A: Primary Normal Route */}
        <RouteCardItem
          name="Route A"
          subtitle="Primary Normal Route"
          statusBadge={
            isDisrupted
              ? { label: 'Affected by Flood', variant: 'red' }
              : { label: normal ? 'Open' : 'Pending', variant: normal ? 'green' : 'slate' }
          }
          metrics={normal}
          detourKm={0}
          disruptedEdgesCount={isDisrupted ? (impact?.affected_original_edges ?? 1) : 0}
          altRoutesRemaining={normal?.alternative_routes}
          isActive={!isDisrupted && !!normal}
        />

        {/* Route B: Alternative Rerouted Path */}
        <RouteCardItem
          name="Route B"
          subtitle="Alternative Bypass Route"
          statusBadge={
            isDisrupted && disrupted
              ? { label: 'Active Alternative', variant: 'purple' }
              : { label: 'Standby Bypass', variant: 'slate' }
          }
          metrics={disrupted}
          detourKm={isDisrupted ? resilience?.detour_distance_km : null}
          disruptedEdgesCount={disrupted ? disrupted.disrupted_edges : null}
          altRoutesRemaining={resilience?.alternative_routes_remaining}
          isActive={isDisrupted && !!disrupted}
        />

        {/* Route C: Secondary Standby Route */}
        <RouteCardItem
          name="Route C"
          subtitle="Secondary Alternative"
          statusBadge={{ label: 'Standby / N/A', variant: 'slate' }}
          metrics={null}
          detourKm={null}
          disruptedEdgesCount={null}
          altRoutesRemaining={null}
          isActive={false}
        />
      </div>
    </div>
  );
};
