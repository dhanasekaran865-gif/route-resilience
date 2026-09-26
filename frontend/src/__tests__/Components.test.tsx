import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { RouteCards } from '../components/RouteCards';
import { ComparisonPanel } from '../components/ComparisonPanel';
import type { ResilienceResponse } from '../api';
import '@testing-library/jest-dom';

const mockResilienceData: ResilienceResponse = {
  route_id: 'test-route-1',
  normal: {
    distance_km: 3.2,
    travel_time_min: 7.5,
    average_congestion: 0.35,
    max_congestion: 0.6,
    safety_risk_exposure: 0.15,
    weather_risk_exposure: 0.2,
    disrupted_edges: 0,
    alternative_routes: 2,
    resilience_score: 88,
    geometry: {
      type: 'LineString',
      coordinates: [[-0.12, 51.51], [-0.125, 51.505]]
    }
  },
  disrupted: {
    distance_km: 4.1,
    travel_time_min: 10.2,
    average_congestion: 0.42,
    max_congestion: 0.7,
    safety_risk_exposure: 0.18,
    weather_risk_exposure: 0.22,
    disrupted_edges: 0,
    alternative_routes: 1,
    resilience_score: 79,
    geometry: {
      type: 'LineString',
      coordinates: [[-0.12, 51.51], [-0.122, 51.508], [-0.125, 51.505]]
    }
  },
  disruptions: [
    {
      segment_id: '12345 -> 67890',
      type: 'FLOODED',
      geometry: {
        type: 'LineString',
        coordinates: [[-0.121, 51.509], [-0.123, 51.507]]
      }
    }
  ],
  resilience: {
    reachable: true,
    detour_distance_km: 0.9,
    travel_time_increase_min: 2.7,
    alternative_routes_remaining: 1
  },
  disruption_impact: {
    affected_original_edges: 1,
    final_route_disrupted_edges: 0
  },
  comparison: {
    scenario: 'disrupted',
    selected_route: 'Alternative Route',
    reason: 'Dynamically rerouted around disrupted segment to maintain connectivity.'
  }
};

describe('RouteCards Component', () => {
  it('renders Route A, Route B, and Route C with correct metrics', () => {
    render(<RouteCards resilienceData={mockResilienceData} isDisrupted={false} />);

    expect(screen.getByText('Route A')).toBeDefined();
    expect(screen.getByText('Route B')).toBeDefined();
    expect(screen.getByText('Route C')).toBeDefined();

    // Route A normal metrics
    expect(screen.getByText('3.2 km')).toBeDefined();
    expect(screen.getByText('8 min')).toBeDefined();
    expect(screen.getByText('35%')).toBeDefined();

    // Route C displays N/A per Phase 5 guidelines
    expect(screen.getAllByText('N/A').length).toBeGreaterThan(0);
  });

  it('updates route badges when disruption is active', () => {
    render(<RouteCards resilienceData={mockResilienceData} isDisrupted={true} />);

    expect(screen.getByText('Affected by Flood')).toBeDefined();
    expect(screen.getByText('Active Alternative')).toBeDefined();
  });
});

describe('ComparisonPanel Component', () => {
  it('renders Flood Scenario, Before/After Table, and Resilience Evaluation', () => {
    render(<ComparisonPanel data={mockResilienceData} isDisrupted={true} />);

    // Phase 8 Flood Scenario
    expect(screen.getByText(/Flood Scenario/i)).toBeDefined();
    expect(screen.getAllByText(/Simulated flood-induced road disruption/i).length).toBeGreaterThan(0);
    expect(screen.getByText('12345 -> 67890')).toBeDefined();

    // Phase 10 Before vs After Table
    expect(screen.getByText(/Before vs After Flood Comparison/i)).toBeDefined();
    expect(screen.getByText('Route Status')).toBeDefined();
    expect(screen.getByText('Detour')).toBeDefined();
    expect(screen.getByText('Disrupted Edges')).toBeDefined();

    // Phase 12 Resilience Evaluation
    expect(screen.getByText(/Route Resilience Evaluation/i)).toBeDefined();
    expect(screen.getByText(/Prototype Resilience Score: 79\/100/i)).toBeDefined();
    expect(screen.getByText(/Resilience indicates how well the route\/network continues to provide viable travel options/i)).toBeDefined();
  });
});
