import { render, screen, fireEvent } from '@testing-library/react';
import { describe, test, expect, beforeEach, vi } from 'vitest';
import App from './App';
import { graphApi, resilienceApi } from './api';
import '@testing-library/jest-dom';

vi.mock('./api', () => ({
  graphApi: {
    uploadGraph: vi.fn(),
    demoHealing: vi.fn(),
    healGraph: vi.fn(),
  },
  resilienceApi: {
    analyzeRoute: vi.fn(),
  }
}));

describe('App End-to-End Flow', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test('renders header and components', () => {
    render(<App />);
    expect(screen.getByText(/ROUTE RESILIENCE/i)).toBeDefined();
    expect(screen.getByText(/RUN LONDON DEMO/i)).toBeDefined();
  });

  test('handles london demo loading', async () => {
    (graphApi.demoHealing as any).mockResolvedValue({
      graph_id: 'test-123',
      original: { nodes: 100, edges: 200 },
      damage: { removed_edges: [] },
      healing: { candidate_edges: [], validated_edges: [], rejected_edges: [] },
      evaluation: { precision: 1.0, recall: 1.0, f1: 1.0, true_positives: 1, false_positives: 0, false_negatives: 0 },
      healed: { nodes: 100, edges: 200 },
      diagnostics: {
        original_edge_count: 200,
        damaged_edge_count: 200,
        removed_edge_count: 0,
        accepted_healed_edge_count: 0,
        healed_edge_count: 200
      }
    });

    render(<App />);
    const demoBtn = screen.getByText(/RUN LONDON DEMO/i);
    fireEvent.click(demoBtn);
    
    expect(await screen.findByText(/Healing graph.../i)).toBeDefined();
    expect(await screen.findByText(/Original graph edges/i)).toBeDefined();
  });

  test('handles route calculation failure', async () => {
    (resilienceApi.analyzeRoute as any).mockRejectedValue(new Error('Failed'));
    
    render(<App />);
    const analyzeBtn = screen.getByText(/NORMAL CONDITIONS/i);
    fireEvent.click(analyzeBtn);
    
    expect(await screen.findByText(/Unable to calculate route./i)).toBeDefined();
  });

  test('displays missing geometry warning', async () => {
    (resilienceApi.analyzeRoute as any).mockResolvedValue({
      normal: { distance_km: 1, travel_time_min: 1, geometry: null }
    });
    
    render(<App />);
    const analyzeBtn = screen.getByText(/NORMAL CONDITIONS/i);
    fireEvent.click(analyzeBtn);
    
    expect(await screen.findByText(/Graph geometry is unavailable./i)).toBeDefined();
  });
});
