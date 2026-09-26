import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import App from '../App';
import '@testing-library/jest-dom';

vi.mock('../api', () => ({
  graphApi: {
    uploadGraph: vi.fn(),
    demoHealing: vi.fn(),
    healGraph: vi.fn(),
  },
  resilienceApi: {
    analyzeRoute: vi.fn(),
  },
  systemApi: {
    checkHealth: vi.fn().mockResolvedValue(true),
  }
}));

describe('Route Planning Coordinate Input', () => {
  it('renders Route Planning with manual coordinate inputs and action buttons', () => {
    render(<App />);

    expect(screen.getByText('ROUTE PLANNING')).toBeDefined();
    expect(screen.getByText('Origin (Lat, Lon):')).toBeDefined();
    expect(screen.getByText('Destination (Lat, Lon):')).toBeDefined();

    // Map selection buttons should NOT be present
    expect(screen.queryByText('Select on Map')).toBeNull();
    expect(screen.queryByText('AUTO-SELECT HEALED-ROAD ROUTE')).toBeNull();

    // Action buttons present
    expect(screen.getByText('NORMAL CONDITIONS')).toBeDefined();
    expect(screen.getByText('🌊 SIMULATE FLOOD')).toBeDefined();
  });

  it('updates origin coordinates upon user input', () => {
    render(<App />);

    const originInput = screen.getByPlaceholderText('51.51615, -0.12268') as HTMLInputElement;
    expect(originInput.value).toBe('51.51615, -0.12268');

    fireEvent.change(originInput, { target: { value: '51.52, -0.13' } });
    expect(originInput.value).toBe('51.52, -0.13');
  });

  it('updates destination coordinates upon user input', () => {
    render(<App />);

    const destInput = screen.getByPlaceholderText('51.50969, -0.12336') as HTMLInputElement;
    expect(destInput.value).toBe('51.50969, -0.12336');

    fireEvent.change(destInput, { target: { value: '51.505, -0.12' } });
    expect(destInput.value).toBe('51.505, -0.12');
  });
});
