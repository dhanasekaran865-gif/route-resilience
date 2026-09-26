import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import App from '../App';

describe('App Integration', () => {
  it('renders application sections', () => {
    render(<App />);
    expect(screen.getByText('ROUTE RESILIENCE')).toBeDefined();
    expect(screen.getByText('Graph Input')).toBeDefined();
    expect(screen.getByText('ROUTE PLANNING')).toBeDefined();
    expect(screen.getByText('RUN LONDON DEMO')).toBeDefined();
  });
});
