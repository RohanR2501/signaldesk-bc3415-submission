import type {
  ComplaintCategory,
  InvestigationSignal,
} from './api/contracts.generated';
type MetricName = InvestigationSignal['metric'];

export const categories: Record<ComplaintCategory, string> = {
  product_defect: 'Reported product defect',
  arrival_damage: 'Arrival damage',
  wrong_item: 'Wrong item',
  delivery_issue: 'Delivery issue',
  service_return_issue: 'Service & returns',
};
export const number = (value: number) =>
  new Intl.NumberFormat('en-SG').format(value);
export const percent = (value: number | null) =>
  value === null
    ? 'Unavailable'
    : value > 0 && value < 0.001
      ? '<0.1%'
      : `${(value * 100).toFixed(1)}%`;
export const date = (value: string | null) =>
  value === null
    ? 'Date unavailable'
    : new Intl.DateTimeFormat('en-SG', {
        day: 'numeric',
        month: 'short',
        year: 'numeric',
        timeZone: 'UTC',
      }).format(new Date(value));
export const productName = (id: string) => id.replace(/^amazon2023:/, '');
export const metricName = (metric: MetricName) =>
  metric === 'negative_rating' ? 'Negative rating burden' : categories[metric];
export const workbenchLink = (
  product: string,
  signal?: string,
  review?: string,
) =>
  `#/incidents?${new URLSearchParams({ product, ...(signal ? { signal } : {}), ...(review ? { review } : {}) })}`;

export const shareChange = (value: number | null) =>
  value === null
    ? 'Not comparable'
    : `${value > 0 ? '+' : ''}${(value * 100).toFixed(1)} pp`;

export const periodLabel = (
  start: string,
  granularity: 'month' | 'quarter',
) => {
  const value = new Date(start);
  return granularity === 'quarter'
    ? `Q${Math.floor(value.getUTCMonth() / 3) + 1} ${value.getUTCFullYear()}`
    : new Intl.DateTimeFormat('en-SG', {
        month: 'short',
        year: 'numeric',
        timeZone: 'UTC',
      }).format(value);
};
