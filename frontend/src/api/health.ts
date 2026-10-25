import type { HealthResponse } from './contracts.generated';

export async function fetchHealth(
  signal: AbortSignal,
): Promise<HealthResponse> {
  const base = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');
  const response = await fetch(`${base}/api/v1/health`, {
    signal: AbortSignal.any([signal, AbortSignal.timeout(5000)]),
    headers: { Accept: 'application/json' },
    cache: 'no-store',
  });
  if (!response.ok)
    throw new Error(`Health check returned HTTP ${response.status}.`);
  const data: unknown = await response.json();
  if (
    typeof data !== 'object' ||
    data === null ||
    !('status' in data) ||
    data.status !== 'ok' ||
    !('service' in data) ||
    data.service !== 'signaldesk' ||
    !('api_version' in data) ||
    data.api_version !== 'v1' ||
    !('schema_version' in data) ||
    data.schema_version !== '1.1' ||
    !('phase' in data) ||
    data.phase !== 4 ||
    !('model_status' in data) ||
    data.model_status !== 'not_checked' ||
    !('version' in data) ||
    typeof data.version !== 'string'
  ) {
    throw new Error('The backend returned an incompatible health response.');
  }
  return data as HealthResponse;
}
