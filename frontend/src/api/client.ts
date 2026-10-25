import schema from '../../../docs/contracts/schema-v1.json';
import type {
  AnalyzeRequest,
  AnalyzeResponse,
  ExecutiveSummary,
  ProductDetail,
  ProductTimeline,
  Page_ProductSummary_,
  Page_InvestigationSignal_,
  InvestigationDetail,
  ReviewEvidenceResponse,
  ComplaintCategory,
  InvestigationSignal,
} from './contracts.generated';

type Schema = {
  $ref?: string;
  const?: unknown;
  enum?: unknown[];
  anyOf?: Schema[];
  type?: string;
  properties?: Record<string, Schema>;
  required?: string[];
  items?: Schema;
  prefixItems?: Schema[];
  additionalProperties?: Schema | boolean;
};
const definitions: Record<string, Schema> = schema.$defs;
const record = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

// Check response structure against the same exported contracts that generate our
// types. Incompatible data becomes an explicit error, never a zero-filled dashboard.
function matches(value: unknown, shape: Schema): boolean {
  if (shape.$ref)
    return matches(value, definitions[shape.$ref.split('/').at(-1)!]);
  if ('const' in shape) return value === shape.const;
  if (shape.enum) return shape.enum.includes(value);
  if (shape.anyOf) return shape.anyOf.some((part) => matches(value, part));
  if (shape.type === 'null') return value === null;
  if (shape.type === 'number' || shape.type === 'integer')
    return (
      typeof value === 'number' &&
      Number.isFinite(value) &&
      (shape.type !== 'integer' || Number.isInteger(value))
    );
  if (shape.type === 'string' || shape.type === 'boolean')
    return typeof value === shape.type;
  if (shape.type === 'array')
    return (
      Array.isArray(value) &&
      (shape.prefixItems
        ? value.length === shape.prefixItems.length &&
          shape.prefixItems.every((part, index) => matches(value[index], part))
        : value.every((item) => matches(item, shape.items ?? {})))
    );
  if (shape.type === 'object') {
    if (!record(value) || shape.required?.some((key) => !(key in value)))
      return false;
    return Object.entries(value).every(([key, item]) => {
      const part = shape.properties?.[key];
      if (part) return matches(item, part);
      if (shape.additionalProperties === false) return false;
      return (
        typeof shape.additionalProperties !== 'object' ||
        matches(item, shape.additionalProperties)
      );
    });
  }
  return true;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly requestId?: string,
  ) {
    super(message);
  }
}

async function request<T>(
  path: string,
  model: string,
  signal?: AbortSignal,
  body?: AnalyzeRequest,
): Promise<T> {
  const base = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');
  let response: Response;
  try {
    response = await fetch(`${base}/api/v1${path}`, {
      method: body ? 'POST' : 'GET',
      headers: {
        Accept: 'application/json',
        ...(body ? { 'Content-Type': 'application/json' } : {}),
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
      signal: AbortSignal.any([
        ...(signal ? [signal] : []),
        AbortSignal.timeout(body ? 600_000 : 20_000),
      ]),
      cache: 'no-store',
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new ApiError(
      body
        ? 'The analysis connection ended. The server may still be processing; check connectivity before retrying.'
        : 'Could not reach the backend. Check the local service and try again.',
    );
  }
  const requestId = response.headers.get('X-Request-ID') ?? undefined;
  if (!response.ok)
    throw new ApiError(
      response.status === 503
        ? 'The historical dataset is unavailable or failed its integrity checks.'
        : response.status === 422
          ? 'The request was rejected. Check the input or reset the filters.'
          : response.status === 404
            ? 'The requested resource was not found.'
            : `The backend could not complete the request (HTTP ${response.status}).`,
      requestId,
    );
  let data: unknown;
  try {
    data = await response.json();
  } catch {
    throw new ApiError('The backend returned unreadable data.', requestId);
  }
  if (
    !record(data) ||
    data.schema_version !== '1.1' ||
    !matches(data, definitions[model])
  )
    throw new ApiError(
      'The backend returned data incompatible with the API contract.',
      requestId,
    );
  return data as T;
}

type ListFilters = { limit?: number; cursor?: string };
type MetricName = InvestigationSignal['metric'];
export type ProductFilters = ListFilters & {
  q?: string;
  category?: ComplaintCategory;
  min_reviews?: number;
  has_signals?: boolean;
};
export type InvestigationFilters = ListFilters & {
  product_id?: string;
  metric?: MetricName;
  reason?: 'rising_share' | 'sustained_burden';
};
function query(values: object): string {
  const params = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => {
    if (value !== undefined && value !== '') params.set(key, String(value));
  });
  return params.size ? `?${params}` : '';
}
export const api = {
  summary: (signal?: AbortSignal) =>
    request<ExecutiveSummary>('/summary', 'ExecutiveSummary', signal),
  products: (filters: ProductFilters = {}, signal?: AbortSignal) =>
    request<Page_ProductSummary_>(
      `/products${query(filters)}`,
      'Page_ProductSummary_',
      signal,
    ),
  product: (id: string, signal?: AbortSignal) =>
    request<ProductDetail>(
      `/products/${encodeURIComponent(id)}`,
      'ProductDetail',
      signal,
    ),
  timeline: (
    id: string,
    filters: { from?: string; to?: string } = {},
    signal?: AbortSignal,
  ) =>
    request<ProductTimeline>(
      `/products/${encodeURIComponent(id)}/timeline${query(filters)}`,
      'ProductTimeline',
      signal,
    ),
  investigations: (filters: InvestigationFilters = {}, signal?: AbortSignal) =>
    request<Page_InvestigationSignal_>(
      `/investigations${query(filters)}`,
      'Page_InvestigationSignal_',
      signal,
    ),
  investigation: (id: string, signal?: AbortSignal) =>
    request<InvestigationDetail>(
      `/investigations/${encodeURIComponent(id)}`,
      'InvestigationDetail',
      signal,
    ),
  review: (id: string, signal?: AbortSignal) =>
    request<ReviewEvidenceResponse>(
      `/reviews/${encodeURIComponent(id)}`,
      'ReviewEvidenceResponse',
      signal,
    ),
  analyze: (text: string, signal?: AbortSignal) =>
    request<AnalyzeResponse>('/reviews/analyze', 'AnalyzeResponse', signal, {
      schema_version: '1.1',
      text,
    }),
};
