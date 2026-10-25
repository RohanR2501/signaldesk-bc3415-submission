import type {
  AnalyzeResponse,
  ComplaintCategory,
} from '../api/contracts.generated';

export const MAX_BATCH_ROWS = 100;
export const MAX_CSV_BYTES = 2 * 1024 * 1024;
export const complaintKeys: ComplaintCategory[] = [
  'product_defect',
  'arrival_damage',
  'wrong_item',
  'delivery_issue',
  'service_return_issue',
];
export type CsvReview = { row: number; review_id: string; text: string };
export type BatchItem = CsvReview & {
  status: 'unprocessed' | 'complete' | 'failed';
  response?: AnalyzeResponse;
  error?: string;
};

/** Strict comma-separated CSV. Quoted newlines and original field bytes are retained. */
export function parseReviewCsv(source: string): CsvReview[] {
  const records: string[][] = [];
  let fields: string[] = [],
    value = '',
    quoted = false,
    closed = false,
    touched = false;
  const field = () => {
    fields.push(value);
    value = '';
    closed = false;
  };
  const record = () => {
    // Ignore physical blank lines, but validate explicit empty fields such as "".
    if (touched || fields.length || value) {
      field();
      records.push(fields);
      if (records.length > MAX_BATCH_ROWS + 1)
        throw new Error('Use at most 100 review rows.');
    }
    fields = [];
    touched = false;
  };
  const input = source.replace(/^\uFEFF/, '');
  for (let i = 0; i < input.length; i++) {
    const char = input[i];
    if (quoted) {
      if (char === '"') {
        if (input[i + 1] === '"') {
          value += '"';
          i++;
        } else {
          quoted = false;
          closed = true;
        }
      } else value += char;
    } else if (char === ',') {
      field();
      touched = true;
    } else if (char === '\r' || char === '\n') {
      record();
      if (char === '\r' && input[i + 1] === '\n') i++;
    } else if (closed)
      throw new Error(
        'Unexpected text after a closing quote. Check CSV quoting.',
      );
    else if (char === '"') {
      if (value)
        throw new Error(
          'Quotes inside fields must be escaped and the field quoted.',
        );
      quoted = true;
      touched = true;
    } else {
      value += char;
      touched = true;
    }
  }
  if (quoted) throw new Error('An opening quote has no closing quote.');
  record();
  if (!records.length) throw new Error('The CSV is empty.');
  const headers = records.shift()!.map((name) => name.trim());
  if (headers.some((name) => !name) || new Set(headers).size !== headers.length)
    throw new Error('Column headers must be nonempty and unique.');
  const textIndex = headers.indexOf('text'),
    idIndex = headers.indexOf('review_id');
  if (textIndex < 0)
    throw new Error('Include a column named text (case-sensitive).');
  if (!records.length) throw new Error('Include at least one review row.');
  return records.map((cells, index) => {
    const row = index + 1;
    if (cells.length !== headers.length)
      throw new Error(`Data row ${row} has a different number of columns.`);
    const text = cells[textIndex];
    if (!text.trim() || Array.from(text).length > 50_000 || text.includes('\0'))
      throw new Error(
        `Data row ${row}: text must be nonblank, without NUL, and at most 50,000 Unicode characters.`,
      );
    return { row, review_id: idIndex < 0 ? '' : cells[idIndex], text };
  });
}

export async function readReviewCsv(file: File): Promise<CsvReview[]> {
  if (!file.name.toLowerCase().endsWith('.csv'))
    throw new Error('Choose a .csv file encoded as UTF-8.');
  if (!file.size || file.size > MAX_CSV_BYTES)
    throw new Error('Choose a nonempty CSV no larger than 2 MiB.');
  let text: string;
  try {
    text = new TextDecoder('utf-8', { fatal: true }).decode(
      await file.arrayBuffer(),
    );
  } catch {
    throw new Error(
      'Could not read UTF-8 CSV. Save the file as CSV UTF-8 and try again.',
    );
  }
  return parseReviewCsv(text);
}

/** Await each request. Stop on transport failure; never retry an uncertain POST. */
export async function processBatch(
  reviews: CsvReview[],
  analyze: (text: string, signal: AbortSignal) => Promise<AnalyzeResponse>,
  onItem: (item: BatchItem) => void,
  shouldStop: () => boolean,
  signal: AbortSignal,
): Promise<void> {
  for (const row of reviews) {
    if (shouldStop() || signal.aborted) break;
    try {
      const response = await analyze(row.text, signal);
      if (signal.aborted) break;
      onItem({ ...row, status: 'complete', response });
    } catch (error) {
      if (!signal.aborted)
        onItem({
          ...row,
          status: 'failed',
          error:
            error instanceof Error ? error.message : 'Analysis request failed.',
        });
      break;
    }
  }
}

export function batchTotals(items: BatchItem[]) {
  const sentiment = {
    positive: 0,
    neutral: 0,
    negative: 0,
    unavailable: 0,
    unknown: 0,
    abstained: 0,
  };
  const complaints = Object.fromEntries(
    complaintKeys.map((key) => [key, 0]),
  ) as Record<ComplaintCategory, number>;
  let processed = 0,
    failed = 0;
  for (const item of items) {
    if (item.status === 'failed') failed++;
    if (item.status !== 'complete' || !item.response) continue;
    processed++;
    const prediction = item.response.sentiment;
    if (prediction.state === 'detected' && prediction.label)
      sentiment[prediction.label]++;
    else if (prediction.state !== 'detected') sentiment[prediction.state]++;
    for (const complaint of item.response.complaints)
      if (complaint.state === 'detected') complaints[complaint.category]++;
  }
  return {
    processed,
    failed,
    unprocessed: items.length - processed - failed,
    sentiment,
    complaints,
  };
}

/** Formula protection is explicit and reversible via spreadsheet_sanitized_fields. */
export function exportBatchCsv(items: BatchItem[]): string {
  const headers = [
    'input_row',
    'review_id',
    'text',
    'processing_status',
    'error',
    'api_status',
    'analysis_review_id',
    'analyzed_at',
    'sentiment_state',
    'sentiment_label',
    'model_version',
    'device',
    'fallback_reason',
    'sentiment_provenance_json',
    ...complaintKeys.map((key) => `${key}_state`),
    'complaints_json',
    'warnings_json',
    'spreadsheet_sanitized_fields',
  ];
  const quote = (cell: string) => `"${cell.replaceAll('"', '""')}"`;
  const lines = [headers.map(quote).join(',')];
  for (const item of items) {
    const result = item.response,
      sentiment = result?.sentiment;
    const cells = [
      String(item.row),
      item.review_id,
      item.text,
      item.status,
      item.error ?? '',
      result?.status ?? '',
      result?.review_id ?? '',
      result?.analyzed_at ?? '',
      sentiment?.state ?? 'not_analyzed',
      sentiment?.label ?? '',
      sentiment?.model_version ?? '',
      sentiment?.device ?? '',
      sentiment?.fallback_reason ?? '',
      sentiment ? JSON.stringify(sentiment) : '',
      ...complaintKeys.map(
        (key) =>
          result?.complaints.find((c) => c.category === key)?.state ??
          'not_analyzed',
      ),
      result ? JSON.stringify(result.complaints) : '',
      result ? JSON.stringify(result.warnings) : '',
    ];
    const sanitized: string[] = [];
    const safe = cells.map((cell, index) => {
      if (/^[\s\uFEFF]*[=+\-@]/u.test(cell) || /^[\t\r\n]/u.test(cell)) {
        sanitized.push(headers[index]);
        return `'${cell}`;
      }
      return cell;
    });
    lines.push([...safe, sanitized.join(';')].map(quote).join(','));
  }
  return '\uFEFF' + lines.join('\r\n') + '\r\n';
}
