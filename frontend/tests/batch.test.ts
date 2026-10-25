import assert from 'node:assert/strict';
import test from 'node:test';
import type { AnalyzeResponse } from '../src/api/contracts.generated.ts';
import {
  batchTotals,
  complaintKeys,
  exportBatchCsv,
  MAX_CSV_BYTES,
  parseReviewCsv,
  processBatch,
  readReviewCsv,
  type BatchItem,
} from '../src/views/batch.ts';

const reviews = parseReviewCsv(
  'review_id,text\r\na,"Works well 😀, thanks"\r\nb,"It arrived damaged.\nSecond line with ""quotes""."\r\nc,Not used',
);
const partial: AnalyzeResponse = {
  schema_version: '1.1',
  review_id: 'submitted:test-fixture',
  analyzed_at: '2026-01-01T00:00:00Z',
  status: 'partial',
  sentiment: {
    state: 'unavailable',
    label: null,
    model_version: null,
    device: null,
    manifest_sha256: null,
    artifact_sha256: {},
    preprocessing: null,
    max_tokens: null,
    fallback_reason: 'artifact_missing',
    attempts: [],
    input_provenance: 'submitted_review_text_only',
    score_state: 'not_returned_by_saved_label_interface',
    calibration: 'uncalibrated',
  },
  complaints: complaintKeys.map((category) => ({
    category,
    state: category === 'arrival_damage' ? 'detected' : 'unknown',
    method: 'explicit_rules',
    rule_version: 'test-fixture',
    taxonomy_version: 'test-fixture',
    evidence:
      category === 'arrival_damage'
        ? [
            {
              review_id: 'submitted:test-fixture',
              reviewed_at: null,
              start: 3,
              end: 18,
              text: 'arrived damaged',
              method: 'rule_match',
              rule_id: 'arrival.condition',
              resolution: 'not_established',
              resolution_excerpt: null,
              resolution_start: null,
              resolution_end: null,
              is_model_attribution: false,
            },
          ]
        : [],
    reason: 'test fixture',
    reliability_warning: 'Synthetic unit fixture, not model evidence.',
    learned_model_state: 'not_used',
  })),
  warnings: [],
};

test('quoted CSV preserves original IDs, commas, Unicode, multiline text and escaped quotes', () => {
  assert.equal(reviews.length, 3);
  assert.equal(reviews[0].text, 'Works well 😀, thanks');
  assert.equal(
    reviews[1].text,
    'It arrived damaged.\nSecond line with "quotes".',
  );
  assert.deepEqual(parseReviewCsv('\uFEFFtext\r\n"  😀\r\noriginal  "\r\n'), [
    { row: 1, review_id: '', text: '  😀\r\noriginal  ' },
  ]);
});

test('rejects missing/duplicate headers, blank text, malformed quotes, mismatched columns and excess rows', () => {
  for (const csv of [
    'message\nhello',
    'text,text\na,b',
    'text\n""',
    'text\n"unfinished',
    'text\n"closed"junk',
    'text\na,b',
    'text\n"   "',
    'text\na"b',
  ])
    assert.throws(() => parseReviewCsv(csv));
  assert.equal(
    parseReviewCsv('text\n' + Array(100).fill('fine').join('\n')).length,
    100,
  );
  assert.throws(() =>
    parseReviewCsv('text\n' + Array(101).fill('fine').join('\n')),
  );
  assert.throws(() => parseReviewCsv('text\n' + '😀'.repeat(50_001)));
});

test('file validation rejects large, non-CSV and invalid UTF-8 input', async () => {
  await assert.rejects(readReviewCsv(new File(['text\nx'], 'review.txt')));
  await assert.rejects(
    readReviewCsv(new File([new Uint8Array(MAX_CSV_BYTES + 1)], 'large.csv')),
  );
  await assert.rejects(
    readReviewCsv(new File([new Uint8Array([0xff])], 'bad.csv')),
  );
});

test('requests stay sequential; cancellation retains the current response and stops remaining rows', async () => {
  const completed: BatchItem[] = [];
  let concurrent = 0,
    peak = 0,
    stop = false;
  await processBatch(
    reviews,
    async () => {
      concurrent++;
      peak = Math.max(peak, concurrent);
      await new Promise((resolve) => setTimeout(resolve, 2));
      concurrent--;
      return partial;
    },
    (item) => {
      completed.push(item);
      stop = completed.length === 2;
    },
    () => stop,
    new AbortController().signal,
  );
  assert.equal(peak, 1);
  assert.equal(completed.length, 2);
  assert.equal(completed[1].response, partial);
});

test('failed requests stop dispatch without retry; unmounted requests do not publish results', async () => {
  let calls = 0;
  const failed: BatchItem[] = [];
  await processBatch(
    reviews,
    async () => {
      calls++;
      throw new Error('connection ended');
    },
    (item) => failed.push(item),
    () => false,
    new AbortController().signal,
  );
  assert.equal(calls, 1);
  assert.equal(failed[0].status, 'failed');
  const controller = new AbortController();
  await processBatch(
    reviews,
    async () => {
      controller.abort();
      return partial;
    },
    () => assert.fail('aborted response published'),
    () => false,
    controller.signal,
  );
});

test('aggregates partial responses without counting failed/unprocessed rows as unavailable predictions', () => {
  const items: BatchItem[] = [
    { ...reviews[0], status: 'complete', response: partial },
    { ...reviews[1], status: 'failed' },
    { ...reviews[2], status: 'unprocessed' },
  ];
  const totals = batchTotals(items);
  assert.equal(totals.processed, 1);
  assert.equal(totals.failed, 1);
  assert.equal(totals.unprocessed, 1);
  assert.equal(totals.sentiment.unavailable, 1);
  assert.equal(totals.sentiment.negative, 0);
  assert.equal(totals.complaints.arrival_damage, 1);
  assert.equal(totals.complaints.product_defect, 0);
});

test('export quotes original multiline text, preserves states and explicitly marks reversible formula guards', () => {
  const csv = exportBatchCsv([
    {
      row: 1,
      review_id: '=SUM(1,2)',
      text: '  @formula\n"😀"',
      status: 'complete',
      response: partial,
    },
    { ...reviews[2], status: 'unprocessed' },
  ]);
  const roundtrip = parseReviewCsv(csv);
  assert.equal(roundtrip[0].review_id, "'=SUM(1,2)");
  assert.equal(roundtrip[0].text, '\'  @formula\n"😀"');
  assert.ok(csv.includes('"review_id;text"'));
  assert.ok(csv.includes('"unavailable"'));
  assert.ok(csv.includes('"unknown"'));
  assert.ok(csv.includes('"not_analyzed"'));
  assert.equal(roundtrip[1].text, reviews[2].text);
});
