import { useEffect, useRef, useState } from 'react';
import { Download, LoaderCircle, ShieldCheck, Upload } from 'lucide-react';
import { api } from '../api/client';
import { Badge, ErrorNotice, SectionTitle } from '../components/ui';
import { categories } from '../lib';
import { ComplaintResult } from './ComplaintEvidence';
import {
  batchTotals,
  complaintKeys,
  exportBatchCsv,
  processBatch,
  readReviewCsv,
  type BatchItem,
} from './batch';

export function BatchAnalyzer({
  onBusyChange,
}: {
  onBusyChange: (busy: boolean) => void;
}) {
  const [items, setItems] = useState<BatchItem[]>([]);
  const [filename, setFilename] = useState('');
  const [error, setError] = useState<string>();
  const [reading, setReading] = useState(false);
  const [running, setRunning] = useState(false);
  const [hasRun, setHasRun] = useState(false);
  const [cancelled, setCancelled] = useState(false);
  const [search, setSearch] = useState('');
  const [outcome, setOutcome] = useState('all');
  const [category, setCategory] = useState('all');
  const active = useRef(true);
  const inFlight = useRef(false);
  const stop = useRef(false);
  const request = useRef<AbortController | null>(null);
  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
      request.current?.abort();
    };
  }, []);
  const totals = batchTotals(items);
  const busy = reading || running;
  const reset = () => {
    setItems([]);
    setFilename('');
    setError(undefined);
    setHasRun(false);
    setCancelled(false);
    setSearch('');
    setOutcome('all');
    setCategory('all');
  };
  const upload = async (file: File) => {
    if (inFlight.current) return;
    inFlight.current = true;
    reset();
    setReading(true);
    onBusyChange(true);
    try {
      const rows = await readReviewCsv(file);
      if (active.current) {
        setItems(rows.map((row) => ({ ...row, status: 'unprocessed' })));
        setFilename(file.name);
      }
    } catch (failure) {
      if (active.current)
        setError(
          failure instanceof Error
            ? failure.message
            : 'Could not read this CSV.',
        );
    } finally {
      inFlight.current = false;
      if (active.current) {
        setReading(false);
        onBusyChange(false);
      }
    }
  };
  const start = async () => {
    if (inFlight.current || !items.length || hasRun) return;
    inFlight.current = true;
    stop.current = false;
    setRunning(true);
    setHasRun(true);
    setCancelled(false);
    onBusyChange(true);
    const controller = new AbortController();
    request.current = controller;
    try {
      await processBatch(
        items,
        api.analyze,
        (item) => {
          setItems((current) =>
            current.map((row) => (row.row === item.row ? item : row)),
          );
        },
        () => stop.current,
        controller.signal,
      );
    } finally {
      inFlight.current = false;
      if (active.current) {
        setRunning(false);
        onBusyChange(false);
      }
    }
  };
  const download = () => {
    const url = URL.createObjectURL(
      new Blob([exportBatchCsv(items)], { type: 'text/csv;charset=utf-8' }),
    );
    const link = document.createElement('a');
    link.href = url;
    link.download = 'signaldesk-batch-results.csv';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const filtered = items.filter((item) => {
    const sentiment = item.response?.sentiment;
    return (
      `${item.review_id}\n${item.text}`
        .toLowerCase()
        .includes(search.toLowerCase()) &&
      (outcome === 'all' ||
        outcome === item.status ||
        outcome === sentiment?.label ||
        outcome === sentiment?.state) &&
      (category === 'all' ||
        item.response?.complaints.some(
          (c) => c.category === category && c.state === 'detected',
        ))
    );
  });
  return (
    <div className="view-stack batch-analyzer">
      <section className="panel batch-upload">
        <SectionTitle
          title="Analyse a review CSV"
          subtitle="One request at a time, using the same saved sentiment models and independent rules."
          action={<Upload size={20} aria-hidden="true" />}
        />
        <label htmlFor="batch-file" className="editor-label">
          Review CSV file
        </label>
        <input
          id="batch-file"
          type="file"
          accept=".csv,text/csv"
          disabled={busy}
          aria-describedby="batch-file-help"
          onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = '';
            if (file) void upload(file);
          }}
        />
        <p id="batch-file-help" className="muted">
          UTF-8 · up to 2 MiB / 100 reviews · required <code>text</code>,
          optional <code>review_id</code>. Quoted commas and multiline text are
          supported. Other columns are ignored; each data row is analysed
          separately.
        </p>
        {reading && <p role="status">Reading and validating CSV…</p>}
        {error && <ErrorNotice message={error} />}
        {!!items.length && !hasRun && (
          <>
            <p className="batch-file-summary">
              <strong>{filename}</strong> · {items.length} valid reviews · first{' '}
              {Math.min(3, items.length)} previewed
            </p>
            <ol className="batch-preview">
              {items.slice(0, 3).map((item) => (
                <li key={item.row}>
                  <small>
                    Data row {item.row} · {item.review_id || 'No supplied ID'}
                  </small>
                  <p>
                    {Array.from(item.text).slice(0, 180).join('')}
                    {Array.from(item.text).length > 180 && '…'}
                  </p>
                </li>
              ))}
            </ol>
          </>
        )}
        <div className="batch-actions">
          <button
            type="button"
            className="button primary"
            disabled={busy || !items.length || hasRun}
            onClick={() => void start()}
          >
            {running && (
              <LoaderCircle size={16} className="spinner" aria-hidden="true" />
            )}
            Analyse batch
          </button>
          {running && (
            <button
              type="button"
              className="button secondary"
              disabled={cancelled}
              onClick={() => {
                stop.current = true;
                setCancelled(true);
              }}
            >
              Cancel remaining
            </button>
          )}
          <button
            type="button"
            className="button secondary"
            disabled={busy || !items.length}
            onClick={reset}
          >
            Clear batch
          </button>
          <button
            type="button"
            className="button secondary"
            disabled={busy || !hasRun}
            onClick={download}
          >
            <Download size={16} aria-hidden="true" />
            Export CSV
          </button>
        </div>
        <div className="privacy-note">
          <ShieldCheck size={18} aria-hidden="true" />
          <p>
            Reviews stay in temporary browser memory and clear when you leave
            the analyzer or clear the batch. Each text is sent to your local
            analysis API. Results never enter historical product trends.
          </p>
        </div>
      </section>
      {hasRun && (
        <>
          <section className="panel batch-summary" aria-label="Batch summary">
            <SectionTitle title="Batch results" subtitle={filename} />
            <div role="status" aria-live="polite" className="batch-progress">
              <strong>
                {totals.processed} processed · {totals.failed} failed ·{' '}
                {totals.unprocessed} unprocessed / {items.length} uploaded
              </strong>
              {running ? (
                <p>
                  {cancelled
                    ? 'Cancellation requested. Waiting for the current review; remaining rows will not be sent.'
                    : `Processing data row ${totals.processed + totals.failed + 1}. First model loading can take several minutes; keep this view open.`}
                </p>
              ) : (
                <p>
                  {totals.unprocessed
                    ? 'Stopped. Remaining rows were not sent. Load a new CSV to start another batch.'
                    : 'Batch finished.'}
                </p>
              )}
              <progress
                aria-label="Reviews finished"
                max={items.length}
                value={totals.processed + totals.failed}
              />
            </div>
            {totals.failed > 0 && (
              <p className="input-error" role="alert">
                A request failed; further requests were stopped. The backend may
                still have processed that request. Inspect its error before
                starting another batch.
              </p>
            )}
            <dl className="batch-sentiments">
              {Object.entries(totals.sentiment)
                .filter(
                  ([key, value]) =>
                    !['unknown', 'abstained'].includes(key) || value > 0,
                )
                .map(([key, value]) => (
                  <div key={key}>
                    <dt>
                      {key === 'unavailable' ? 'Sentiment unavailable' : key}
                    </dt>
                    <dd>{value}</dd>
                  </div>
                ))}
            </dl>
            <h3>Independent rule-evidenced complaints</h3>
            <dl className="batch-complaints">
              {complaintKeys.map((key) => (
                <div key={key}>
                  <dt>{categories[key]}</dt>
                  <dd>
                    {totals.complaints[key]}{' '}
                    <small>/ {totals.processed} processed</small>
                  </dd>
                </div>
              ))}
            </dl>
            <p className="chart-note">
              Counts use completed API responses, including partial analyses;
              failed and unprocessed rows are excluded. Categories overlap. An
              unmatched category remains unknown; it does not establish an
              issue-free review. Reports are not verified incidents.
            </p>
          </section>
          <section className="panel batch-results">
            <SectionTitle
              title="Review-level results"
              subtitle="Original text, actual provenance and exact evidence. Export includes every uploaded row, regardless of filters."
            />
            <div className="batch-filters">
              <label>
                Search ID or text
                <input
                  type="search"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                />
              </label>
              <label>
                Outcome
                <select
                  value={outcome}
                  onChange={(event) => setOutcome(event.target.value)}
                >
                  <option value="all">All rows</option>
                  <option value="positive">Positive</option>
                  <option value="neutral">Neutral</option>
                  <option value="negative">Negative</option>
                  <option value="unavailable">Sentiment unavailable</option>
                  <option value="unknown">Sentiment unknown</option>
                  <option value="abstained">Sentiment abstained</option>
                  <option value="failed">Failed</option>
                  <option value="unprocessed">Unprocessed</option>
                </select>
              </label>
              <label>
                Detected complaint
                <select
                  value={category}
                  onChange={(event) => setCategory(event.target.value)}
                >
                  <option value="all">All categories</option>
                  {complaintKeys.map((key) => (
                    <option key={key} value={key}>
                      {categories[key]}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <p className="muted">
              {filtered.length} of {items.length} rows shown
            </p>
            <div
              className="table-scroll batch-table-scroll"
              role="region"
              aria-label="Batch review results"
              tabIndex={0}
            >
              <table>
                <thead>
                  <tr>
                    <th>Source review</th>
                    <th>Original text</th>
                    <th>Sentiment & provenance</th>
                    <th>Independent complaints</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((item) => (
                    <tr key={item.row}>
                      <th scope="row">
                        <small>Data row {item.row}</small>
                        <code>{item.review_id || 'No supplied ID'}</code>
                        <Badge
                          tone={item.status === 'failed' ? 'amber' : 'neutral'}
                        >
                          {item.status === 'complete'
                            ? item.response?.status
                            : item.status}
                        </Badge>
                      </th>
                      <td>
                        <details>
                          <summary>Read original text</summary>
                          <pre className="batch-original">{item.text}</pre>
                        </details>
                      </td>
                      <td>
                        {item.response ? (
                          <>
                            <strong>
                              {item.response.sentiment.label ??
                                item.response.sentiment.state}
                            </strong>
                            <p>
                              {item.response.sentiment.model_version ??
                                'No sentiment model available'}{' '}
                              · {item.response.sentiment.device ?? 'no device'}
                            </p>
                            {item.response.sentiment.fallback_reason && (
                              <p>
                                Fallback:{' '}
                                {item.response.sentiment.fallback_reason}
                              </p>
                            )}
                            <details>
                              <summary>Model provenance & limitations</summary>
                              <pre>
                                {JSON.stringify(
                                  item.response.sentiment,
                                  null,
                                  2,
                                )}
                              </pre>
                              <ul>
                                {item.response.warnings.map((warning) => (
                                  <li key={warning}>{warning}</li>
                                ))}
                              </ul>
                            </details>
                          </>
                        ) : (
                          <p>
                            {item.error ??
                              'Not analysed; no prediction supplied.'}
                          </p>
                        )}
                      </td>
                      <td>
                        {item.response ? (
                          <details>
                            <summary>
                              {
                                item.response.complaints.filter(
                                  (c) => c.state === 'detected',
                                ).length
                              }{' '}
                              categories detected · inspect all five
                            </summary>
                            {item.response.complaints.map((complaint) => (
                              <ComplaintResult
                                key={complaint.category}
                                complaint={complaint}
                                submitted={item.text}
                              />
                            ))}
                          </details>
                        ) : (
                          'Not analysed'
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {!filtered.length && (
                <p className="empty-message">No rows match these filters.</p>
              )}
            </div>
            <p className="chart-note">
              CSV export prefixes spreadsheet-formula-leading cells with an
              apostrophe. The spreadsheet_sanitized_fields column identifies
              adjusted fields; remove only that added apostrophe to recover the
              exact original value. Unprocessed/failed rows have no invented
              predictions.
            </p>
          </section>
        </>
      )}
    </div>
  );
}
