import { useEffect, useState, type FormEvent } from 'react';
import {
  ArrowDownRight,
  ArrowUpRight,
  ChevronLeft,
  ChevronRight,
  Database,
  Search,
  ShieldCheck,
  ScanText,
  Package,
  MessagesSquare,
  Info,
} from 'lucide-react';
import { api, type ProductFilters } from '../api/client';
import type {
  ExecutiveSummary,
  Page_InvestigationSignal_,
  Page_ProductSummary_,
} from '../api/contracts.generated';
import {
  categories,
  date,
  metricName,
  number,
  percent,
  productName,
  workbenchLink,
} from '../lib';
import {
  Badge,
  ErrorNotice,
  Loading,
  RouteLink,
  SectionTitle,
} from '../components/ui';

export function Overview() {
  const [attempt, setAttempt] = useState(0);
  return (
    <OverviewData
      key={attempt}
      retry={() => setAttempt((value) => value + 1)}
    />
  );
}

function OverviewData({ retry }: { retry: () => void }) {
  const [summary, setSummary] = useState<ExecutiveSummary | null>(null);
  const [error, setError] = useState<string>();
  const [queue, setQueue] = useState<Page_InvestigationSignal_ | null>(null);
  const [queueError, setQueueError] = useState<string>();
  useEffect(() => {
    const controller = new AbortController();
    api
      .summary(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setSummary(data);
      })
      .catch((failure: unknown) => {
        if (!controller.signal.aborted)
          setError(
            failure instanceof Error ? failure.message : 'Summary unavailable.',
          );
      });
    api
      .investigations({ limit: 3 }, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setQueue(data);
      })
      .catch((failure: unknown) => {
        if (!controller.signal.aborted)
          setQueueError(
            failure instanceof Error
              ? failure.message
              : 'Investigation queue unavailable.',
          );
      });
    return () => controller.abort();
  }, []);
  if (error) return <ErrorNotice message={error} retry={retry} />;
  if (!summary) return <Loading />;
  const { totals, context } = summary;
  const scanShare = totals.unique_reviews
    ? totals.rule_scanned_reviews / totals.unique_reviews
    : null;
  const excluded = totals.unique_reviews - totals.rule_scanned_reviews;
  const matchedShare = totals.rule_scanned_reviews
    ? totals.any_complaint_reviews / totals.rule_scanned_reviews
    : null;
  const largest = Math.max(1, ...Object.values(totals.complaint_reviews));
  return (
    <div className="view-stack">
      <div className="snapshot-strip">
        <div>
          <Database size={17} aria-hidden="true" />
          <strong>Historical intelligence</strong>
          <span>
            Amazon Appliances · last review {date(context.last_review_at)}
          </span>
        </div>
        <span className="quiet-label">Human-led investigation</span>
      </div>
      <div className="kpi-grid">
        <article className="kpi">
          <span className="kpi-label">
            Historical reviews
            <MessagesSquare size={17} aria-hidden="true" />
          </span>
          <strong>{number(totals.unique_reviews)}</strong>
          <span>Unique stored review IDs</span>
        </article>
        <article className="kpi">
          <span className="kpi-label">
            Products covered
            <Package size={17} aria-hidden="true" />
          </span>
          <strong>{number(summary.product_count)}</strong>
          <span>Selected product histories</span>
        </article>
        <article className="kpi">
          <span className="kpi-label">
            Negative rating burden
            <ArrowDownRight size={18} aria-hidden="true" />
          </span>
          <strong>{percent(totals.negative_rating_share)}</strong>
          <span>
            {number(totals.negative_rating_reviews)} /{' '}
            {number(totals.rated_reviews)} rated reviews · 1–2 stars
          </span>
        </article>
        <article className="kpi accent">
          <span className="kpi-label">
            Investigation candidates
            <ArrowUpRight size={18} aria-hidden="true" />
          </span>
          <strong>{number(summary.candidate_count)}</strong>
          <span>
            Across {number(summary.candidate_product_count)} products ·
            historical signals
          </span>
        </article>
      </div>

      <div className="overview-main-grid">
        <section
          className="panel complaint-chart"
          aria-label="Complaint category profile"
        >
          <SectionTitle
            title="What customers reported"
            subtitle="Explicit complaint-rule matches, by category"
            action={
              <span className="panel-icon">
                <ScanText size={18} aria-hidden="true" />
              </span>
            }
          />
          <div className="chart-headline">
            <strong>{number(totals.any_complaint_reviews)}</strong>
            <div>
              reviews with evidence
              <span>
                {percent(matchedShare)} of {number(totals.rule_scanned_reviews)}{' '}
                scanned reviews
              </span>
            </div>
            <Badge tone="teal">Rule-evidenced</Badge>
          </div>
          <div
            className="category-chart"
            role="group"
            aria-label="Complaint category counts and shares of scanned review bodies. Categories can overlap."
          >
            {Object.entries(categories).map(([key, label]) => {
              const category = key as keyof typeof categories;
              const count = totals.complaint_reviews[category];
              return (
                <div className="chart-row" key={key}>
                  <div className="chart-row-label">
                    <span>{label}</span>
                    <span>
                      <strong>{number(count)}</strong>
                      <small>
                        {percent(totals.complaint_shares[category])}
                      </small>
                    </span>
                  </div>
                  <div className="bar-track">
                    <div
                      className={`bar-fill ${key}`}
                      style={{ width: `${(100 * count) / largest}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
          <p className="chart-note">
            <Info size={14} aria-hidden="true" />
            Shares use scanned bodies. Category counts overlap; the unique total
            counts each review once.
          </p>
        </section>

        <section
          className="panel coverage-panel"
          aria-label="Evidence scan coverage"
        >
          <SectionTitle
            title="Evidence coverage"
            subtitle="Know what the rules could inspect"
          />
          <div className="coverage-visual">
            <svg
              className="coverage-ring"
              viewBox="0 0 120 120"
              role="img"
              aria-label={`${percent(scanShare)} of review bodies scanned`}
            >
              <circle cx="60" cy="60" r="49" className="ring-track" />
              <circle
                cx="60"
                cy="60"
                r="49"
                className="ring-value"
                pathLength="100"
                strokeDasharray={`${(scanShare ?? 0) * 100} 100`}
                transform="rotate(-90 60 60)"
              />
            </svg>
            <div>
              <strong>{percent(scanShare)}</strong>
              <span>scan coverage</span>
            </div>
          </div>
          <dl className="coverage-list">
            <div>
              <dt>
                <span className="legend-dot teal" />
                Bodies scanned
              </dt>
              <dd>{number(totals.rule_scanned_reviews)}</dd>
            </div>
            <div>
              <dt>
                <span className="legend-dot slate" />
                Protected / unavailable
              </dt>
              <dd>{number(excluded)}</dd>
            </div>
            <div>
              <dt>No detected rule evidence</dt>
              <dd>{number(totals.unknown_no_evidence_reviews)}</dd>
            </div>
          </dl>
          <div className="context-note">
            <ShieldCheck size={17} aria-hidden="true" />
            <p>
              No match means <strong>unknown</strong>, not issue-free. Protected
              review bodies stay excluded.
            </p>
          </div>
        </section>
      </div>

      <section className="panel priority-panel">
        <SectionTitle
          title="Investigation priorities"
          subtitle="Most recent qualifying historical window per product and metric"
          action={<RouteLink href="#/incidents">Open workbench</RouteLink>}
        />
        {queueError ? (
          <ErrorNotice message={queueError} retry={retry} />
        ) : !queue ? (
          <Loading>Loading candidate queue…</Loading>
        ) : queue.items.length === 0 ? (
          <p className="empty-message">
            No historical candidates meet the current support rules.
          </p>
        ) : (
          <div className="priority-grid">
            {queue.items.map((signal, index) => (
              <a
                key={signal.signal_id}
                className="priority-item"
                href={workbenchLink(signal.product_id, signal.signal_id)}
                aria-label={`Open workbench for ${productName(signal.product_id)}, ${metricName(signal.metric)}`}
              >
                <div className="priority-top">
                  <span className="priority-index">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                  <Badge
                    tone={
                      signal.reason === 'rising_share' ? 'amber' : 'neutral'
                    }
                  >
                    {signal.reason === 'rising_share'
                      ? 'Rising share'
                      : 'Sustained burden'}
                  </Badge>
                  <ArrowUpRight size={17} aria-hidden="true" />
                </div>
                <strong className="product-code">
                  {productName(signal.product_id)}
                </strong>
                <span>{metricName(signal.metric)}</span>
                <div className="priority-bottom">
                  <strong>
                    {number(signal.window_report_count)} /{' '}
                    {number(signal.window_denominator)}
                  </strong>
                  <span>
                    {signal.metric === 'negative_rating'
                      ? 'rated reviews'
                      : 'scanned bodies'}
                  </span>
                </div>
                <small>
                  {date(signal.window_start)} – {date(signal.window_end)}{' '}
                  <span title="Window end is exclusive">[end exclusive]</span>
                </small>
              </a>
            ))}
          </div>
        )}
        <p className="panel-footnote">
          Historical candidates prioritise human investigation. They are not
          confirmed incidents or current alerts.
        </p>
      </section>

      <ProductTable />
      <details className="provenance-details">
        <summary>About this snapshot & metric provenance</summary>
        <div>
          <p>
            Negative ratings are a rating-derived proxy, not sentiment-model
            predictions. Complaint shares describe rule matches only, using
            scanned reviews as their denominator. Products are a selected
            historical cohort, not a population sample.
          </p>
          <p>
            Category bars scale to the largest count. Candidate order favours
            rising over sustained shares, then more recent qualifying windows,
            larger share changes and supporting review counts. These reading
            heuristics are not validated risk probabilities or incident
            severity.
          </p>
          <p>
            Source: {context.dataset_revision} · {context.rule_version} ·{' '}
            {context.engine_version}
          </p>
          <p>
            Exclusive data cutoff: {date(context.as_of)}. The last review was
            recorded {date(context.last_review_at)}. Recent sample support may
            be insufficient.
          </p>
          <ul>
            {context.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </div>
      </details>
    </div>
  );
}

function ProductTable() {
  const [search, setSearch] = useState('');
  const [filters, setFilters] = useState<ProductFilters>({ limit: 6 });
  const [previous, setPrevious] = useState<string[]>([]);
  const [attempt, setAttempt] = useState(0);
  const key = JSON.stringify({ ...filters, attempt });
  const [response, setResponse] = useState<{
    key: string;
    data?: Page_ProductSummary_;
    error?: string;
  }>();
  useEffect(() => {
    const controller = new AbortController();
    api
      .products(filters, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setResponse({ key, data });
      })
      .catch((failure: unknown) => {
        if (!controller.signal.aborted)
          setResponse({
            key,
            error:
              failure instanceof Error
                ? failure.message
                : 'Product listing unavailable.',
          });
      });
    return () => controller.abort();
  }, [filters, key]);
  const current = response?.key === key ? response : undefined;
  const busy = !current;
  const applySearch = (event: FormEvent) => {
    event.preventDefault();
    setFilters({ ...filters, q: search.trim(), cursor: undefined });
    setPrevious([]);
  };
  return (
    <section className="panel products-panel">
      <SectionTitle
        title="Product portfolio"
        subtitle="Review support and rating proxies across the selected cohort"
        action={
          <span className="quiet-label">
            Product IDs · no catalogue names supplied
          </span>
        }
      />
      <div className="table-toolbar">
        <form className="search-form" onSubmit={applySearch}>
          <Search size={17} aria-hidden="true" />
          <label className="sr-only" htmlFor="product-search">
            Search product ID
          </label>
          <input
            id="product-search"
            placeholder="Search product ID…"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
          <button
            type="submit"
            className="button secondary small"
            disabled={busy}
          >
            Search
          </button>
        </form>
        <label className="filter-label">
          <span className="sr-only">Product filter</span>
          <select
            value={filters.has_signals ? 'candidates' : 'all'}
            onChange={(event) => {
              setFilters({
                ...filters,
                cursor: undefined,
                has_signals:
                  event.target.value === 'candidates' ? true : undefined,
              });
              setPrevious([]);
            }}
          >
            <option value="all">All products</option>
            <option value="candidates">With candidates</option>
          </select>
        </label>
      </div>
      {busy ? (
        <Loading>Loading products…</Loading>
      ) : current.error ? (
        <ErrorNotice
          message={current.error}
          retry={() => setAttempt((value) => value + 1)}
        />
      ) : (
        current.data && (
          <>
            <div
              className="table-scroll"
              tabIndex={0}
              role="region"
              aria-label="Product review metrics"
            >
              <table>
                <thead>
                  <tr>
                    <th>Product</th>
                    <th>Reviews</th>
                    <th>Negative ratings</th>
                    <th>Rule-evidenced reviews</th>
                    <th>Latest period support</th>
                    <th>
                      <span className="sr-only">Navigate</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {current.data.items.map((product) => (
                    <tr key={product.product_id}>
                      <td>
                        <a
                          className="product-code"
                          href={workbenchLink(product.product_id)}
                        >
                          {productName(product.product_id)}
                        </a>
                        <small>Amazon Appliances</small>
                      </td>
                      <td>{number(product.totals.unique_reviews)}</td>
                      <td>
                        <strong>
                          {percent(product.totals.negative_rating_share)}
                        </strong>
                        <small>
                          {number(product.totals.negative_rating_reviews)} /{' '}
                          {number(product.totals.rated_reviews)} rated
                        </small>
                      </td>
                      <td>
                        <strong>
                          {number(product.totals.any_complaint_reviews)}
                        </strong>
                        <small>
                          of {number(product.totals.rule_scanned_reviews)}{' '}
                          scanned
                        </small>
                      </td>
                      <td>
                        <Badge
                          tone={
                            product.comparisons.some(
                              (item) => item.state === 'comparable',
                            )
                              ? 'teal'
                              : 'neutral'
                          }
                        >
                          {product.comparisons.some(
                            (item) => item.state === 'comparable',
                          )
                            ? 'Some metrics supported'
                            : 'Insufficient support'}
                        </Badge>
                        <small>
                          {product.granularity === 'quarter'
                            ? 'Quarterly'
                            : 'Monthly'}{' '}
                          comparisons
                        </small>
                      </td>
                      <td>
                        <a
                          className="table-action"
                          href={workbenchLink(product.product_id)}
                          aria-label={`Open workbench for ${productName(product.product_id)}`}
                        >
                          <ArrowUpRight size={17} aria-hidden="true" />
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {current.data.items.length === 0 && (
                <p className="empty-message">
                  No products match these filters.
                </p>
              )}
            </div>
            <div className="table-pagination">
              <span>
                {number(current.data.total)} products match · page{' '}
                {previous.length + 1}
              </span>
              <div>
                <button
                  className="button secondary small"
                  disabled={previous.length === 0}
                  onClick={() => {
                    const cursors = [...previous];
                    const cursor = cursors.pop();
                    setPrevious(cursors);
                    setFilters({ ...filters, cursor: cursor || undefined });
                  }}
                >
                  <ChevronLeft size={15} aria-hidden="true" />
                  Previous
                </button>
                <button
                  className="button secondary small"
                  disabled={!current.data.next_cursor}
                  onClick={() => {
                    if (!current.data?.next_cursor) return;
                    setPrevious([...previous, filters.cursor ?? '']);
                    setFilters({
                      ...filters,
                      cursor: current.data.next_cursor,
                    });
                  }}
                >
                  Next
                  <ChevronRight size={15} aria-hidden="true" />
                </button>
              </div>
            </div>
          </>
        )
      )}
    </section>
  );
}
