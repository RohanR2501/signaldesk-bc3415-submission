import { useCallback } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  ChevronLeft,
  ChevronRight,
  Database,
  Layers3,
  ScanText,
} from 'lucide-react';
import { api, type InvestigationFilters } from '../api/client';
import { useResource } from '../api/useResource';
import type {
  InvestigationDetail,
  InvestigationSignal,
  ProductDetail,
} from '../api/contracts.generated';
import {
  Badge,
  ErrorNotice,
  Loading,
  RouteLink,
  SectionTitle,
} from '../components/ui';
import {
  categories,
  date,
  metricName,
  number,
  percent,
  productName,
  shareChange,
  workbenchLink,
} from '../lib';
import { ProductTimeline } from './ProductTimeline';
import { InvestigationEvidence, SourceReview } from './ReviewEvidence';
import './workbench.css';

const metrics = [
  'negative_rating',
  ...Object.keys(categories),
] as InvestigationSignal['metric'][];
const reasonName = (reason: InvestigationSignal['reason']) =>
  reason === 'rising_share' ? 'Rising share' : 'Sustained burden';
const denominatorName = (metric: InvestigationSignal['metric']) =>
  metric === 'negative_rating' ? 'rated reviews' : 'scanned bodies';

/** Hash parameters keep selections, filters and cursor navigation shareable. */
export function Workbench({ hash }: { hash: string }) {
  const params = new URLSearchParams(hash.split('?')[1]);
  const product = params.get('product');
  const signal = params.get('signal');
  const review = params.get('review');
  const returnParams = new URLSearchParams(params);
  ['product', 'signal', 'review'].forEach((key) => returnParams.delete(key));
  const queueHref = `#/incidents${returnParams.size ? `?${returnParams}` : ''}`;
  return (
    <div className="view-stack workbench">
      <div className="snapshot-strip">
        <div>
          <Database size={17} aria-hidden="true" />
          <strong>Historical investigation workspace</strong>
          <span>Source records end September 2023</span>
        </div>
        <Badge>Candidate signals · not verified incidents</Badge>
      </div>
      {(product || signal || review) && (
        <nav
          className="workbench-breadcrumb"
          aria-label="Investigation navigation"
        >
          <a href={queueHref}>
            <ArrowLeft size={14} aria-hidden="true" /> Investigation queue
          </a>
          {product && (
            <>
              <ChevronRight size={13} aria-hidden="true" />
              <a href={workbenchLink(product)} className="product-code">
                {productName(product)}
              </a>
            </>
          )}
          {signal && (
            <>
              <ChevronRight size={13} aria-hidden="true" />
              <span>Candidate evidence</span>
            </>
          )}
        </nav>
      )}
      {signal ? (
        <Candidate key={signal} id={signal} review={review} />
      ) : product ? (
        <Product key={product} id={product} review={review} />
      ) : review ? (
        <SourceReview id={review} />
      ) : (
        <InvestigationQueue params={params} />
      )}
    </div>
  );
}

function InvestigationQueue({ params }: { params: URLSearchParams }) {
  const product = params.get('queue_product') ?? '';
  const metric = params.get('metric') ?? '';
  const reason = params.get('reason') ?? '';
  const cursor = params.get('cursor') ?? undefined;
  const invalid =
    (metric && !metrics.includes(metric as InvestigationSignal['metric'])) ||
    (reason && reason !== 'rising_share' && reason !== 'sustained_burden');
  const load = useCallback(
    (abort: AbortSignal) =>
      api.investigations(
        {
          limit: 8,
          product_id: product || undefined,
          metric: (metric as InvestigationFilters['metric']) || undefined,
          reason: (reason as InvestigationFilters['reason']) || undefined,
          cursor,
        },
        abort,
      ),
    [product, metric, reason, cursor],
  );
  const queue = useResource(load);
  const summary = useResource(
    useCallback((abort: AbortSignal) => api.summary(abort), []),
  );
  const products = useResource(
    useCallback(
      (abort: AbortSignal) =>
        api.products({ has_signals: true, limit: 100 }, abort),
      [],
    ),
  );
  const href = (next: URLSearchParams) =>
    `#/incidents${next.size ? `?${next}` : ''}`;
  const filter = (name: string, value: string) => {
    const next = new URLSearchParams(params);
    ['cursor', 'back'].forEach((key) => next.delete(key));
    if (value) next.set(name, value);
    else next.delete(name);
    window.location.hash = href(next);
  };
  const next = new URLSearchParams(params);
  if (queue.data?.next_cursor) {
    next.append('back', cursor ?? '');
    next.set('cursor', queue.data.next_cursor);
  }
  const previous = new URLSearchParams(params);
  const history = previous.getAll('back');
  const previousCursor = history.pop();
  previous.delete('back');
  history.forEach((value) => previous.append('back', value));
  if (previousCursor) previous.set('cursor', previousCursor);
  else previous.delete('cursor');
  return (
    <>
      <section className="queue-intro">
        <div>
          <p className="eyebrow">FROM SIGNAL TO SOURCE</p>
          <h2>Where investigation begins.</h2>
          <p>Read the historical comparison, then the customer’s words.</p>
        </div>
        {summary.data && (
          <div className="queue-totals">
            <div>
              <strong>{number(summary.data.candidate_count)}</strong>
              <span>historical candidates</span>
            </div>
            <div>
              <strong>{number(summary.data.candidate_product_count)}</strong>
              <span>products represented</span>
            </div>
          </div>
        )}
        {summary.error && (
          <span className="muted">Portfolio totals unavailable.</span>
        )}
      </section>
      <section className="panel queue-panel">
        <SectionTitle
          title="Investigation queue"
          subtitle="Backend priority order · rising share first, then sustained burden"
          action={<Badge tone="teal">Human-led review</Badge>}
        />
        <div className="queue-filters">
          <label>
            Product
            <select
              value={product}
              onChange={(event) => filter('queue_product', event.target.value)}
              disabled={!products.data}
            >
              <option value="">All candidate products</option>
              {product &&
                !products.data?.items.some(
                  (item) => item.product_id === product,
                ) && <option value={product}>{productName(product)}</option>}
              {products.data?.items.map((item) => (
                <option key={item.product_id} value={item.product_id}>
                  {productName(item.product_id)}
                </option>
              ))}
            </select>
          </label>
          <label>
            Metric / category
            <select
              value={metric}
              onChange={(event) => filter('metric', event.target.value)}
            >
              <option value="">All metrics</option>
              {metrics.map((item) => (
                <option key={item} value={item}>
                  {metricName(item)}
                </option>
              ))}
            </select>
          </label>
          <label>
            Prioritisation reason
            <select
              value={reason}
              onChange={(event) => filter('reason', event.target.value)}
            >
              <option value="">All reasons</option>
              <option value="rising_share">Rising share</option>
              <option value="sustained_burden">Sustained burden</option>
            </select>
          </label>
          <a href="#/incidents" className="text-link">
            Reset filters
          </a>
        </div>
        {products.error && (
          <ErrorNotice
            message={`Product filter unavailable. ${products.error}`}
            retry={products.retry}
          />
        )}
        {invalid ? (
          <ErrorNotice message="This URL contains an unsupported filter. Reset the filters to continue." />
        ) : queue.loading ? (
          <Loading>Loading historical candidates…</Loading>
        ) : queue.error ? (
          <>
            <ErrorNotice message={queue.error} retry={queue.retry} />
            <a className="text-link" href="#/incidents">
              Reset filters and cursor
            </a>
          </>
        ) : (
          queue.data && (
            <>
              <div className="queue-results-label" role="status">
                <strong>{number(queue.data.total)} matching candidates</strong>
                <span>Original priority order preserved</span>
              </div>
              {queue.data.items.length ? (
                <div className="candidate-list">
                  {queue.data.items.map((item) => (
                    <CandidateCard
                      key={item.signal_id}
                      signal={item}
                      href={`${workbenchLink(item.product_id, item.signal_id)}&${params}`}
                    />
                  ))}
                </div>
              ) : (
                <div className="empty-message">
                  <Layers3 size={25} aria-hidden="true" />
                  <p>No historical candidates match these filters.</p>
                  <a className="text-link" href="#/incidents">
                    Clear filters
                  </a>
                </div>
              )}
              <div className="table-pagination">
                <span>
                  {number(queue.data.items.length)} shown ·{' '}
                  {date(queue.data.context.last_review_at)} last source review
                </span>
                <div>
                  {cursor ? (
                    <a href={href(previous)} className="button secondary small">
                      <ChevronLeft size={15} aria-hidden="true" />
                      {previousCursor === undefined ? 'First page' : 'Previous'}
                    </a>
                  ) : (
                    <button className="button secondary small" disabled>
                      <ChevronLeft size={15} aria-hidden="true" />
                      Previous
                    </button>
                  )}
                  {queue.data.next_cursor ? (
                    <a href={href(next)} className="button secondary small">
                      Next candidates
                      <ChevronRight size={15} aria-hidden="true" />
                    </a>
                  ) : (
                    <button className="button secondary small" disabled>
                      Next candidates
                      <ChevronRight size={15} aria-hidden="true" />
                    </button>
                  )}
                </div>
              </div>
            </>
          )
        )}
      </section>
      <details className="provenance-details">
        <summary>How candidates are prioritised</summary>
        <div>
          <p>
            The queue retains the latest qualifying historical window per
            product and metric. Rising shares come before sustained burden;
            subsequent ordering uses recency, share change, report count and
            stable identifiers.
          </p>
          {summary.data && (
            <p>
              Each comparison needs at least{' '}
              {summary.data.policy.min_denominator} relevant reviews in each
              complete period. Rising shares need{' '}
              {summary.data.policy.min_reports} current reports and an increase
              of {shareChange(summary.data.policy.negative_increase)} for
              negative ratings or{' '}
              {shareChange(summary.data.policy.complaint_increase)} for rule
              matches. Sustained burden needs{' '}
              {summary.data.policy.min_sustained_reports} combined reports and
              at least {percent(summary.data.policy.sustained_negative_share)}{' '}
              negative ratings or{' '}
              {percent(summary.data.policy.sustained_complaint_share)} rule
              matches.
            </p>
          )}
          <p>
            These are transparent reading-budget heuristics, not statistical
            significance, incident severity or calibrated risk. Wrong-item and
            delivery learned predictions do not drive this queue. Rule
            reliability is limited; no rule match means unknown.
          </p>
        </div>
      </details>
    </>
  );
}

function CandidateCard({
  signal,
  href,
}: {
  signal: InvestigationSignal;
  href?: string;
}) {
  const comparison = signal.comparison;
  return (
    <article className="candidate-card">
      <div className="candidate-main">
        <div className="candidate-heading">
          <a className="product-code" href={workbenchLink(signal.product_id)}>
            {productName(signal.product_id)}
          </a>
          <Badge tone={signal.reason === 'rising_share' ? 'amber' : 'neutral'}>
            {reasonName(signal.reason)}
          </Badge>
        </div>
        <a
          className="candidate-title"
          href={href ?? workbenchLink(signal.product_id, signal.signal_id)}
        >
          {metricName(signal.metric)}
          <ArrowUpRight size={16} aria-hidden="true" />
        </a>
        <span className="muted">
          {date(signal.window_start)} – {date(signal.window_end)}{' '}
          <small>(end exclusive)</small>
        </span>
      </div>
      <div className="candidate-support">
        <strong>
          {number(signal.window_report_count)}{' '}
          <span>/ {number(signal.window_denominator)}</span>
        </strong>
        <span>{denominatorName(signal.metric)} in window</span>
      </div>
      <div className="candidate-change">
        <strong>{shareChange(comparison.share_change)}</strong>
        <span>
          {comparison.previous_count}/{comparison.previous_denominator} →{' '}
          {comparison.current_count}/{comparison.current_denominator}
        </span>
        <small>Previous → current period</small>
      </div>
      <a
        className="button secondary small"
        href={href ?? workbenchLink(signal.product_id, signal.signal_id)}
      >
        Inspect evidence
        <ArrowRight size={14} aria-hidden="true" />
      </a>
    </article>
  );
}

function Candidate({ id, review }: { id: string; review: string | null }) {
  const result = useResource(
    useCallback((abort: AbortSignal) => api.investigation(id, abort), [id]),
  );
  if (result.error)
    return <ErrorNotice message={result.error} retry={result.retry} />;
  if (!result.data)
    return <Loading>Loading candidate and evidence provenance…</Loading>;
  const detail = result.data;
  return (
    <>
      <CandidateSummary detail={detail} />
      <Product id={detail.signal.product_id} detail={detail} review={review} />
    </>
  );
}

function CandidateSummary({ detail }: { detail: InvestigationDetail }) {
  const { signal, policy } = detail;
  const comparison = signal.comparison;
  return (
    <section className="panel candidate-summary">
      <div className="section-title">
        <div>
          <p className="eyebrow">HISTORICAL INVESTIGATION CANDIDATE</p>
          <h2>{metricName(signal.metric)}</h2>
          <a className="product-code" href={workbenchLink(signal.product_id)}>
            {productName(signal.product_id)}
          </a>
        </div>
        <Badge tone={signal.reason === 'rising_share' ? 'amber' : 'neutral'}>
          {reasonName(signal.reason)}
        </Badge>
      </div>
      <p className="candidate-rationale">{signal.explanation}</p>
      <div className="signal-facts">
        <div>
          <span>Qualifying historical window</span>
          <strong>
            {date(signal.window_start)} – {date(signal.window_end)}
          </strong>
          <small>UTC · end exclusive</small>
        </div>
        <div>
          <span>Window support</span>
          <strong>
            {number(signal.window_report_count)} /{' '}
            {number(signal.window_denominator)}
          </strong>
          <small>
            {denominatorName(signal.metric)} ·{' '}
            {signal.metric === 'negative_rating'
              ? 'rating proxy, not model sentiment'
              : 'explicit rule evidence'}
          </small>
        </div>
        <div>
          <span>Period share change</span>
          <strong>{shareChange(comparison.share_change)}</strong>
          <small>
            {comparison.previous_count}/{comparison.previous_denominator} →{' '}
            {comparison.current_count}/{comparison.current_denominator}{' '}
            {denominatorName(signal.metric)}
          </small>
        </div>
      </div>
      <details className="candidate-policy">
        <summary>Prioritisation rule & source identifiers</summary>
        <p>
          Both comparison periods require at least {policy.min_denominator}{' '}
          {denominatorName(signal.metric)}.{' '}
          {signal.reason === 'rising_share'
            ? `A rising share needs ${policy.min_reports} current reports and an increase of ${shareChange(signal.metric === 'negative_rating' ? policy.negative_increase : policy.complaint_increase)}.`
            : `Sustained burden needs ${policy.min_sustained_reports} combined reports and a combined share of at least ${percent(signal.metric === 'negative_rating' ? policy.sustained_negative_share : policy.sustained_complaint_share)}.`}
        </p>
        <p>
          Candidate ID: <code>{signal.signal_id}</code> · Product ID:{' '}
          <code>{signal.product_id}</code>
        </p>
      </details>
    </section>
  );
}

function Product({
  id,
  detail,
  review,
}: {
  id: string;
  detail?: InvestigationDetail;
  review: string | null;
}) {
  const result = useResource(
    useCallback((abort: AbortSignal) => api.product(id, abort), [id]),
  );
  return (
    <>
      {result.error ? (
        <ErrorNotice message={result.error} retry={result.retry} />
      ) : !result.data ? (
        <Loading>Loading product history…</Loading>
      ) : (
        <ProductMetrics detail={result.data} />
      )}
      <ProductTimeline
        key={`${id}:${detail?.signal.signal_id ?? ''}`}
        id={id}
        signal={detail?.signal}
      />
      {detail && (
        <InvestigationEvidence detail={detail} selectedReview={review} />
      )}
      {review && (
        <SourceReview
          key={review}
          id={review}
          expectedProduct={id}
          signalId={detail?.signal.signal_id}
        />
      )}
      <RelatedCandidates product={id} selected={detail?.signal.signal_id} />
      {result.data && (
        <details className="provenance-details">
          <summary>Product history provenance & limitations</summary>
          <div>
            <p>
              Last source review: {date(result.data.context.last_review_at)}.
              Snapshot {result.data.context.snapshot_id} ·{' '}
              {result.data.context.engine_version} ·{' '}
              {result.data.context.rule_version}.
            </p>
            <p>
              Ratings use {result.data.product.totals.negative_provenance}.
              Complaint evidence uses{' '}
              {result.data.product.totals.complaint_provenance}. Overall matches
              are deduplicated across all five categories.
            </p>
            <p>
              Historical review dates are not independently dated defect
              occurrences. Unknown rule results do not establish absence.
              Wrong-item and delivery methods are unreliable and never verify an
              incident; rare-category positive support remains limited.
            </p>
            {Object.entries(
              result.data.product.totals.text_unavailable_reasons,
            ).map(([reason, count]) => (
              <p key={reason}>
                {reason}: {number(count)} withheld bodies
              </p>
            ))}
          </div>
        </details>
      )}
    </>
  );
}

function ProductMetrics({ detail }: { detail: ProductDetail }) {
  const { product } = detail;
  const { totals } = product;
  return (
    <section className="panel product-summary">
      <SectionTitle
        title={`Product ${productName(product.product_id)}`}
        subtitle={product.product_id}
        action={<Badge>{detail.candidate_count} historical candidates</Badge>}
      />
      <div className="product-metric-grid">
        <div>
          <span>Historical reviews</span>
          <strong>{number(totals.unique_reviews)}</strong>
          <small>Unique stored review IDs</small>
        </div>
        <div>
          <span>Negative rating burden</span>
          <strong>{percent(totals.negative_rating_share)}</strong>
          <small>
            {number(totals.negative_rating_reviews)} /{' '}
            {number(totals.rated_reviews)} rated · 1–2 stars
          </small>
        </div>
        <div>
          <span>Rule-evidenced reviews</span>
          <strong>{number(totals.any_complaint_reviews)}</strong>
          <small>Union of all category matches</small>
        </div>
        <div>
          <span>Body scan coverage</span>
          <strong>
            {percent(
              totals.unique_reviews
                ? totals.rule_scanned_reviews / totals.unique_reviews
                : null,
            )}
          </strong>
          <small>
            {number(totals.rule_scanned_reviews)} /{' '}
            {number(totals.unique_reviews)} bodies scanned
          </small>
        </div>
      </div>
      <div
        className="product-category-strip"
        aria-label="Five complaint categories"
      >
        {Object.entries(categories).map(([category, label]) => (
          <div key={category}>
            <span>{label}</span>
            <strong>
              {number(
                totals.complaint_reviews[category as keyof typeof categories],
              )}{' '}
              <small>/ {number(totals.rule_scanned_reviews)} scanned</small>
            </strong>
          </div>
        ))}
      </div>
      <p className="chart-note">
        <ScanText size={14} aria-hidden="true" />
        {number(totals.unknown_no_evidence_reviews)} scanned reviews have no
        qualifying rule evidence (unknown). Categories can overlap.
      </p>
    </section>
  );
}

function RelatedCandidates({
  product,
  selected,
}: {
  product: string;
  selected?: string;
}) {
  const result = useResource(
    useCallback(
      (abort: AbortSignal) =>
        api.investigations({ product_id: product, limit: 100 }, abort),
      [product],
    ),
  );
  return (
    <section className="panel related-candidates">
      <SectionTitle
        title="Related investigation candidates"
        subtitle="Historical candidates for this product, in backend priority order"
      />
      {result.error ? (
        <ErrorNotice message={result.error} retry={result.retry} />
      ) : !result.data ? (
        <Loading>Loading related candidates…</Loading>
      ) : (
        <>
          {result.data.items
            .filter((item) => item.signal_id !== selected)
            .map((item) => (
              <CandidateCard key={item.signal_id} signal={item} />
            ))}
          {result.data.items.every((item) => item.signal_id === selected) && (
            <p className="muted">
              {selected
                ? 'No additional candidates for this product.'
                : 'No historical candidates met the support rules. This does not establish the absence of complaints.'}
            </p>
          )}
          <RouteLink href="#/incidents">Explore the full queue</RouteLink>
        </>
      )}
    </section>
  );
}
