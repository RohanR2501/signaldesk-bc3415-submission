import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
} from 'react';
import { ArrowRight, CalendarDays } from 'lucide-react';
import { api } from '../api/client';
import { useResource } from '../api/useResource';
import type {
  InvestigationSignal,
  ProductTimeline as Timeline,
  TimelinePeriod,
} from '../api/contracts.generated';
import { Badge, ErrorNotice, Loading, SectionTitle } from '../components/ui';
import {
  categories,
  date,
  metricName,
  number,
  percent,
  periodLabel,
  shareChange,
} from '../lib';

type Metric = 'volume' | InvestigationSignal['metric'];

function observation(period: TimelinePeriod, metric: Metric) {
  const data = period.metrics;
  if (metric === 'volume')
    return {
      count: data.unique_reviews,
      denominator: null,
      value: data.unique_reviews,
      supported: true,
    };
  if (metric === 'negative_rating')
    return {
      count: data.negative_rating_reviews,
      denominator: data.rated_reviews,
      value: data.negative_rating_share,
      supported: period.rating_support_sufficient,
    };
  return {
    count: data.complaint_reviews[metric],
    denominator: data.rule_scanned_reviews,
    value: data.complaint_shares[metric],
    supported: period.rule_support_sufficient,
  };
}

export function ProductTimeline({
  id,
  signal,
}: {
  id: string;
  signal?: InvestigationSignal;
}) {
  const result = useResource(
    useCallback((abort: AbortSignal) => api.timeline(id, {}, abort), [id]),
  );
  return (
    <section className="panel timeline-panel">
      <SectionTitle
        title="Historical review calendar"
        subtitle="Complete calendar periods · select a cell to inspect sample support"
        action={<CalendarDays size={19} aria-hidden="true" />}
      />
      {result.error ? (
        <ErrorNotice message={result.error} retry={result.retry} />
      ) : !result.data ? (
        <Loading>Loading calendar observations…</Loading>
      ) : result.data.periods.length ? (
        <Calendar data={result.data} signal={signal} />
      ) : (
        <p className="empty-message">
          No complete calendar periods are available for this product.
        </p>
      )}
    </section>
  );
}

function Calendar({
  data,
  signal,
}: {
  data: Timeline;
  signal?: InvestigationSignal;
}) {
  const [metric, setMetric] = useState<Metric>(signal?.metric ?? 'volume');
  const [selectedIndex, setSelectedIndex] = useState(() => {
    const found = signal
      ? data.periods.findIndex((period) => period.end === signal.window_end)
      : -1;
    return found >= 0 ? found : data.periods.length - 1;
  });
  const scroll = useRef<HTMLDivElement>(null);
  const selectedButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const container = scroll.current;
    const button = selectedButton.current;
    // Scroll the calendar itself to the chosen year, without moving page focus.
    if (container && button)
      container.scrollTop = button.offsetTop - container.offsetTop - 80;
  }, []);
  const selected = data.periods[selectedIndex];
  const current = observation(selected, metric);
  const previousPeriod = data.periods[selectedIndex - 1];
  const previous = previousPeriod ? observation(previousPeriod, metric) : null;
  const comparable =
    metric !== 'volume' &&
    current.supported &&
    previous?.supported &&
    current.value !== null &&
    previous.value !== null;
  const years = [
    ...new Set(
      data.periods.map((period) => new Date(period.start).getUTCFullYear()),
    ),
  ];
  const columns =
    data.granularity === 'quarter'
      ? ['Q1', 'Q2', 'Q3', 'Q4']
      : Array.from({ length: 12 }, (_, month) =>
          new Intl.DateTimeFormat('en-SG', {
            month: 'short',
            timeZone: 'UTC',
          }).format(new Date(Date.UTC(2020, month))),
        );
  const maximum =
    metric === 'volume'
      ? Math.max(
          1,
          ...data.periods.map((period) => period.metrics.unique_reviews),
        )
      : 1;
  const unit =
    metric === 'volume'
      ? 'unique reviews'
      : metric === 'negative_rating'
        ? 'rated reviews'
        : 'scanned bodies';
  const qualifyingIndex = signal
    ? data.periods.findIndex((period) => period.end === signal.window_end)
    : -1;
  const qualifyingPeriods =
    qualifyingIndex > 0
      ? [data.periods[qualifyingIndex - 1], data.periods[qualifyingIndex]]
      : [];
  const inWindow = (period: TimelinePeriod) =>
    !!signal &&
    period.start >= signal.window_start &&
    period.end <= signal.window_end;
  return (
    <>
      {signal && qualifyingPeriods.length === 2 && (
        <div
          className="qualifying-comparison"
          aria-label="Candidate comparison periods"
        >
          {qualifyingPeriods.map((period, index) => {
            const count = index
              ? signal.comparison.current_count
              : signal.comparison.previous_count;
            const denominator = index
              ? signal.comparison.current_denominator
              : signal.comparison.previous_denominator;
            return (
              <div key={period.start}>
                <span>
                  {index
                    ? 'Current qualifying period'
                    : 'Previous comparison period'}{' '}
                  · {periodLabel(period.start, data.granularity)}
                </span>
                <strong>
                  {number(count)} / {number(denominator)}{' '}
                  <small>
                    {signal.metric === 'negative_rating'
                      ? 'rated reviews'
                      : 'scanned bodies'}
                  </small>
                </strong>
                <span>
                  {date(period.start)} – {date(period.end)} · end exclusive
                </span>
              </div>
            );
          })}
        </div>
      )}
      <div className="timeline-toolbar">
        <label>
          Calendar metric
          <select
            value={metric}
            onChange={(event) => setMetric(event.target.value as Metric)}
          >
            <option value="volume">Review volume</option>
            <option value="negative_rating">Negative rating share</option>
            {Object.entries(categories).map(([key, label]) => (
              <option key={key} value={key}>
                {label} · rule share
              </option>
            ))}
          </select>
        </label>
        <span>
          {data.granularity === 'quarter' ? 'Quarterly' : 'Monthly'} aggregation
          ·{' '}
          {metric === 'volume'
            ? 'colour scales to maximum observed count'
            : 'colour uses a fixed 0–100% share scale'}
        </span>
      </div>
      <div
        className="calendar-scroll"
        ref={scroll}
        role="region"
        aria-label="Interactive historical calendar"
        tabIndex={0}
      >
        <div
          className="calendar-grid"
          style={{ '--periods': columns.length } as CSSProperties}
        >
          <div className="calendar-year-heading">UTC</div>
          {columns.map((column) => (
            <div className="calendar-column" key={column}>
              {column}
            </div>
          ))}
          {years.map((year) => (
            <CalendarYear
              key={year}
              year={year}
              columns={columns.length}
              data={data}
              metric={metric}
              maximum={maximum}
              selectedIndex={selectedIndex}
              onSelect={setSelectedIndex}
              selectedRef={selectedButton}
              inWindow={inWindow}
            />
          ))}
        </div>
      </div>
      <div className="calendar-legend">
        <span>
          <i className="calendar-key observed" />
          Observed value
        </span>
        <span>
          <i className="calendar-key sparse" />
          Below comparison support
        </span>
        {signal && (
          <span>
            <i className="calendar-key qualifying" />
            Candidate window
          </span>
        )}
        <span>Blank slots are outside the returned series</span>
      </div>
      <div className="period-inspector" aria-live="polite">
        <div className="period-heading">
          <div>
            <p className="eyebrow">SELECTED PERIOD</p>
            <h3>{periodLabel(selected.start, data.granularity)}</h3>
            <span>
              {date(selected.start)} – {date(selected.end)} · end exclusive
            </span>
          </div>
          <Badge
            tone={metric === 'volume' || current.supported ? 'teal' : 'neutral'}
          >
            {metric === 'volume'
              ? 'Observed review activity'
              : current.supported
                ? 'Denominator support met'
                : 'Insufficient support'}
          </Badge>
        </div>
        <div className="period-metrics">
          <div>
            <strong>
              {metric === 'volume'
                ? number(current.count)
                : percent(current.value)}
            </strong>
            <span>
              {metric === 'volume'
                ? 'unique review IDs'
                : `${number(current.count)} / ${number(current.denominator ?? 0)} ${unit}`}
            </span>
            <small>
              {metric === 'volume'
                ? 'Review counts, not verified incidents'
                : metricName(metric)}
            </small>
          </div>
          <div>
            <strong>
              {number(selected.metrics.rule_scanned_reviews)} /{' '}
              {number(selected.metrics.unique_reviews)}
            </strong>
            <span>bodies scanned / stored reviews</span>
            <small>
              {number(selected.metrics.unknown_no_evidence_reviews)} unknown
              without rule evidence
            </small>
          </div>
        </div>
        {metric !== 'volume' && (
          <div className="period-comparison">
            <ArrowRight size={16} aria-hidden="true" />
            <div>
              {comparable && previous ? (
                <>
                  <strong>
                    {shareChange(current.value! - previous.value!)} versus{' '}
                    {periodLabel(previousPeriod.start, data.granularity)}
                  </strong>
                  <span>
                    {previous.count}/{previous.denominator} → {current.count}/
                    {current.denominator} {unit}. Descriptive change; no
                    statistical significance claim.
                  </span>
                </>
              ) : (
                <>
                  <strong>Adjacent-period comparison unavailable</strong>
                  <span>
                    Requires at least {data.min_denominator} {unit} in both
                    complete periods.{' '}
                    {previous
                      ? `Previous support: ${previous.count}/${previous.denominator}.`
                      : 'No preceding period in the returned series.'}
                  </span>
                </>
              )}
            </div>
          </div>
        )}
      </div>
      <p className="chart-note">
        {number(data.incomplete_period_reviews)} reviews in the unfinished
        period are excluded from this calendar and comparisons. Missing dates
        remain in product totals only. Review activity does not date verified
        defects.
      </p>
      {metric !== 'volume' && metric !== 'negative_rating' && (
        <p className="reliability-note">
          Rule-match share is not complaint prevalence. Zero matches mean no
          detected evidence, not confirmed absence. Meeting the denominator
          floor does not establish reliable rare-category detection.
        </p>
      )}
    </>
  );
}

function CalendarYear({
  year,
  columns,
  data,
  metric,
  maximum,
  selectedIndex,
  onSelect,
  selectedRef,
  inWindow,
}: {
  year: number;
  columns: number;
  data: Timeline;
  metric: Metric;
  maximum: number;
  selectedIndex: number;
  onSelect: (index: number) => void;
  selectedRef: React.RefObject<HTMLButtonElement | null>;
  inWindow: (period: TimelinePeriod) => boolean;
}) {
  return (
    <>
      <div className="calendar-year">{year}</div>
      {Array.from({ length: columns }, (_, column) => {
        const index = data.periods.findIndex((period) => {
          const start = new Date(period.start);
          return (
            start.getUTCFullYear() === year &&
            Math.floor(start.getUTCMonth() / (columns === 4 ? 3 : 1)) === column
          );
        });
        if (index < 0)
          return (
            <div className="calendar-blank" key={column} aria-hidden="true" />
          );
        const period = data.periods[index];
        const value = observation(period, metric);
        const noDenominator = metric !== 'volume' && !value.denominator;
        const label = `${periodLabel(period.start, data.granularity)}: ${metric === 'volume' ? `${value.count} reviews` : `${percent(value.value)}, ${value.count} of ${value.denominator} ${metric === 'negative_rating' ? 'rated' : 'scanned'}`}${!value.supported ? ', insufficient comparison support' : ''}${inWindow(period) ? ', candidate window' : ''}`;
        return (
          <button
            key={column}
            ref={index === selectedIndex ? selectedRef : undefined}
            type="button"
            className={`calendar-cell ${index === selectedIndex ? 'selected' : ''} ${!value.supported ? 'sparse' : ''} ${inWindow(period) ? 'in-window' : ''} ${noDenominator ? 'no-observation' : ''}`}
            style={
              {
                '--intensity':
                  value.value === null
                    ? 0
                    : 0.04 + (0.6 * value.value) / maximum,
              } as CSSProperties
            }
            aria-label={label}
            aria-pressed={index === selectedIndex}
            onClick={() => onSelect(index)}
          >
            <strong>
              {metric === 'volume'
                ? number(value.count)
                : value.value === null
                  ? '—'
                  : percent(value.value)}
            </strong>
            <span>
              {metric === 'volume'
                ? 'reviews'
                : `${value.count} / ${value.denominator}`}
            </span>
            <small>
              {metric === 'volume'
                ? 'stored IDs'
                : noDenominator
                  ? 'no denominator'
                  : value.supported
                    ? 'sample supported'
                    : 'low sample'}
            </small>
          </button>
        );
      })}
    </>
  );
}
