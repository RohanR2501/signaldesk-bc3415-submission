import { useCallback, useEffect, useRef } from 'react';
import { ArrowUpRight, FileText, LockKeyhole, X } from 'lucide-react';
import { api } from '../api/client';
import { useResource } from '../api/useResource';
import type {
  InvestigationDetail,
  ReviewExcerpt,
} from '../api/contracts.generated';
import { Badge, ErrorNotice, Loading, SectionTitle } from '../components/ui';
import { categories, date, number, productName, workbenchLink } from '../lib';

export function InvestigationEvidence({
  detail,
  selectedReview,
}: {
  detail: InvestigationDetail;
  selectedReview: string | null;
}) {
  const { signal } = detail;
  return (
    <section className="panel investigation-evidence">
      <SectionTitle
        title="Supporting customer reviews"
        subtitle={`${number(detail.supporting_review_ids.length)} supporting review IDs · ${number(detail.evidence.length)} stored excerpts shown`}
        action={<FileText size={19} aria-hidden="true" />}
      />
      <p className="evidence-intro">
        {signal.metric === 'negative_rating'
          ? 'These source excerpts provide context for 1–2 star ratings. They are not complaint attributions or sentiment-model explanations.'
          : 'These exact rule matches support the selected complaint category. They describe customer reports, not verified incidents.'}{' '}
        {number(signal.evidence_unavailable_reviews)} supporting bodies are
        protected or unavailable.
      </p>
      <div className="source-excerpts">
        {detail.evidence.map((evidence, index) => (
          <article
            className="source-excerpt-card"
            key={`${evidence.review_id}:${index}`}
          >
            <div className="source-excerpt-top">
              <Badge
                tone={evidence.method === 'rule_match' ? 'amber' : 'neutral'}
              >
                {evidence.method === 'rule_match'
                  ? 'Rule-evidenced complaint'
                  : 'Rating-only context'}
              </Badge>
              <span>{date(evidence.reviewed_at)}</span>
            </div>
            <ExactSourceExcerpt evidence={evidence} />
            <a
              className="text-link source-id"
              href={workbenchLink(
                signal.product_id,
                signal.signal_id,
                evidence.review_id,
              )}
            >
              Open source review <code>{evidence.review_id}</code>
              <ArrowUpRight size={14} aria-hidden="true" />
            </a>
          </article>
        ))}
      </div>
      {!detail.evidence.length && (
        <p className="empty-message">
          No accessible source excerpts were returned. Missing text is not
          reconstructed.
        </p>
      )}
      <label className="supporting-review-select">
        Explore any supporting review ID
        <select
          value={
            selectedReview &&
            detail.supporting_review_ids.includes(selectedReview)
              ? selectedReview
              : ''
          }
          onChange={(event) => {
            if (event.target.value)
              window.location.hash = workbenchLink(
                signal.product_id,
                signal.signal_id,
                event.target.value,
              );
          }}
        >
          <option value="">Choose a source review…</option>
          {detail.supporting_review_ids.map((id, index) => (
            <option key={id} value={id}>
              {index + 1}. {id}
            </option>
          ))}
        </select>
      </label>
      <p className="chart-note">
        Excerpts are a bounded sample, not all supporting text. Source offsets
        use Unicode code points, start inclusive and end exclusive. No excerpt
        is a model explanation span.
      </p>
    </section>
  );
}

/** Render returned source substrings; validate offsets only against an accessible original. */
function ExactSourceExcerpt({
  evidence,
  original,
}: {
  evidence: ReviewExcerpt;
  original?: string;
}) {
  const points = original === undefined ? undefined : Array.from(original);
  const exact = points
    ? validSpan(points, evidence.start, evidence.end, evidence.text)
    : undefined;
  const resolutionExact =
    points &&
    evidence.resolution_excerpt !== null &&
    evidence.resolution_start !== null &&
    evidence.resolution_end !== null
      ? validSpan(
          points,
          evidence.resolution_start,
          evidence.resolution_end,
          evidence.resolution_excerpt,
        )
      : undefined;
  return (
    <div className="evidence-item">
      <blockquote>
        {exact && points ? (
          <>
            {evidence.start > 80 && '…'}
            {points
              .slice(Math.max(0, evidence.start - 80), evidence.start)
              .join('')}
            <mark>{evidence.text}</mark>
            {points.slice(evidence.end, evidence.end + 80).join('')}
            {evidence.end + 80 < points.length && '…'}
          </>
        ) : (
          evidence.text
        )}
      </blockquote>
      <div className="evidence-caption">
        <span>
          {evidence.method === 'rule_match'
            ? 'Exact rule match'
            : 'Original source excerpt'}{' '}
          · [{evidence.start}, {evidence.end})
        </span>
        {evidence.rule_id && <code>{evidence.rule_id}</code>}
      </div>
      {exact === false && (
        <p className="input-error">
          The returned offsets do not match the original body. The returned
          excerpt is shown without a reconstructed highlight.
        </p>
      )}
      {evidence.resolution_excerpt !== null && (
        <div className="resolution-note">
          <Badge tone="teal">Reported resolved</Badge>
          <q>{evidence.resolution_excerpt}</q>
          <small>
            Source offsets [{evidence.resolution_start},{' '}
            {evidence.resolution_end}) · the reported occurrence is retained
          </small>
          {resolutionExact === false && (
            <p className="input-error">
              Resolution offsets do not match the original body.
            </p>
          )}
        </div>
      )}
      {evidence.method === 'rule_match' &&
        evidence.resolution === 'not_established' && (
          <small className="resolution-status">
            Resolution not established by the rule evidence.
          </small>
        )}
    </div>
  );
}

function validSpan(points: string[], start: number, end: number, text: string) {
  return (
    start >= 0 &&
    end >= start &&
    end <= points.length &&
    points.slice(start, end).join('') === text
  );
}

function OriginalBody({
  text,
  evidence,
}: {
  text: string;
  evidence: ReviewExcerpt[];
}) {
  // Python offsets count Unicode code points. Merge only validated, overlapping
  // rule spans; every other character stays exactly as returned by the source.
  const points = Array.from(text);
  const spans = evidence
    .filter(
      (span) =>
        span.method === 'rule_match' &&
        validSpan(points, span.start, span.end, span.text),
    )
    .sort((a, b) => a.start - b.start);
  const ranges: { start: number; end: number }[] = [];
  for (const span of spans) {
    const last = ranges.at(-1);
    if (last && span.start <= last.end) last.end = Math.max(last.end, span.end);
    else ranges.push({ start: span.start, end: span.end });
  }
  let offset = 0;
  return (
    <div className="original-review-body">
      {ranges.map((range) => {
        const before = points.slice(offset, range.start).join('');
        offset = range.end;
        return (
          <span key={range.start}>
            {before}
            <mark>{points.slice(range.start, range.end).join('')}</mark>
          </span>
        );
      })}
      {points.slice(offset).join('')}
    </div>
  );
}

export function SourceReview({
  id,
  expectedProduct,
  signalId,
}: {
  id: string;
  expectedProduct?: string;
  signalId?: string;
}) {
  const result = useResource(
    useCallback((abort: AbortSignal) => api.review(id, abort), [id]),
  );
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    if (result.data || result.error) {
      heading.current?.focus();
      heading.current?.scrollIntoView({ block: 'start' });
    }
  }, [result.data, result.error]);
  const review = result.data;
  const available = review?.text_state === 'available' && review.text !== null;
  return (
    <section
      className="panel source-review"
      aria-label="Source review explorer"
    >
      <div className="section-title">
        <div>
          <p className="eyebrow">SOURCE REVIEW EXPLORER</p>
          <h2 ref={heading} tabIndex={-1}>
            Original review & evidence
          </h2>
        </div>
        {expectedProduct && (
          <a
            className="button secondary small"
            href={workbenchLink(expectedProduct, signalId)}
          >
            <X size={14} aria-hidden="true" />
            Close review
          </a>
        )}
      </div>
      {result.error ? (
        <ErrorNotice message={result.error} retry={result.retry} />
      ) : !review ? (
        <Loading>Loading permitted source review…</Loading>
      ) : expectedProduct && review.product_id !== expectedProduct ? (
        <>
          <ErrorNotice message="This review belongs to another product. Open its product history to inspect it." />
          <a
            className="text-link"
            href={workbenchLink(review.product_id, undefined, review.review_id)}
          >
            Open the review’s product history
          </a>
        </>
      ) : (
        <>
          <div className="review-metadata">
            <code>{review.review_id}</code>
            <span>{date(review.reviewed_at)}</span>
            <span>
              {review.rating === null
                ? 'Rating unavailable'
                : `${review.rating} / 5 stars`}
            </span>
            <a href={workbenchLink(review.product_id)} className="product-code">
              {productName(review.product_id)}
              <ArrowUpRight size={13} aria-hidden="true" />
            </a>
          </div>
          {available ? (
            <div className="review-source-grid">
              <div>
                <h3>Original stored body</h3>
                <OriginalBody
                  text={review.text!}
                  evidence={review.complaints.flatMap(
                    (complaint) => complaint.evidence,
                  )}
                />
                <p className="chart-note">
                  Original Unicode text, without rewriting. Highlights mark
                  validated rule matches; surrounding text retains negation and
                  context.
                </p>
              </div>
              <div className="source-complaints">
                <h3>Independent complaint evidence</h3>
                {review.complaints.map((complaint) => (
                  <details
                    className="source-complaint"
                    key={complaint.category}
                    open={complaint.state === 'detected'}
                  >
                    <summary>
                      <span>{categories[complaint.category]}</span>
                      <Badge
                        tone={
                          complaint.state === 'detected' ? 'amber' : 'neutral'
                        }
                      >
                        {complaint.state === 'detected'
                          ? 'Detected by rule'
                          : complaint.state}
                      </Badge>
                    </summary>
                    <div>
                      {complaint.evidence.map((evidence, index) => (
                        <ExactSourceExcerpt
                          key={`${evidence.start}:${index}`}
                          evidence={evidence}
                          original={review.text!}
                        />
                      ))}
                      <p>
                        {complaint.state === 'unknown'
                          ? 'No qualifying rule evidence; complaint absence is not established.'
                          : complaint.reason}
                      </p>
                      <p className="reliability-note">
                        {complaint.reliability_warning}
                      </p>
                      <small>
                        {complaint.rule_version} · {complaint.taxonomy_version}{' '}
                        · learned component{' '}
                        {complaint.learned_model_state.replaceAll('_', ' ')}
                      </small>
                    </div>
                  </details>
                ))}
              </div>
            </div>
          ) : (
            <div className="withheld-review">
              <LockKeyhole size={23} aria-hidden="true" />
              <div>
                <h3>Review body unavailable</h3>
                <Badge>{review.text_state.replaceAll('_', ' ')}</Badge>
                <p>
                  The backend withholds this body. Metadata remains accessible;
                  text and evidence are not reconstructed.
                </p>
                <div className="withheld-categories">
                  {review.complaints.map((complaint) => (
                    <span key={complaint.category}>
                      {categories[complaint.category]}:{' '}
                      <strong>{complaint.state}</strong>
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}
          <details className="candidate-policy">
            <summary>Review provenance</summary>
            <p>
              {review.context.dataset_revision} · {review.context.rule_version}{' '}
              · snapshot <code>{review.context.snapshot_id}</code>
            </p>
            <p>
              Review ID: <code>{review.review_id}</code> · product ID:{' '}
              <code>{review.product_id}</code> · body state: {review.text_state}
              . Rule matches are customer-reported evidence, not verified
              incidents or model explanations.
            </p>
          </details>
        </>
      )}
    </section>
  );
}
