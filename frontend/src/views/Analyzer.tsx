import { useEffect, useRef, useState, type FormEvent } from 'react';
import {
  ArrowRight,
  Check,
  ChevronDown,
  Cpu,
  FileText,
  Fingerprint,
  Frown,
  Info,
  LoaderCircle,
  Meh,
  ScanText,
  ShieldCheck,
  Smile,
  Sparkles,
} from 'lucide-react';
import { api } from '../api/client';
import type {
  AnalyzeResponse,
  ComplaintDetection,
  ReviewExcerpt,
} from '../api/contracts.generated';
import { categories, number } from '../lib';
import { Badge, ErrorNotice, SectionTitle } from '../components/ui';

const stateLabels = {
  detected: 'Detected',
  unknown: 'Unknown',
  abstained: 'Abstained',
  unavailable: 'Unavailable',
};
const MAX_CHARACTERS = 50_000;

export function Analyzer() {
  const [text, setText] = useState('');
  const [submitted, setSubmitted] = useState('');
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string>();
  const [busy, setBusy] = useState(false);
  const inFlight = useRef(false);
  const request = useRef<AbortController | null>(null);
  const resultHeading = useRef<HTMLHeadingElement>(null);
  const count = Array.from(text).length;
  const tooLong = count > MAX_CHARACTERS;
  useEffect(
    () => () => {
      request.current?.abort();
    },
    [],
  );

  const analyse = async (event: FormEvent) => {
    event.preventDefault();
    if (inFlight.current || !text.trim() || tooLong) return;
    inFlight.current = true;
    setBusy(true);
    setError(undefined);
    setResult(null);
    setSubmitted(text);
    const controller = new AbortController();
    request.current = controller;
    try {
      const response = await api.analyze(text, controller.signal);
      if (!controller.signal.aborted) {
        setResult(response);
        requestAnimationFrame(() =>
          resultHeading.current?.focus({ preventScroll: true }),
        );
      }
    } catch (failure) {
      if (!controller.signal.aborted)
        setError(
          failure instanceof Error
            ? failure.message
            : 'The analysis could not be completed.',
        );
    } finally {
      inFlight.current = false;
      if (!controller.signal.aborted) setBusy(false);
    }
  };

  return (
    <div className="view-stack">
      <div className="analyzer-intro">
        <span className="badge teal">
          <span className="status-dot online" />
          On-demand analysis
        </span>
        <p>One review. Two independent perspectives.</p>
        <span>
          Sentiment describes tone. Complaints describe customer-reported
          experiences.
        </span>
      </div>
      <div className="analyzer-layout">
        <section className="panel editor-panel">
          <SectionTitle
            title="Start with the customer's words"
            subtitle="Paste the original review, including any context or resolution."
            action={<FileText size={19} aria-hidden="true" />}
          />
          <form
            onSubmit={(event) => {
              void analyse(event);
            }}
          >
            <label htmlFor="review-text" className="editor-label">
              Review text
            </label>
            <textarea
              id="review-text"
              value={text}
              readOnly={busy}
              aria-invalid={tooLong}
              aria-describedby="editor-help character-count"
              autoComplete="off"
              spellCheck={false}
              placeholder="Paste a customer review here…"
              onChange={(event) => {
                setText(event.target.value);
                setResult(null);
                setError(undefined);
              }}
            />
            <div className="editor-meta">
              <span id="editor-help">
                Original text only · no rating required
              </span>
              <span
                id="character-count"
                className={tooLong ? 'invalid-text' : ''}
              >
                {number(count)} / {number(MAX_CHARACTERS)}
              </span>
            </div>
            {tooLong && (
              <p className="input-error" role="alert">
                Use at most 50,000 Unicode characters.
              </p>
            )}
            <div className="editor-actions">
              <button
                type="button"
                className="button secondary"
                disabled={busy || !text}
                onClick={() => {
                  setText('');
                  setSubmitted('');
                  setResult(null);
                  setError(undefined);
                }}
              >
                Clear
              </button>
              <button
                type="submit"
                className="button primary"
                disabled={busy || !text.trim() || tooLong}
              >
                {busy ? (
                  <LoaderCircle
                    size={17}
                    className="spinner"
                    aria-hidden="true"
                  />
                ) : (
                  <Sparkles size={17} aria-hidden="true" />
                )}
                {busy ? 'Analysing…' : 'Analyse review'}
                {!busy && <ArrowRight size={16} aria-hidden="true" />}
              </button>
            </div>
          </form>
          <div className="privacy-note">
            <ShieldCheck size={17} aria-hidden="true" />
            <p>
              Text is sent to your local backend for this analysis. It is not
              saved by the UI and clears when you leave this view.
            </p>
          </div>
          <details className="editor-details">
            <summary>How the analysis works</summary>
            <p>
              The backend first tries saved weighted DistilBERT on GPU, then
              saved balanced logistic regression on CPU if needed. Frozen
              complaint rules inspect the full original text independently.
              Neither establishes a verified defect.
            </p>
            <p>
              Transformer sentiment may truncate long text to its saved token
              limit. No model confidence or sentiment explanation spans are
              supplied.
            </p>
          </details>
        </section>

        <div className="analysis-output" aria-busy={busy}>
          {busy && (
            <section className="panel processing-panel" role="status">
              <div className="processing-icon">
                <LoaderCircle
                  className="spinner"
                  size={28}
                  aria-hidden="true"
                />
              </div>
              <h2>Analysing your review</h2>
              <p>
                Local models may take several minutes to initialise on the first
                request. Keep this view open while the backend processes your
                text.
              </p>
              <span className="quiet-label">
                Waiting for actual model and rule outputs
              </span>
            </section>
          )}
          {error && <ErrorNotice message={error} />}
          {!busy && result && (
            <>
              <section
                className={`panel sentiment-panel ${result.sentiment.label ?? 'unavailable'}`}
              >
                <div className="section-title">
                  <div>
                    <p className="eyebrow">GLOBAL SENTIMENT</p>
                    <h2
                      ref={resultHeading}
                      tabIndex={-1}
                      className="sentiment-label"
                    >
                      {result.sentiment.label
                        ? result.sentiment.label.charAt(0).toUpperCase() +
                          result.sentiment.label.slice(1)
                        : stateLabels[result.sentiment.state]}
                    </h2>
                  </div>
                  <div className="sentiment-symbol">
                    {result.sentiment.label === 'positive' ? (
                      <Smile size={31} aria-hidden="true" />
                    ) : result.sentiment.label === 'negative' ? (
                      <Frown size={31} aria-hidden="true" />
                    ) : result.sentiment.label === 'neutral' ? (
                      <Meh size={31} aria-hidden="true" />
                    ) : (
                      <Cpu size={30} aria-hidden="true" />
                    )}
                  </div>
                </div>
                {result.sentiment.model_version ? (
                  <>
                    <p className="model-identity">
                      {result.sentiment.model_version.includes('weighted')
                        ? 'Weighted DistilBERT'
                        : result.sentiment.model_version.includes(
                              'logistic_regression_balanced',
                            )
                          ? 'Balanced logistic regression'
                          : result.sentiment.model_version}
                      <Badge
                        tone={
                          result.sentiment.device === 'cuda:0'
                            ? 'teal'
                            : 'neutral'
                        }
                      >
                        {result.sentiment.device === 'cuda:0'
                          ? 'GPU · CUDA'
                          : 'CPU'}
                      </Badge>
                    </p>
                    <p className="muted">
                      Model-predicted label · no confidence score supplied
                    </p>
                  </>
                ) : (
                  <p className="muted">
                    The sentiment models are unavailable. Complaint-rule results
                    remain available below.
                  </p>
                )}
                {result.sentiment.fallback_reason && (
                  <div className="fallback-note">
                    <Cpu size={16} aria-hidden="true" />
                    <span>
                      {result.sentiment.model_version
                        ? 'Automatic fallback used.'
                        : 'Transformer unavailable.'}{' '}
                      {result.sentiment.fallback_reason.replaceAll('_', ' ')}.
                    </span>
                  </div>
                )}
                {result.status === 'partial' && (
                  <div className="partial-note" role="status">
                    <Badge tone="amber">Partial analysis</Badge>
                    <span>
                      Sentiment is unavailable; keep the separate complaint
                      results.
                    </span>
                  </div>
                )}
              </section>

              <section className="panel complaint-results">
                <SectionTitle
                  title="Complaint evidence"
                  subtitle="Five categories, independent of the sentiment label"
                  action={<ScanText size={19} aria-hidden="true" />}
                />
                <p className="results-intro">
                  A detection is a customer-reported signal. Unknown means no
                  rule evidence was found.
                </p>
                <div>
                  {result.complaints.map((complaint) => (
                    <ComplaintResult
                      key={complaint.category}
                      complaint={complaint}
                      submitted={submitted}
                    />
                  ))}
                </div>
              </section>
              <details className="provenance-details">
                <summary>
                  <Fingerprint size={16} aria-hidden="true" />
                  Model identity, provenance & limitations
                </summary>
                <div>
                  <dl className="provenance-list">
                    <div>
                      <dt>Model</dt>
                      <dd>{result.sentiment.model_version ?? 'Unavailable'}</dd>
                    </div>
                    <div>
                      <dt>Preprocessing</dt>
                      <dd>{result.sentiment.preprocessing ?? 'Unavailable'}</dd>
                    </div>
                    <div>
                      <dt>Transformer token limit</dt>
                      <dd>
                        {result.sentiment.max_tokens ??
                          'Not applicable / unavailable'}
                      </dd>
                    </div>
                    <div>
                      <dt>Input</dt>
                      <dd>Submitted review body only</dd>
                    </div>
                    <div>
                      <dt>Review reference</dt>
                      <dd>{result.review_id}</dd>
                    </div>
                    <div>
                      <dt>Manifest SHA256</dt>
                      <dd>
                        {result.sentiment.manifest_sha256 ?? 'Unavailable'}
                      </dd>
                    </div>
                  </dl>
                  <ul>
                    {result.sentiment.attempts.map((attempt) => (
                      <li key={attempt.model}>
                        {attempt.model.replaceAll('_', ' ')}: {attempt.state}
                        {attempt.reason
                          ? ` (${attempt.reason.replaceAll('_', ' ')})`
                          : ''}
                      </li>
                    ))}
                  </ul>
                  <ul>
                    {result.warnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                </div>
              </details>
            </>
          )}
          {!busy && !result && !error && (
            <section className="analysis-guide">
              <span className="guide-icon">
                <ScanText size={28} aria-hidden="true" />
              </span>
              <h2>Make the evidence visible.</h2>
              <p>
                Read overall tone alongside specific complaints, without letting
                one erase the other.
              </p>
              <ol>
                <li>
                  <span>01</span>
                  <div>
                    <strong>Understand the sentiment</strong>
                    <p>
                      A real three-class prediction with its actual model
                      identity.
                    </p>
                  </div>
                </li>
                <li>
                  <span>02</span>
                  <div>
                    <strong>Inspect each complaint</strong>
                    <p>
                      Five independent categories, including explicit unknown
                      states.
                    </p>
                  </div>
                </li>
                <li>
                  <span>03</span>
                  <div>
                    <strong>Read the customer's evidence</strong>
                    <p>
                      Exact rule excerpts and resolution context for human
                      judgment.
                    </p>
                  </div>
                </li>
              </ol>
              <div className="guide-footer">
                <Info size={16} aria-hidden="true" />
                No prediction verifies a product defect.
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}

function ComplaintResult({
  complaint,
  submitted,
}: {
  complaint: ComplaintDetection;
  submitted: string;
}) {
  return (
    <details
      className={`complaint-result ${complaint.state}`}
      open={complaint.state === 'detected'}
    >
      <summary>
        <span className="category-icon">
          {complaint.state === 'detected' ? (
            <Check size={15} aria-hidden="true" />
          ) : (
            <ScanText size={15} aria-hidden="true" />
          )}
        </span>
        <strong>{categories[complaint.category]}</strong>
        <Badge tone={complaint.state === 'detected' ? 'amber' : 'neutral'}>
          {stateLabels[complaint.state]}
        </Badge>
        <ChevronDown size={15} aria-hidden="true" />
      </summary>
      <div className="complaint-body">
        {complaint.evidence.map((evidence, index) => (
          <ExactExcerpt
            key={`${evidence.start}-${evidence.rule_id}-${index}`}
            evidence={evidence}
            submitted={submitted}
          />
        ))}
        {complaint.state !== 'detected' && (
          <p>
            {complaint.state === 'unknown'
              ? 'No qualifying rule match. This does not confirm the absence of a complaint.'
              : complaint.state === 'abstained'
                ? 'The method withheld a decision. Human interpretation is needed.'
                : 'This category could not be assessed by the available method.'}
          </p>
        )}
        <p className="reliability-note">{complaint.reliability_warning}</p>
        <small>
          {complaint.rule_version} · {complaint.taxonomy_version}
          {complaint.learned_model_state === 'unavailable'
            ? ' · learned model unavailable'
            : ' · explicit rules only'}
        </small>
      </div>
    </details>
  );
}

function ExactExcerpt({
  evidence,
  submitted,
}: {
  evidence: ReviewExcerpt;
  submitted: string;
}) {
  // Backend offsets are Python Unicode code points; JS string.slice uses UTF-16.
  const points = Array.from(submitted);
  const exact =
    points.slice(evidence.start, evidence.end).join('') === evidence.text;
  const before = points
    .slice(Math.max(0, evidence.start - 60), evidence.start)
    .join('');
  const after = points
    .slice(evidence.end, Math.min(points.length, evidence.end + 60))
    .join('');
  return (
    <div className="evidence-item">
      <blockquote>
        {exact ? (
          <>
            {evidence.start > 60 && '…'}
            {before}
            <mark>{evidence.text}</mark>
            {after}
            {evidence.end + 60 < points.length && '…'}
          </>
        ) : (
          evidence.text
        )}
      </blockquote>
      {!exact && (
        <p className="input-error">
          Source offsets do not match this submitted text. Inspect the excerpt
          before interpreting it.
        </p>
      )}
      <div className="evidence-caption">
        <span>
          Exact rule match · [{evidence.start}, {evidence.end})
        </span>
        <code>{evidence.rule_id}</code>
      </div>
      {evidence.resolution_excerpt && (
        <div className="resolution-note">
          <Badge tone="teal">Reported resolved</Badge>
          <q>{evidence.resolution_excerpt}</q>
          <small>
            Original offsets [{evidence.resolution_start},{' '}
            {evidence.resolution_end}) · the occurrence remains reported
          </small>
        </div>
      )}
    </div>
  );
}
