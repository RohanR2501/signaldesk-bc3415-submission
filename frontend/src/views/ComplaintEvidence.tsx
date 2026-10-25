import { Check, ChevronDown, ScanText } from 'lucide-react';
import type {
  ComplaintDetection,
  ReviewExcerpt,
} from '../api/contracts.generated';
import { categories } from '../lib';
import { Badge } from '../components/ui';

const stateLabels = {
  detected: 'Detected',
  unknown: 'Unknown',
  abstained: 'Abstained',
  unavailable: 'Unavailable',
};

export function ComplaintResult({
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
