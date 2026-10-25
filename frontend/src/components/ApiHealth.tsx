import { useEffect, useState } from 'react';
import * as Tooltip from '@radix-ui/react-tooltip';
import { Info, RefreshCw } from 'lucide-react';
import { fetchHealth } from '../api/health';

type HealthState =
  | { status: 'checking' }
  | { status: 'online'; version: string }
  | { status: 'offline'; message: string };

export function ApiHealth() {
  const [state, setState] = useState<HealthState>({ status: 'checking' });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    fetchHealth(controller.signal).then(
      (health) => {
        if (!controller.signal.aborted)
          setState({ status: 'online', version: health.version });
      },
      (error: unknown) => {
        if (!controller.signal.aborted) {
          setState({
            status: 'offline',
            message:
              error instanceof Error ? error.message : 'Health check failed.',
          });
        }
      },
    );
    return () => controller.abort();
  }, [attempt]);

  return (
    <div className="api-health">
      <div className="health-label" role="status" aria-live="polite">
        <span className={`status-dot ${state.status}`} aria-hidden="true" />
        {state.status === 'checking' && 'Checking API…'}
        {state.status === 'online' && `API connected · v${state.version}`}
        {state.status === 'offline' && 'API unavailable'}
      </div>
      <Tooltip.Root>
        <Tooltip.Trigger asChild>
          <button className="icon-button" aria-label="About service health">
            <Info size={16} aria-hidden="true" />
          </button>
        </Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content className="tooltip" sideOffset={8}>
            Backend connectivity only. ML availability is checked when a review
            is analysed.
            <Tooltip.Arrow className="tooltip-arrow" />
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
      <button
        className="icon-button"
        aria-label="Recheck API connection"
        disabled={state.status === 'checking'}
        onClick={() => {
          setState({ status: 'checking' });
          setAttempt((value) => value + 1);
        }}
      >
        <RefreshCw size={16} aria-hidden="true" />
      </button>
      {state.status === 'offline' && (
        <p className="health-error">
          {state.message} Start the backend, then recheck.
        </p>
      )}
    </div>
  );
}
