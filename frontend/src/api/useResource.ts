import { useEffect, useState } from 'react';

/** Abort stale reads and keep old responses out of a newly selected resource. */
export function useResource<T>(load: (signal: AbortSignal) => Promise<T>) {
  const [attempt, setAttempt] = useState(0);
  const [response, setResponse] = useState<{
    load: typeof load;
    attempt: number;
    data?: T;
    error?: string;
  }>();
  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal).then(
      (data) => {
        if (!controller.signal.aborted) setResponse({ load, attempt, data });
      },
      (failure: unknown) => {
        if (!controller.signal.aborted)
          setResponse({
            load,
            attempt,
            error:
              failure instanceof Error
                ? failure.message
                : 'Information unavailable.',
          });
      },
    );
    return () => controller.abort();
  }, [load, attempt]);
  const current =
    response?.load === load && response.attempt === attempt
      ? response
      : undefined;
  return {
    data: current?.data,
    error: current?.error,
    loading: !current,
    retry: () => setAttempt((value) => value + 1),
  };
}
