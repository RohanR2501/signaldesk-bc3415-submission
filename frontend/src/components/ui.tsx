import type { ReactNode } from 'react';
import { AlertCircle, LoaderCircle, ArrowRight } from 'lucide-react';

export function ErrorNotice({
  message,
  retry,
}: {
  message: string;
  retry?: () => void;
}) {
  return (
    <div className="error-notice" role="alert">
      <AlertCircle size={19} aria-hidden="true" />
      <div>
        <strong>Unable to load this information</strong>
        <p>{message}</p>
      </div>
      {retry && (
        <button className="button secondary small" onClick={retry}>
          Try again
        </button>
      )}
    </div>
  );
}
export function Loading({
  children = 'Loading historical intelligence…',
}: {
  children?: ReactNode;
}) {
  return (
    <div className="loading-state" role="status">
      <LoaderCircle size={21} className="spinner" aria-hidden="true" />
      <span>{children}</span>
    </div>
  );
}
export function Badge({
  children,
  tone = 'neutral',
}: {
  children: ReactNode;
  tone?: 'neutral' | 'teal' | 'amber' | 'red';
}) {
  return <span className={`badge ${tone}`}>{children}</span>;
}
export function SectionTitle({
  title,
  subtitle,
  action,
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
}) {
  return (
    <div className="section-title">
      <div>
        <h2>{title}</h2>
        {subtitle && <p>{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}
export function RouteLink({
  href,
  children,
}: {
  href: string;
  children: ReactNode;
}) {
  return (
    <a className="text-link" href={href}>
      {children}
      <ArrowRight size={15} aria-hidden="true" />
    </a>
  );
}
