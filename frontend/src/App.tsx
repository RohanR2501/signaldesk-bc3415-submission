import { useEffect, useRef, useState } from 'react';
import * as Tooltip from '@radix-ui/react-tooltip';
import {
  Activity,
  ArrowUpRight,
  BarChart3,
  ChevronRight,
  Menu,
  MessageSquareText,
  PackageSearch,
  ShieldCheck,
  X,
} from 'lucide-react';
import { ApiHealth } from './components/ApiHealth';
import { Overview } from './views/Overview';
import { Analyzer } from './views/Analyzer';
import { Workbench } from './views/Workbench';

const views = [
  {
    id: 'overview',
    title: 'Executive Overview',
    shortTitle: 'Executive overview',
    icon: BarChart3,
    description:
      'A clearer view of customer feedback. A stronger starting point for investigation.',
  },
  {
    id: 'analyzer',
    title: 'Live Review Analyzer',
    shortTitle: 'Live review analyzer',
    icon: MessageSquareText,
    description:
      'Understand the sentiment. Read the evidence. Keep human judgment at the centre.',
  },
  {
    id: 'incidents',
    title: 'Product Investigation Workbench',
    shortTitle: 'Investigation workbench',
    icon: PackageSearch,
    description:
      'A workspace for investigating customer-reported and suspected product issues.',
  },
] as const;

function currentView(hash: string) {
  return (
    views.find((view) => `#/${view.id}` === hash.split('?')[0]) ?? views[0]
  );
}

export function App() {
  const [location, setLocation] = useState(() => window.location.hash);
  const view = currentView(location);
  const [menuOpen, setMenuOpen] = useState(false);
  const heading = useRef<HTMLHeadingElement>(null);
  const menuButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const onNavigate = () => {
      setLocation(window.location.hash);
      setMenuOpen(false);
      requestAnimationFrame(() => {
        heading.current?.focus();
        window.scrollTo(0, 0);
      });
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setMenuOpen(false);
        menuButton.current?.focus();
      }
    };
    const onResize = () => {
      if (window.innerWidth > 900) setMenuOpen(false);
    };
    window.addEventListener('hashchange', onNavigate);
    window.addEventListener('keydown', onKey);
    window.addEventListener('resize', onResize);
    return () => {
      window.removeEventListener('hashchange', onNavigate);
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('resize', onResize);
    };
  }, []);
  useEffect(() => {
    document.title = `${view.title} · SignalDesk`;
  }, [view]);
  return (
    <Tooltip.Provider delayDuration={200}>
      <a
        href="#main"
        className="skip-link"
        onClick={(event) => {
          event.preventDefault();
          document.getElementById('main')?.focus();
        }}
      >
        Skip to content
      </a>
      <div className="app-layout">
        <aside
          id="workspace-navigation"
          className={`sidebar ${menuOpen ? 'is-open' : ''}`}
        >
          <a className="brand" href="#/overview" aria-label="SignalDesk home">
            <span className="brand-symbol">
              <Activity size={23} aria-hidden="true" />
            </span>
            <span>
              SignalDesk
              <span className="brand-caption">RETAIL INTELLIGENCE</span>
            </span>
          </a>
          <div className="workspace-selector">
            <span className="workspace-avatar">A</span>
            <div>
              Appliances workspace<small>Amazon Reviews · Historical</small>
            </div>
          </div>
          <p className="nav-label">INTELLIGENCE</p>
          <nav aria-label="Main navigation">
            {views.map(({ id, shortTitle, icon: Icon }) => (
              <a
                key={id}
                href={`#/${id}`}
                aria-current={view.id === id ? 'page' : undefined}
                onClick={() => setMenuOpen(false)}
              >
                <Icon size={18} aria-hidden="true" />
                <span>{shortTitle}</span>
              </a>
            ))}
          </nav>
          <div className="sidebar-bottom">
            <div className="sidebar-note">
              <ShieldCheck size={20} aria-hidden="true" />
              <strong>Evidence before action.</strong>
              <p>
                Customer-reported signals support human investigation. They do
                not verify defects.
              </p>
            </div>
            <div className="sidebar-footer">
              <span className="project-avatar">SD</span>
              <div>
                SignalDesk project<small>NTU · BC3415 Option A</small>
              </div>
            </div>
          </div>
        </aside>
        {menuOpen && (
          <button
            className="nav-backdrop"
            aria-label="Close navigation"
            onClick={() => setMenuOpen(false)}
          />
        )}
        <div className="workspace">
          <header className="topbar">
            <div className="topbar-location">
              <button
                ref={menuButton}
                className="icon-button menu-button"
                aria-expanded={menuOpen}
                aria-controls="workspace-navigation"
                aria-label={menuOpen ? 'Close navigation' : 'Open navigation'}
                onClick={() => setMenuOpen(!menuOpen)}
              >
                {menuOpen ? <X size={20} /> : <Menu size={20} />}
              </button>
              <span>Workspace</span>
              <ChevronRight size={13} aria-hidden="true" />
              <strong>Intelligence</strong>
            </div>
            <div className="topbar-status">
              <ApiHealth />
              <span className="topbar-divider" />
              <span className="local-label">LOCAL WORKSPACE</span>
            </div>
          </header>
          <main
            id="main"
            className="main-content"
            tabIndex={-1}
            inert={menuOpen}
          >
            <div className="page-heading">
              <div>
                <p className="eyebrow">CUSTOMER SIGNALS, IN CONTEXT</p>
                <h1 ref={heading} tabIndex={-1}>
                  {view.title}
                </h1>
                <p className="page-description">{view.description}</p>
              </div>
              {view.id === 'overview' && (
                <a className="button secondary" href="#/analyzer">
                  <MessageSquareText size={16} aria-hidden="true" />
                  Analyse a review
                  <ArrowUpRight size={15} aria-hidden="true" />
                </a>
              )}
            </div>
            {view.id === 'overview' ? (
              <Overview />
            ) : view.id === 'analyzer' ? (
              <Analyzer />
            ) : (
              <Workbench hash={location} />
            )}
            <footer className="main-footer">
              <span>
                <Activity size={13} aria-hidden="true" />
                SignalDesk · Evidence for better judgment
              </span>
              <span>NTU BC3415 · Local decision support</span>
            </footer>
          </main>
        </div>
      </div>
    </Tooltip.Provider>
  );
}
