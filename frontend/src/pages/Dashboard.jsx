import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../AuthContext.jsx'
import api from '../api.js'

const MOOD_QUOTES = [
  "Every feeling deserves a witness.",
  "You are allowed to take up space.",
  "Healing is not linear — and that's okay.",
  "What you feel is always valid.",
  "Stillness is its own kind of strength.",
]

export default function Dashboard() {
  const nav = useNavigate()
  const { user, logout } = useAuth()
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const quote = MOOD_QUOTES[new Date().getDay() % MOOD_QUOTES.length]

  useEffect(() => {
    api.get('/sessions/history')
      .then(r => setSessions(r.data))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const fmt = iso => iso
    ? new Date(iso).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })
    : '—'

  const fmtTime = iso => iso
    ? new Date(iso).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
    : ''

  return (
    <div style={s.page}>
      {/* Decorative background elements */}
      <div style={s.bgCircle1} />
      <div style={s.bgCircle2} />

      {/* Header */}
      <header style={s.header}>
        <div style={s.logoMark}>
          <span style={s.logoGlyph}>✦</span>
          <span style={s.logoText}>Serene</span>
        </div>
        <nav style={s.nav}>
          <span style={s.navUser}>
            <span style={s.navAvatar}>{(user?.name || 'U')[0].toUpperCase()}</span>
            {user?.name}
          </span>
          <button onClick={logout} style={s.logoutBtn}>Sign out</button>
        </nav>
      </header>

      <main style={s.main}>
        {/* Hero section */}
        <section style={s.hero} className="fade-up">
          <p style={s.heroGreeting}>Good to see you,</p>
          <h1 style={s.heroName}>{user?.name}</h1>
          <p style={s.heroQuote}>❝ {quote} ❞</p>
          <button onClick={() => nav('/session')} style={s.startBtn}>
            <span style={s.startBtnInner}>
              <span style={s.startBtnGlyph}>✦</span>
              Begin a New Session
            </span>
          </button>
          <p style={s.startHint}>Voice-powered · Emotion-aware · Deeply private</p>
        </section>

        {/* Stats row */}
        {sessions.length > 0 && (
          <div style={s.statsRow} className="fade-up">
            {[
              ['Sessions', sessions.length],
              ['Completed', sessions.filter(s => s.status === 'completed').length],
              ['Messages', sessions.reduce((a, s) => a + (s.message_count || 0), 0)],
            ].map(([label, val]) => (
              <div key={label} style={s.statCard}>
                <div style={s.statVal}>{val}</div>
                <div style={s.statLabel}>{label}</div>
              </div>
            ))}
          </div>
        )}

        {/* Past sessions */}
        <section style={s.section}>
          <div style={s.sectionHeader}>
            <h2 style={s.sectionTitle}>Your Journey</h2>
            <span style={s.sectionSub}>{sessions.length} session{sessions.length !== 1 ? 's' : ''}</span>
          </div>

          {loading && (
            <div style={s.loadingRow}>
              {[0,1,2].map(i => (
                <div key={i} style={{ ...s.skeleton, animationDelay: `${i * 0.15}s` }} />
              ))}
            </div>
          )}

          {!loading && sessions.length === 0 && (
            <div style={s.emptyState}>
              <div style={s.emptyGlyph}>✿</div>
              <p style={s.emptyTitle}>Your journey begins here</p>
              <p style={s.emptyDesc}>Start your first session to see it reflected here.</p>
            </div>
          )}

          <div style={s.sessionGrid}>
            {!loading && sessions.map((sess, i) => (
              <div key={sess.session_id} style={s.sessionCard} className="fade-up"
                   data-style={`animation-delay: ${i * 0.06}s`}>
                <div style={s.sessionCardTop}>
                  <div>
                    <div style={s.sessionDate}>{fmt(sess.started_at)}</div>
                    <div style={s.sessionTime}>{fmtTime(sess.started_at)}</div>
                  </div>
                  <span style={{
                    ...s.badge,
                    background: sess.status === 'completed'
                      ? 'rgba(120,170,100,0.12)' : 'rgba(201,169,110,0.12)',
                    color: sess.status === 'completed' ? '#5a8a4a' : '#a07820',
                    border: `1px solid ${sess.status === 'completed' ? 'rgba(120,170,100,0.25)' : 'rgba(201,169,110,0.25)'}`,
                  }}>
                    {sess.status === 'completed' ? '✓ Complete' : '◌ Active'}
                  </span>
                </div>
                {sess.summary && (
                  <p style={s.sessionSummary}>❝ {sess.summary.slice(0, 140)}{sess.summary.length > 140 ? '…' : ''} ❞</p>
                )}
                <div style={s.sessionFooter}>
                  <span style={s.sessionMeta}>{sess.message_count || 0} exchanges</span>
                  {sess.ended_at && (
                    <span style={s.sessionMeta}>Ended {fmtTime(sess.ended_at)}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  )
}

const s = {
  page: {
    minHeight: '100vh', background: 'var(--cream)',
    position: 'relative', overflow: 'hidden',
  },
  bgCircle1: {
    position: 'fixed', top: '-15%', right: '-10%',
    width: 600, height: 600, borderRadius: '50%',
    background: 'radial-gradient(circle, rgba(232,196,196,0.18) 0%, transparent 70%)',
    pointerEvents: 'none',
  },
  bgCircle2: {
    position: 'fixed', bottom: '-20%', left: '-10%',
    width: 500, height: 500, borderRadius: '50%',
    background: 'radial-gradient(circle, rgba(168,184,154,0.14) 0%, transparent 70%)',
    pointerEvents: 'none',
  },

  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '20px 48px',
    background: 'rgba(253,248,242,0.85)',
    backdropFilter: 'blur(20px)',
    borderBottom: '1px solid rgba(232,196,196,0.25)',
    position: 'sticky', top: 0, zIndex: 10,
  },
  logoMark: { display: 'flex', alignItems: 'center', gap: 10 },
  logoGlyph: { fontSize: 20, color: 'var(--gold)' },
  logoText: {
    fontFamily: 'var(--font-display)', fontSize: 28, fontWeight: 400,
    color: 'var(--charcoal)', letterSpacing: '0.03em',
  },
  nav: { display: 'flex', alignItems: 'center', gap: 20 },
  navUser: {
    display: 'flex', alignItems: 'center', gap: 10,
    fontSize: 14, color: 'var(--dusty)',
  },
  navAvatar: {
    width: 32, height: 32, borderRadius: '50%',
    background: 'linear-gradient(135deg, var(--rose), var(--rose-deep))',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 13, color: 'white', fontWeight: 600,
  },
  logoutBtn: {
    padding: '8px 18px',
    border: '1px solid rgba(232,196,196,0.5)',
    borderRadius: 999, background: 'transparent',
    fontSize: 13, color: 'var(--dusty)', cursor: 'pointer',
    fontFamily: 'var(--font-body)', transition: 'all 0.2s',
  },

  main: {
    maxWidth: 780, margin: '0 auto', padding: '56px 24px 80px',
    position: 'relative', zIndex: 1,
  },

  hero: { textAlign: 'center', marginBottom: 56 },
  heroGreeting: {
    fontFamily: 'var(--font-display)', fontSize: 20, fontStyle: 'italic',
    color: 'var(--mist)', marginBottom: 4,
  },
  heroName: {
    fontFamily: 'var(--font-display)', fontSize: 64, fontWeight: 300,
    color: 'var(--charcoal)', marginBottom: 20, letterSpacing: '-0.01em',
  },
  heroQuote: {
    fontFamily: 'var(--font-display)', fontSize: 18, fontStyle: 'italic',
    color: 'var(--dusty)', marginBottom: 40, lineHeight: 1.6,
  },
  startBtn: {
    display: 'inline-block', padding: '0',
    background: 'linear-gradient(135deg, #2d1f2e 0%, #4a2d3e 100%)',
    border: 'none', borderRadius: 999,
    cursor: 'pointer', fontFamily: 'var(--font-body)',
    boxShadow: '0 8px 32px rgba(45,31,46,0.22)',
    transition: 'transform 0.2s, box-shadow 0.2s',
  },
  startBtnInner: {
    display: 'flex', alignItems: 'center', gap: 10,
    padding: '18px 44px', fontSize: 16, fontWeight: 500,
    color: 'var(--gold-light)', letterSpacing: '0.03em',
  },
  startBtnGlyph: { fontSize: 14, color: 'var(--gold)' },
  startHint: { marginTop: 14, fontSize: 12, color: 'var(--mist)' },

  statsRow: {
    display: 'flex', gap: 16, marginBottom: 48,
    justifyContent: 'center',
  },
  statCard: {
    flex: 1, maxWidth: 160, textAlign: 'center',
    background: 'rgba(255,255,255,0.7)',
    backdropFilter: 'blur(12px)',
    border: '1px solid rgba(232,196,196,0.3)',
    borderRadius: 'var(--radius)', padding: '20px 16px',
    boxShadow: 'var(--shadow-card)',
  },
  statVal: {
    fontFamily: 'var(--font-display)', fontSize: 40, fontWeight: 400,
    color: 'var(--charcoal)', marginBottom: 4,
  },
  statLabel: { fontSize: 12, color: 'var(--mist)', letterSpacing: '0.06em', textTransform: 'uppercase' },

  section: {},
  sectionHeader: {
    display: 'flex', alignItems: 'baseline', justifyContent: 'space-between',
    marginBottom: 24,
  },
  sectionTitle: {
    fontFamily: 'var(--font-display)', fontSize: 32, fontWeight: 400,
    color: 'var(--charcoal)',
  },
  sectionSub: { fontSize: 13, color: 'var(--mist)' },

  loadingRow: { display: 'flex', flexDirection: 'column', gap: 12 },
  skeleton: {
    height: 100, borderRadius: 'var(--radius)',
    background: 'linear-gradient(90deg, var(--parchment) 25%, var(--rose-blush) 50%, var(--parchment) 75%)',
    backgroundSize: '200% 100%', animation: 'shimmer 1.5s infinite',
  },

  emptyState: {
    textAlign: 'center', padding: '64px 0',
  },
  emptyGlyph: { fontSize: 48, color: 'var(--rose)', marginBottom: 16 },
  emptyTitle: {
    fontFamily: 'var(--font-display)', fontSize: 24, fontWeight: 400,
    color: 'var(--charcoal)', marginBottom: 8,
  },
  emptyDesc: { fontSize: 14, color: 'var(--mist)' },

  sessionGrid: { display: 'flex', flexDirection: 'column', gap: 14 },
  sessionCard: {
    background: 'rgba(255,255,255,0.75)',
    backdropFilter: 'blur(16px)',
    border: '1px solid rgba(232,196,196,0.28)',
    borderRadius: 'var(--radius)', padding: '22px 26px',
    boxShadow: 'var(--shadow-card)',
    transition: 'transform 0.2s, box-shadow 0.2s',
  },
  sessionCardTop: {
    display: 'flex', justifyContent: 'space-between',
    alignItems: 'flex-start', marginBottom: 12,
  },
  sessionDate: {
    fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 400,
    color: 'var(--charcoal)',
  },
  sessionTime: { fontSize: 12, color: 'var(--mist)', marginTop: 2 },
  badge: {
    fontSize: 11, fontWeight: 600, padding: '5px 12px',
    borderRadius: 999, letterSpacing: '0.04em',
  },
  sessionSummary: {
    fontFamily: 'var(--font-display)', fontSize: 15, fontStyle: 'italic',
    color: 'var(--dusty)', lineHeight: 1.65, marginBottom: 12,
  },
  sessionFooter: {
    display: 'flex', gap: 20,
  },
  sessionMeta: { fontSize: 12, color: 'var(--mist)' },
}
