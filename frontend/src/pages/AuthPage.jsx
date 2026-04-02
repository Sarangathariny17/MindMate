import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../AuthContext.jsx'
import api from '../api.js'

export default function AuthPage() {
  const nav = useNavigate()
  const { login } = useAuth()
  const [mode, setMode] = useState('login')
  const [name, setName] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const submit = async () => {
    setError('')
    if (!name.trim() || !password.trim()) { setError('Please fill in all fields.'); return }
    setLoading(true)
    try {
      const { data } = await api.post(mode === 'login' ? '/auth/login' : '/auth/signup', {
        name: name.trim(), password,
      })
      login(data)
      nav('/')
    } catch (e) {
      setError(e.response?.data?.detail || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={s.page}>
      {/* Floating petals */}
      {[...Array(8)].map((_, i) => (
        <div key={i} style={{
          ...s.petal,
          left: `${10 + i * 12}%`,
          animationDelay: `${i * 1.2}s`,
          animationDuration: `${8 + i * 1.5}s`,
          fontSize: `${14 + (i % 3) * 6}px`,
          opacity: 0.25 + (i % 3) * 0.1,
        }}>✿</div>
      ))}

      <div style={s.layout}>
        {/* Left: branding panel */}
        <div style={s.brandPanel}>
          <div style={s.brandInner}>
            <div style={s.brandGlyph}>✦</div>
            <h1 style={s.brandTitle}>Serene</h1>
            <p style={s.brandTagline}>Your sanctuary for the mind.</p>
            <div style={s.brandDivider} />
            <p style={s.brandDesc}>
              A safe, private space to speak freely — guided by an AI therapist who listens deeply,
              reads your emotional state in real time, and responds with warmth and care.
            </p>
            <div style={s.features}>
              {['Real-time emotion detection', 'Voice-powered sessions', 'Persistent memory across sessions', 'Private & secure'].map(f => (
                <div key={f} style={s.featureItem}>
                  <span style={s.featureDot}>◈</span>
                  <span>{f}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: auth card */}
        <div style={s.authPanel}>
          <div style={s.card} className="fade-up">
            <div style={s.cardTop}>
              <h2 style={s.cardTitle}>
                {mode === 'login' ? 'Welcome back' : 'Begin your journey'}
              </h2>
              <p style={s.cardSub}>
                {mode === 'login' ? 'Sign in to continue your sessions' : 'Create your private sanctuary'}
              </p>
            </div>

            {/* Tabs */}
            <div style={s.tabs}>
              {[['login','Sign In'], ['signup','Sign Up']].map(([m, label]) => (
                <button key={m} onClick={() => { setMode(m); setError('') }}
                  style={{ ...s.tab, ...(mode === m ? s.tabActive : {}) }}>
                  {label}
                </button>
              ))}
            </div>

            <div style={s.fields}>
              <div style={s.field}>
                <label style={s.label}>Your Name</label>
                <input
                  style={s.input}
                  type="text"
                  placeholder="How shall we call you?"
                  value={name}
                  onChange={e => setName(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && submit()}
                  autoFocus
                />
              </div>
              <div style={s.field}>
                <label style={s.label}>Password</label>
                <input
                  style={s.input}
                  type="password"
                  placeholder="Your private key"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && submit()}
                />
              </div>
            </div>

            {error && <div style={s.errorBox}>{error}</div>}

            <button onClick={submit} disabled={loading} style={s.submitBtn}>
              {loading ? (
                <span style={s.loadingText}>
                  <span style={s.loadSpinner} />
                  Please wait…
                </span>
              ) : mode === 'login' ? 'Enter Serene →' : 'Create Sanctuary →'}
            </button>

            <p style={s.switchLine}>
              {mode === 'login' ? "New here? " : "Already a member? "}
              <button onClick={() => { setMode(mode === 'login' ? 'signup' : 'login'); setError('') }}
                style={s.switchBtn}>
                {mode === 'login' ? 'Create account' : 'Sign in'}
              </button>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

const s = {
  page: {
    minHeight: '100vh', display: 'flex', alignItems: 'stretch',
    background: 'var(--cream)', overflow: 'hidden', position: 'relative',
  },
  petal: {
    position: 'fixed', top: '-20px', color: 'var(--rose-deep)',
    animation: 'petals linear infinite', pointerEvents: 'none', zIndex: 0,
    userSelect: 'none',
  },
  layout: { display: 'flex', width: '100%', minHeight: '100vh', zIndex: 1 },

  brandPanel: {
    flex: 1, background: 'linear-gradient(160deg, #2d1f2e 0%, #1a1520 60%, #0e0b12 100%)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    padding: '60px 56px', position: 'relative', overflow: 'hidden',
  },
  brandInner: { maxWidth: 420, zIndex: 1 },
  brandGlyph: { fontSize: 40, color: 'var(--gold)', marginBottom: 16 },
  brandTitle: {
    fontFamily: 'var(--font-display)', fontSize: 72, fontWeight: 300,
    color: '#f5ede0', letterSpacing: '0.02em', lineHeight: 1, marginBottom: 12,
  },
  brandTagline: {
    fontFamily: 'var(--font-display)', fontSize: 22, fontWeight: 300,
    color: 'var(--gold-light)', fontStyle: 'italic', marginBottom: 32,
  },
  brandDivider: {
    width: 48, height: 1,
    background: 'linear-gradient(90deg, var(--gold), transparent)',
    marginBottom: 28,
  },
  brandDesc: {
    fontSize: 15, color: 'rgba(212,200,220,0.7)', lineHeight: 1.75,
    marginBottom: 36,
  },
  features: { display: 'flex', flexDirection: 'column', gap: 12 },
  featureItem: {
    display: 'flex', alignItems: 'center', gap: 12,
    fontSize: 14, color: 'rgba(212,200,220,0.6)',
  },
  featureDot: { color: 'var(--rose-deep)', fontSize: 12, flexShrink: 0 },

  authPanel: {
    width: 480, flexShrink: 0, display: 'flex', alignItems: 'center',
    justifyContent: 'center', padding: '40px 48px',
    background: 'var(--cream)',
  },
  card: { width: '100%' },
  cardTop: { marginBottom: 28 },
  cardTitle: {
    fontFamily: 'var(--font-display)', fontSize: 38, fontWeight: 400,
    color: 'var(--charcoal)', marginBottom: 6,
  },
  cardSub: { fontSize: 14, color: 'var(--mist)' },

  tabs: {
    display: 'flex', background: 'var(--parchment)',
    borderRadius: 999, padding: 4, gap: 2, marginBottom: 28,
  },
  tab: {
    flex: 1, padding: '10px 0', border: 'none', borderRadius: 999,
    fontSize: 14, fontWeight: 500, cursor: 'pointer',
    background: 'transparent', color: 'var(--mist)',
    fontFamily: 'var(--font-body)', transition: 'all 0.2s',
  },
  tabActive: {
    background: 'white', color: 'var(--charcoal)',
    boxShadow: '0 2px 12px rgba(100,60,80,0.10)',
  },

  fields: { display: 'flex', flexDirection: 'column', gap: 18, marginBottom: 8 },
  field: { display: 'flex', flexDirection: 'column', gap: 7 },
  label: {
    fontSize: 11, fontWeight: 600, color: 'var(--dusty)',
    letterSpacing: '0.08em', textTransform: 'uppercase',
  },
  input: {
    padding: '14px 18px', border: '1.5px solid var(--rose)',
    borderRadius: 'var(--radius-sm)', fontSize: 15,
    color: 'var(--charcoal)', background: 'white',
    fontFamily: 'var(--font-body)', outline: 'none',
    transition: 'border-color 0.2s, box-shadow 0.2s',
  },

  errorBox: {
    marginTop: 16, padding: '12px 16px',
    background: 'rgba(201,137,138,0.1)',
    border: '1px solid rgba(201,137,138,0.3)',
    borderRadius: 'var(--radius-sm)', fontSize: 13,
    color: 'var(--rose-deep)', textAlign: 'center',
  },

  submitBtn: {
    marginTop: 24, width: '100%', padding: '16px 0',
    background: 'linear-gradient(135deg, #2d1f2e 0%, #4a2d3e 100%)',
    border: 'none', borderRadius: 999, fontSize: 15, fontWeight: 500,
    color: 'var(--gold-light)', cursor: 'pointer',
    fontFamily: 'var(--font-body)', letterSpacing: '0.03em',
    boxShadow: '0 6px 24px rgba(45,31,46,0.25)',
    transition: 'opacity 0.2s, transform 0.15s',
    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
  },
  loadingText: { display: 'flex', alignItems: 'center', gap: 10 },
  loadSpinner: {
    width: 14, height: 14, border: '2px solid rgba(232,213,168,0.3)',
    borderTop: '2px solid var(--gold-light)', borderRadius: '50%',
    animation: 'spin 0.8s linear infinite',
    display: 'inline-block',
  },

  switchLine: { marginTop: 20, fontSize: 13, color: 'var(--mist)', textAlign: 'center' },
  switchBtn: {
    background: 'none', border: 'none', color: 'var(--rose-deep)',
    fontWeight: 600, cursor: 'pointer', fontSize: 13,
    fontFamily: 'var(--font-body)', textDecoration: 'underline',
  },
}
