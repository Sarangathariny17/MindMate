import { useState, useEffect, useRef } from 'react'
import { useAuth } from './AuthContext.jsx'
import api from './api.js'

// ── Bubble sub-components ─────────────────────────────────────────────────────
function Bubble({ m, user }) {
  const isUser   = m.role === 'user'
  const isSystem = m.role === 'system'
  const fmt = ts => new Date(ts).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })

  if (isSystem) return (
    <div style={bs.system} className="fade-in">
      <span style={{ opacity: 0.4 }}>◦</span> {m.text}
    </div>
  )
  return (
    <div style={{ ...bs.row, justifyContent: isUser ? 'flex-end' : 'flex-start' }} className="fade-in">
      {!isUser && <div style={bs.avatarT}>✦</div>}
      <div style={{ ...bs.bubble, ...(isUser ? bs.bubbleUser : bs.bubbleT) }}>
        <div style={bs.label}>{isUser ? (user?.name || 'You') : 'Serene'}</div>
        <p style={bs.text}>{m.text}</p>
        <div style={bs.time}>{fmt(m.ts)}</div>
      </div>
      {isUser && (
        <div style={bs.avatarU}>{(user?.name || 'Y')[0].toUpperCase()}</div>
      )}
    </div>
  )
}

function TypingBubble() {
  return (
    <div style={{ ...bs.row, justifyContent: 'flex-start' }} className="fade-in">
      <div style={bs.avatarT}>✦</div>
      <div style={{ ...bs.bubble, ...bs.bubbleT, padding: '12px 16px' }}>
        <div style={bs.label}>Serene is thinking</div>
        <div style={{ display: 'flex', gap: 5, paddingTop: 4 }}>
          {[0, 1, 2].map(i => (
            <span key={i} style={{
              width: 7, height: 7, borderRadius: '50%',
              background: '#8a3030', display: 'inline-block',
              animation: 'typing 1.2s ease-in-out infinite',
              animationDelay: `${i * 180}ms`,
            }} />
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Main Widget ───────────────────────────────────────────────────────────────
export default function ChatWidget() {
  const { user } = useAuth()
  const [open, setOpen]           = useState(false)
  const [messages, setMessages]   = useState([])
  const [input, setInput]         = useState('')
  const [sessionId, setSessionId] = useState(null)
  const [loading, setLoading]     = useState(false)
  const [started, setStarted]     = useState(false)
  const [unread, setUnread]       = useState(0)
  const [shake, setShake]         = useState(false)

  const chatBottomRef = useRef()
  const inputRef      = useRef()
  const sessionInitRef = useRef(false)

  // Don't render if not logged in
  if (!user) return null

  // ── Open/close widget ─────────────────────────────────────────────────────
  const handleOpen = () => {
    setOpen(true)
    setUnread(0)
    // Start session on first open
    if (!sessionInitRef.current) {
      sessionInitRef.current = true
      initSession()
    }
    setTimeout(() => inputRef.current?.focus(), 300)
  }

  const handleClose = () => setOpen(false)

  // ── Session init ──────────────────────────────────────────────────────────
  const initSession = async () => {
    setLoading(true)
    try {
      const { data } = await api.post('/sessions/start')
      setSessionId(data.session_id)
      addMsg('therapist', data.greeting)
      setStarted(true)
    } catch {
      addMsg('system', 'Could not connect. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  // ── Send message ──────────────────────────────────────────────────────────
  const send = async () => {
    const text = input.trim()
    if (!text || !sessionId || loading) return

    setInput('')
    addMsg('user', text)
    setLoading(true)

    try {
      const { data } = await api.post('/sessions/text-message', {
        session_id: sessionId,
        text,
        emotion: 'neutral',
      })
      if (data.reply) {
        addMsg('therapist', data.reply)
        if (!open) {
          setUnread(u => u + 1)
          triggerShake()
        }
      }
    } catch (e) {
      addMsg('system', e.response?.data?.detail || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  const addMsg = (role, text) =>
    setMessages(p => [...p, { role, text, ts: Date.now() }])

  const triggerShake = () => {
    setShake(true)
    setTimeout(() => setShake(false), 600)
  }

  // ── Auto-scroll ────────────────────────────────────────────────────────────
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // ── Key handler ────────────────────────────────────────────────────────────
  const onKey = e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  return (
    <>
      {/* ── Floating button ─────────────────────────────────────────────── */}
      <button
        onClick={open ? handleClose : handleOpen}
        style={{
          ...w.fab,
          transform: shake ? 'scale(1.15) rotate(-5deg)' : open ? 'scale(0.92)' : 'scale(1)',
        }}
        title="Chat with Serene"
        aria-label="Open therapy chat"
      >
        {/* Pulse ring when closed */}
        {!open && <span style={w.fabRing} />}

        {/* Icon */}
        <span style={{ fontSize: 22, lineHeight: 1, transition: 'transform 0.3s', display: 'block',
          transform: open ? 'rotate(45deg)' : 'rotate(0deg)' }}>
          {open ? '✕' : '✦'}
        </span>

        {/* Unread badge */}
        {unread > 0 && !open && (
          <span style={w.badge}>{unread}</span>
        )}
      </button>

      {/* ── Chat popup ──────────────────────────────────────────────────── */}
      <div style={{
        ...w.popup,
        opacity:         open ? 1 : 0,
        pointerEvents:   open ? 'all' : 'none',
        transform:       open ? 'translateY(0) scale(1)' : 'translateY(20px) scale(0.96)',
        transformOrigin: 'bottom right',
      }}>

        {/* Header */}
        <div style={w.header}>
          <div style={w.headerLeft}>
            <div style={w.avatar}>✦</div>
            <div>
              <div style={w.headerName}>Serene</div>
              <div style={w.headerSub}>
                <span style={{ ...w.dot, background: started ? '#4e9e6e' : '#a08cb8' }} />
                {loading && !started ? 'Connecting…' : started ? 'AI Therapist · Text mode' : 'Starting…'}
              </div>
            </div>
          </div>
          <button onClick={handleClose} style={w.closeBtn} aria-label="Close chat">✕</button>
        </div>

        {/* Messages */}
        <div style={w.body}>
          {messages.length === 0 && !loading && (
            <div style={w.emptyHint}>
              <div style={w.emptyGlyph}>✿</div>
              <p style={w.emptyText}>Your safe space to speak freely.</p>
              <p style={w.emptySubtext}>Serene listens without judgment.</p>
            </div>
          )}
          {messages.map((m, i) => <Bubble key={i} m={m} user={user} />)}
          {loading && started && <TypingBubble />}
          {loading && !started && (
            <div style={w.connecting}>
              <span style={w.connectingDot} />
              <span style={w.connectingDot2} />
              <span style={w.connectingDot3} />
            </div>
          )}
          <div ref={chatBottomRef} />
        </div>

        {/* Footer divider & note */}
        <div style={w.modeNote}>
          <span style={w.modeNoteText}>✦ Text session · For voice, use Full Session</span>
        </div>

        {/* Input */}
        <div style={w.inputRow}>
          <textarea
            ref={inputRef}
            style={w.input}
            rows={1}
            placeholder={started ? "Share what's on your mind…" : "Connecting…"}
            value={input}
            disabled={!started || loading}
            onChange={e => setInput(e.target.value)}
            onKeyDown={onKey}
          />
          <button
            onClick={send}
            disabled={!input.trim() || !started || loading}
            style={{
              ...w.sendBtn,
              opacity: (!input.trim() || !started || loading) ? 0.4 : 1,
            }}
            aria-label="Send message"
          >
            ↑
          </button>
        </div>
      </div>
    </>
  )
}

// ── Styles ────────────────────────────────────────────────────────────────────
const w = {
  fab: {
    position:     'fixed',
    bottom:       28,
    right:        28,
    width:        56,
    height:       56,
    borderRadius: '50%',
    border:       'none',
    background:   'linear-gradient(135deg, #2d1f2e 0%, #4a2d3e 100%)',
    color:        '#c9a96e',
    cursor:       'pointer',
    zIndex:       9999,
    display:      'flex',
    alignItems:   'center',
    justifyContent: 'center',
    boxShadow:    '0 8px 28px rgba(45,31,46,0.38), 0 2px 8px rgba(0,0,0,0.18)',
    transition:   'transform 0.25s cubic-bezier(0.34,1.56,0.64,1), box-shadow 0.2s',
    flexShrink:   0,
  },
  fabRing: {
    position:     'absolute',
    inset:        -6,
    borderRadius: '50%',
    border:       '2px solid rgba(201,169,110,0.35)',
    animation:    'recPulse 2.4s ease infinite',
    pointerEvents:'none',
  },
  badge: {
    position:     'absolute',
    top:          -4,
    right:        -4,
    width:        20,
    height:       20,
    borderRadius: '50%',
    background:   '#c9898a',
    color:        'white',
    fontSize:     11,
    fontWeight:   700,
    display:      'flex',
    alignItems:   'center',
    justifyContent: 'center',
    fontFamily:   'var(--font-body)',
    boxShadow:    '0 2px 6px rgba(201,137,138,0.5)',
    border:       '2px solid white',
  },

  popup: {
    position:      'fixed',
    bottom:        96,
    right:         28,
    width:         360,
    height:        520,
    borderRadius:  20,
    background:    'rgba(253,248,242,0.97)',
    backdropFilter:'blur(24px)',
    border:        '1px solid rgba(232,196,196,0.35)',
    boxShadow:     '0 24px 64px rgba(45,31,46,0.22), 0 4px 16px rgba(0,0,0,0.10)',
    zIndex:        9998,
    display:       'flex',
    flexDirection: 'column',
    overflow:      'hidden',
    transition:    'opacity 0.28s ease, transform 0.28s cubic-bezier(0.34,1.56,0.64,1)',
  },

  header: {
    display:        'flex',
    alignItems:     'center',
    justifyContent: 'space-between',
    padding:        '14px 16px',
    background:     'linear-gradient(135deg, #2d1f2e 0%, #4a2d3e 100%)',
    flexShrink:     0,
  },
  headerLeft: { display: 'flex', alignItems: 'center', gap: 10 },
  avatar: {
    width:          36,
    height:         36,
    borderRadius:   '50%',
    background:     'rgba(201,169,110,0.18)',
    border:         '1.5px solid rgba(201,169,110,0.45)',
    display:        'flex',
    alignItems:     'center',
    justifyContent: 'center',
    fontSize:       14,
    color:          '#c9a96e',
    flexShrink:     0,
  },
  headerName: {
    fontFamily: 'var(--font-display)',
    fontSize:   18,
    fontWeight: 400,
    color:      '#f5ede0',
    lineHeight: 1.2,
  },
  headerSub: {
    display:    'flex',
    alignItems: 'center',
    gap:        5,
    fontSize:   10,
    color:      'rgba(212,200,220,0.6)',
    fontFamily: 'var(--font-body)',
    marginTop:  2,
  },
  dot: {
    width:        6,
    height:       6,
    borderRadius: '50%',
    display:      'inline-block',
    flexShrink:   0,
  },
  closeBtn: {
    background: 'none',
    border:     'none',
    color:      'rgba(212,200,220,0.5)',
    fontSize:   16,
    cursor:     'pointer',
    padding:    4,
    lineHeight: 1,
    fontFamily: 'var(--font-body)',
    transition: 'color 0.2s',
  },

  body: {
    flex:          1,
    overflowY:     'auto',
    padding:       '16px 14px',
    display:       'flex',
    flexDirection: 'column',
    gap:           12,
  },

  emptyHint: {
    flex:          1,
    display:       'flex',
    flexDirection: 'column',
    alignItems:    'center',
    justifyContent:'center',
    padding:       '32px 0',
    textAlign:     'center',
    gap:           6,
  },
  emptyGlyph:   { fontSize: 32, color: 'var(--rose)', marginBottom: 8 },
  emptyText:    { fontFamily: 'var(--font-display)', fontSize: 17, color: 'var(--charcoal)', fontStyle: 'italic' },
  emptySubtext: { fontSize: 12, color: 'var(--mist)' },

  connecting: { display: 'flex', gap: 6, justifyContent: 'center', padding: '24px 0' },
  connectingDot:  { width: 8, height: 8, borderRadius: '50%', background: 'var(--rose)', animation: 'typing 1.2s ease infinite', animationDelay: '0ms', display: 'inline-block' },
  connectingDot2: { width: 8, height: 8, borderRadius: '50%', background: 'var(--rose)', animation: 'typing 1.2s ease infinite', animationDelay: '180ms', display: 'inline-block' },
  connectingDot3: { width: 8, height: 8, borderRadius: '50%', background: 'var(--rose)', animation: 'typing 1.2s ease infinite', animationDelay: '360ms', display: 'inline-block' },

  modeNote: {
    padding:       '5px 14px',
    borderTop:     '1px solid rgba(232,196,196,0.25)',
    borderBottom:  '1px solid rgba(232,196,196,0.25)',
    background:    'rgba(245,237,224,0.5)',
    flexShrink:    0,
  },
  modeNoteText: {
    fontSize:      10,
    color:         'var(--mist)',
    fontFamily:    'var(--font-body)',
    letterSpacing: '0.04em',
  },

  inputRow: {
    display:    'flex',
    gap:        8,
    padding:    '10px 12px',
    alignItems: 'flex-end',
    flexShrink: 0,
    background: 'rgba(255,255,255,0.7)',
  },
  input: {
    flex:        1,
    padding:     '10px 14px',
    border:      '1.5px solid rgba(232,196,196,0.6)',
    borderRadius:16,
    fontSize:    13,
    fontFamily:  'var(--font-body)',
    color:       'var(--charcoal)',
    background:  'white',
    outline:     'none',
    resize:      'none',
    lineHeight:  1.5,
    maxHeight:   80,
    overflowY:   'auto',
    transition:  'border-color 0.2s',
  },
  sendBtn: {
    width:          36,
    height:         36,
    borderRadius:   '50%',
    border:         'none',
    background:     'linear-gradient(135deg, #2d1f2e 0%, #4a2d3e 100%)',
    color:          '#c9a96e',
    fontSize:       18,
    cursor:         'pointer',
    display:        'flex',
    alignItems:     'center',
    justifyContent: 'center',
    flexShrink:     0,
    transition:     'opacity 0.2s, transform 0.15s',
    lineHeight:     1,
    fontFamily:     'var(--font-body)',
  },
}

const bs = {
  system: {
    alignSelf:  'center',
    fontSize:   11,
    color:      '#3a2a40',
    padding:    '4px 14px',
    background: 'rgba(255,255,255,0.8)',
    borderRadius: 999,
    display:    'flex',
    gap:        5,
    alignItems: 'center',
    fontFamily: 'var(--font-body)',
    fontWeight: 500,
    border:     '1px solid rgba(180,130,160,0.2)',
  },
  row:      { display: 'flex', alignItems: 'flex-end', gap: 6 },
  avatarT: {
    width:          26, height: 26, borderRadius: '50%',
    background:     'linear-gradient(135deg, #2d1f2e, #4a2d3e)',
    display:        'flex', alignItems: 'center', justifyContent: 'center',
    fontSize:       11, color: '#c9a96e', flexShrink: 0,
  },
  avatarU: {
    width:          26, height: 26, borderRadius: '50%',
    background:     'linear-gradient(135deg, #c08080, #a04848)',
    display:        'flex', alignItems: 'center', justifyContent: 'center',
    fontSize:       11, color: 'white', fontWeight: 700, flexShrink: 0,
  },
  bubble:     { maxWidth: '80%', padding: '10px 14px', borderRadius: 16, lineHeight: 1.6 },
  bubbleT: {
    background:            'rgba(255,255,255,0.97)',
    backdropFilter:        'blur(8px)',
    border:                '1px solid rgba(180,130,160,0.28)',
    borderBottomLeftRadius: 4,
    boxShadow:             '0 2px 10px rgba(100,60,80,0.07)',
    color:                 '#1a1520',
  },
  bubbleUser: {
    background:             'linear-gradient(135deg, #2d1f2e 0%, #4a2d3e 100%)',
    borderBottomRightRadius: 4,
    boxShadow:              '0 2px 10px rgba(45,31,46,0.22)',
    color:                  '#f5ede0',
  },
  label: {
    fontSize:      9,
    fontWeight:    700,
    letterSpacing: '0.07em',
    marginBottom:  4,
    textTransform: 'uppercase',
    opacity:       0.55,
    fontFamily:    'var(--font-body)',
  },
  text: { fontSize: 13, margin: 0, whiteSpace: 'pre-wrap', color: 'inherit' },
  time: { fontSize: 9, opacity: 0.4, marginTop: 5, textAlign: 'right', fontFamily: 'var(--font-body)' },
}