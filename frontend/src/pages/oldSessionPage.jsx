import { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../AuthContext.jsx'
import api from '../api.js'

// ── Emotion config ───────────────────────────────────────────────────────────
// Colors chosen to be clearly visible on BOTH the dark waveform panel
// and the light cream background panels.
const EMOTION_COLORS = {
  neutral: '#a08cb8',   // soft purple — visible on dark + light
  happy:   '#d4943a',   // amber — visible on dark + light
  excited: '#c75050',   // rose-red — visible on dark + light
  calm:    '#4e9e6e',   // sage green — visible on dark + light
  sad:     '#5b8fcf',   // steel blue — visible on dark + light
  anger:   '#b84040',   // deep red — visible on dark + light
}
const EMOTION_KEYS  = ['neutral', 'happy', 'excited', 'calm', 'sad', 'anger']
const EMOTION_EMOJI = { neutral:'◎', happy:'◉', excited:'◈', calm:'◍', sad:'◌', anger:'◆' }

// ── Fusion weights (must match backend exactly) ───────────────────────────────
const W_ACOUSTIC = 0.40
const W_SEMANTIC = 0.60

// ── Client-side fusion ────────────────────────────────────────────────────────
// Both acoustic and semantic probs now come from the backend (real librosa +
// Hartmann). The frontend just applies the weighted fusion formula.
function fuseEmotions(acousticProbs, semanticProbs, hasSpeech) {
  const fused = {}
  for (const k of EMOTION_KEYS) {
    fused[k] = hasSpeech
      ? W_ACOUSTIC * (acousticProbs[k] || 0) + W_SEMANTIC * (semanticProbs[k] || 0)
      : (acousticProbs[k] || 0)
  }
  const total = Object.values(fused).reduce((a, b) => a + b, 0)
  if (total > 0) for (const k of EMOTION_KEYS) fused[k] /= total
  return Object.entries(fused).sort((a, b) => b[1] - a[1])[0][0]
}

function _flatProbs() {
  const v = 1.0 / EMOTION_KEYS.length
  const p = {}
  for (const k of EMOTION_KEYS) p[k] = v
  return p
}

// ── WAV encoder ──────────────────────────────────────────────────────────────
function encodeWAV(samples, sr) {
  const buf = new ArrayBuffer(44 + samples.length * 2)
  const v   = new DataView(buf)
  const w   = (s, o) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)) }
  w('RIFF',0); v.setUint32(4, 36 + samples.length*2, true)
  w('WAVE',8); w('fmt ',12)
  v.setUint32(16,16,true); v.setUint16(20,1,true); v.setUint16(22,1,true)
  v.setUint32(24,sr,true); v.setUint32(28,sr*2,true)
  v.setUint16(32,2,true);  v.setUint16(34,16,true)
  w('data',36); v.setUint32(40, samples.length*2, true)
  let off = 44
  for (let i = 0; i < samples.length; i++, off += 2) {
    const x = Math.max(-1, Math.min(1, samples[i]))
    v.setInt16(off, x < 0 ? x * 0x8000 : x * 0x7FFF, true)
  }
  return new Blob([buf], { type:'audio/wav' })
}
function blobToB64(blob) {
  return new Promise((res, rej) => {
    const r = new FileReader()
    r.onload  = () => res(r.result.split(',')[1])
    r.onerror = rej
    r.readAsDataURL(blob)
  })
}

// ── Waveform Canvas ──────────────────────────────────────────────────────────
function WaveCanvas({ data, color, running, height = 150 }) {
  const ref = useRef()
  useEffect(() => {
    const c = ref.current; if (!c) return
    const ctx = c.getContext('2d')
    const W = c.width, H = c.height
    ctx.clearRect(0, 0, W, H)

    const grad = ctx.createLinearGradient(0, 0, 0, H)
    grad.addColorStop(0, '#1a1225')
    grad.addColorStop(1, '#0e0b12')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, W, H)

    ctx.strokeStyle = 'rgba(200,160,200,0.12)'
    ctx.lineWidth = 0.5
    for (let i = 1; i < 4; i++) {
      ctx.beginPath(); ctx.moveTo(0, H*i/4); ctx.lineTo(W, H*i/4); ctx.stroke()
    }
    ctx.strokeStyle = 'rgba(200,160,200,0.20)'
    ctx.lineWidth = 0.8
    ctx.beginPath(); ctx.moveTo(0, H/2); ctx.lineTo(W, H/2); ctx.stroke()

    if (!running || data.length === 0) {
      ctx.strokeStyle = 'rgba(200,160,200,0.35)'
      ctx.lineWidth = 1.5
      ctx.setLineDash([4, 8])
      ctx.beginPath(); ctx.moveTo(0, H/2); ctx.lineTo(W, H/2); ctx.stroke()
      ctx.setLineDash([])
      return
    }

    ctx.shadowBlur   = 12
    ctx.shadowColor  = color
    ctx.strokeStyle  = color
    ctx.lineWidth    = 2.5
    ctx.globalAlpha  = 0.95
    ctx.beginPath()
    for (let x = 0; x < W; x++) {
      const idx = Math.floor((x / W) * data.length)
      const y   = H/2 - (data[idx] || 0) * (H/2) * 0.82
      x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    }
    ctx.stroke()

    ctx.shadowBlur  = 4
    ctx.lineWidth   = 1.0
    ctx.globalAlpha = 0.35
    ctx.beginPath()
    for (let x = 0; x < W; x++) {
      const idx = Math.floor((x / W) * data.length)
      const y   = H/2 - (data[idx] || 0) * (H/2) * 0.82
      x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    }
    ctx.stroke()
    ctx.globalAlpha = 1
    ctx.shadowBlur  = 0
  }, [data, color, running])

  return (
    <canvas
      ref={ref}
      width={900}
      height={height}
      style={{ width: '100%', height, display: 'block', borderRadius: 12 }}
    />
  )
}

// ── Pipeline A readout strip ──────────────────────────────────────────────────
// Shows live librosa features returned by the backend alongside the waveform.
function PipelineReadout({ acousticProbs, semanticProbs, lastText, emotion, running }) {
  const topA = acousticProbs
    ? Object.entries(acousticProbs).sort((a, b) => b[1] - a[1])[0]
    : null
  const topS = semanticProbs
    ? Object.entries(semanticProbs).sort((a, b) => b[1] - a[1])[0]
    : null

  return (
    <div style={pr.wrap}>
      {/* Pipeline A */}
      <div style={pr.row}>
        <span style={pr.pipeLabel}>Pipeline A · Librosa</span>
        {topA ? (
          <span style={{ ...pr.result, color: EMOTION_COLORS[topA[0]] || '#a08cb8' }}>
            {topA[0].toUpperCase()} · {(topA[1] * 100).toFixed(0)}%
          </span>
        ) : (
          <span style={pr.dim}>{running ? 'Computing…' : 'Idle'}</span>
        )}
        <span style={pr.dim}>40%</span>
      </div>
      {/* Pipeline B */}
      <div style={pr.row}>
        <span style={pr.pipeLabel}>Pipeline B · Hartmann NLP</span>
        {topS && lastText ? (
          <span style={{ ...pr.result, color: EMOTION_COLORS[topS[0]] || '#a08cb8' }}>
            {topS[0].toUpperCase()} · {(topS[1] * 100).toFixed(0)}%
          </span>
        ) : (
          <span style={pr.dim}>{running ? 'Waiting for speech…' : 'Idle'}</span>
        )}
        <span style={pr.dim}>60%</span>
      </div>
      {/* Fusion */}
      <div style={{ ...pr.row, borderTop: '1px solid rgba(200,160,200,0.15)', paddingTop: 6, marginTop: 2 }}>
        <span style={{ ...pr.pipeLabel, color: '#d4c8dc' }}>⚡ Fused Prediction</span>
        <span style={{ ...pr.result, color: EMOTION_COLORS[emotion] || '#a08cb8', fontWeight: 800, fontSize: 13 }}>
          {emotion.toUpperCase()}
        </span>
        {lastText && <span style={{ ...pr.dim, fontSize: 10, maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          "{lastText.slice(0, 28)}{lastText.length > 28 ? '…' : ''}"
        </span>}
      </div>
    </div>
  )
}

const pr = {
  wrap: {
    background: 'linear-gradient(135deg, #140e1a 0%, #0e0b12 100%)',
    border: '1px solid rgba(200,160,200,0.18)',
    borderRadius: 12, padding: '10px 14px',
    display: 'flex', flexDirection: 'column', gap: 5,
  },
  row: { display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' },
  pipeLabel: {
    fontSize: 10, fontWeight: 700, letterSpacing: '0.06em',
    textTransform: 'uppercase', color: '#8a7a94',
    fontFamily: 'var(--font-body)', minWidth: 160,
  },
  result: {
    fontSize: 12, fontWeight: 700,
    fontFamily: 'var(--font-body)', flex: 1,
  },
  dim: { fontSize: 10, color: '#5a4a64', fontFamily: 'var(--font-body)' },
}

// ── Live Words Display ────────────────────────────────────────────────────────
function LiveWordsDisplay({ confirmedText, interimText, isRecording, isProcessing }) {
  const confirmedWords = confirmedText ? confirmedText.split(/\s+/).filter(Boolean) : []
  const interimWords   = interimText   ? interimText.split(/\s+/).filter(Boolean)   : []

  return (
    <div style={lw.card}>
      <div style={lw.header}>
        <span style={lw.title}>🎙 Live Spoken Words</span>
        <span style={{
          ...lw.badge,
          background: isRecording ? 'rgba(160,72,72,0.15)' : 'rgba(74,120,64,0.12)',
          color:      isRecording ? '#c05050' : '#4a8050',
          border:     `1px solid ${isRecording ? 'rgba(160,72,72,0.35)' : 'rgba(74,120,64,0.25)'}`,
        }}>
          {isProcessing ? '⚙ Sending…' : isRecording ? '● Recording' : '◌ Idle'}
        </span>
      </div>
      <div style={lw.body}>
        {confirmedWords.length > 0 || interimWords.length > 0 ? (
          <div style={lw.wordWrap}>
            {confirmedWords.map((word, i) => (
              <span key={`c-${i}`} style={{ ...lw.word, color: '#1a1520' }} className="fade-in">
                {word}
              </span>
            ))}
            {interimWords.map((word, i) => (
              <span key={`interim-${i}`} style={{ ...lw.word, color: '#5a4a60', opacity: 0.65 }}>
                {word}
              </span>
            ))}
          </div>
        ) : (
          <p style={lw.placeholder}>
            {isRecording
              ? 'Speak now — words appear here in real time…'
              : 'Press ▶ Start to begin recording, then speak freely.'}
          </p>
        )}
      </div>
      {(confirmedWords.length > 0 || interimWords.length > 0) && (
        <div style={lw.footer}>
          <span style={lw.footerNote}>
            {isRecording ? '● Live — press Stop & Send when done' : '✓ Ready to send'}
          </span>
        </div>
      )}
    </div>
  )
}

const lw = {
  card: {
    background: 'rgba(255,255,255,0.85)',
    border: '1px solid rgba(180,130,160,0.3)',
    borderRadius: 16, overflow: 'hidden',
    boxShadow: '0 4px 20px rgba(100,60,80,0.08)',
  },
  header: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    padding: '10px 16px',
    background: 'rgba(240,230,220,0.7)',
    borderBottom: '1px solid rgba(180,130,160,0.2)',
  },
  title: { fontSize: 11, fontWeight: 700, color: '#3a2a40', letterSpacing: '0.07em', textTransform: 'uppercase' },
  badge: { fontSize: 10, fontWeight: 700, padding: '3px 10px', borderRadius: 999, letterSpacing: '0.04em' },
  body:  { padding: '16px 18px', minHeight: 80, display: 'flex', flexWrap: 'wrap', alignContent: 'flex-start' },
  wordWrap: { display: 'flex', flexWrap: 'wrap', gap: '6px 8px', width: '100%' },
  word: {
    fontFamily: 'var(--font-display)', fontSize: 18, fontStyle: 'italic',
    letterSpacing: '0.01em', animation: 'fadeIn 0.2s ease forwards',
  },
  placeholder: { fontSize: 13, color: '#5a4a60', fontStyle: 'italic', lineHeight: 1.6, margin: 0 },
  footer: { padding: '7px 18px', borderTop: '1px solid rgba(180,130,160,0.15)', background: 'rgba(168,184,154,0.08)' },
  footerNote: { fontSize: 10, color: '#3a6030', fontWeight: 600, letterSpacing: '0.03em' },
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN SESSION PAGE
// ═══════════════════════════════════════════════════════════════════════════════
export default function SessionPage() {
  const nav      = useNavigate()
  const { user } = useAuth()

  // Audio refs
  const audioCtxRef   = useRef()
  const processorRef  = useRef()
  const sourceRef     = useRef()
  const streamRef     = useRef()
  const waveRawRef    = useRef([])
  const sessionAudRef = useRef([])
  const DISPLAY_MAX   = 16000 * 5

  // Web Speech API
  const recognitionRef    = useRef(null)
  const recognitionActive = useRef(false)

  // Pipeline B refs
  const semanticIntervalRef = useRef(null)
  const semanticProbsRef    = useRef(_flatProbs())
  const lastSemanticTextRef = useRef('')

  // Pipeline A ref — now populated from backend (real librosa), not JS DFT
  const acousticProbsRef = useRef(_flatProbs())

  // Acoustic polling: accumulate ~100 ms of PCM, send to backend, get probs
  // A ref to the pending acoustic request so we don't fire more than one at a time
  const acousticInFlightRef = useRef(false)
  // Buffer of samples waiting to be sent for acoustic classification
  const acousticBufRef = useRef([])
  // How many samples = ~200 ms at 16 kHz (enough for librosa to be meaningful)
  const ACOUSTIC_CHUNK_SAMPLES = 3200   // 200 ms

  // State
  const [micRunning,      setMicRunning]      = useState(false)
  const [isProcessing,    setIsProcessing]    = useState(false)
  const [sessionId,       setSessionId]       = useState(null)
  const [sessionStarted,  setSessionStarted]  = useState(false)
  const [sessionEnded,    setSessionEnded]    = useState(false)
  const [hasPastMemory,   setHasPastMemory]   = useState(false)
  const [ending,          setEnding]          = useState(false)

  const [recState, setRecState] = useState('idle')

  const [waveData,      setWaveData]      = useState([])
  const [emotion,       setEmotion]       = useState('neutral')
  const [acousticProbs, setAcousticProbs] = useState(_flatProbs())
  const [semanticProbs, setSemanticProbs] = useState(_flatProbs())
  const [lastText,      setLastText]      = useState('')
  const [recSecs,       setRecSecs]       = useState(0)
  const [totalSecs,     setTotalSecs]     = useState(0)

  const [confirmedText, setConfirmedText] = useState('')
  const [interimText,   setInterimText]   = useState('')
  const [messages,      setMessages]      = useState([])
  const [status,        setStatus]        = useState('Connecting…')

  const chatBottomRef = useRef()
  const recTimerRef   = useRef()

  // ── Session start on mount ───────────────────────────────────────────────
  useEffect(() => {
    api.post('/sessions/start')
      .then(r => {
        setSessionId(r.data.session_id)
        setHasPastMemory(r.data.has_past_memory)
        addMsg('therapist', r.data.greeting)
        setSessionStarted(true)
        setStatus('Ready — press ▶ Start to begin')
      })
      .catch(() => {
        addMsg('system', 'Could not connect to backend.')
        setStatus('Backend offline')
      })
  }, [])

  // ── Recording timer ──────────────────────────────────────────────────────
  useEffect(() => {
    if (micRunning) {
      recTimerRef.current = setInterval(() => {
        setRecSecs(s => s + 1)
        setTotalSecs(s => s + 1)
      }, 1000)
    } else {
      clearInterval(recTimerRef.current)
      setRecSecs(0)
    }
    return () => clearInterval(recTimerRef.current)
  }, [micRunning])

  // ── Auto-scroll ──────────────────────────────────────────────────────────
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const addMsg = (role, text) =>
    setMessages(p => [...p, { role, text, ts: Date.now() }])

  // ── Backend Pipeline A call ──────────────────────────────────────────────
  // Sends a WAV chunk to /api/sessions/acoustic-emotion (real librosa on server).
  // Called automatically whenever enough samples have accumulated.
  const runAcousticPipelineA = useCallback(async (samples) => {
    if (acousticInFlightRef.current) return
    acousticInFlightRef.current = true
    try {
      const blob = encodeWAV(new Float32Array(samples), 16000)
      const b64  = await blobToB64(blob)
      const res  = await api.post('/sessions/acoustic-emotion', { audio_b64: b64 })
      if (res.data?.probs) {
        acousticProbsRef.current = res.data.probs
        setAcousticProbs(res.data.probs)
        // Recompute fused emotion immediately
        const hasSpeech = lastSemanticTextRef.current.trim().length > 0
        const fused = fuseEmotions(res.data.probs, semanticProbsRef.current, hasSpeech)
        setEmotion(fused)
      }
    } catch (_) {
      // Silent fail — keep last known probs
    } finally {
      acousticInFlightRef.current = false
    }
  }, [])

  // ── Build Web Speech recogniser ──────────────────────────────────────────
  const buildRecogniser = useCallback(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SR) return null

    const rec = new SR()
    rec.continuous     = true
    rec.interimResults = true
    rec.lang           = 'en-US'

    rec.onresult = e => {
      let confirmed = ''
      let interim   = ''
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript
        if (e.results[i].isFinal) confirmed += t + ' '
        else                       interim   += t
      }
      if (confirmed) {
        setConfirmedText(p => {
          const next = (p + confirmed).trimEnd()
          window._sereneConfirmed = next
          return next
        })
      }
      setInterimText(interim)
      window._sereneInterim = interim
    }

    rec.onerror = e => {
      if (e.error !== 'no-speech') console.warn('SpeechRecognition:', e.error)
    }

    rec.onend = () => {
      if (recognitionActive.current) {
        try { rec.start() } catch(_) {}
      }
    }

    return rec
  }, [])

  // ── Stop mic hardware only ────────────────────────────────────────────────
  const stopMicHardware = useCallback(() => {
    recognitionActive.current = false
    if (recognitionRef.current) {
      try { recognitionRef.current.stop() } catch(_) {}
      recognitionRef.current = null
    }
    if (semanticIntervalRef.current) {
      clearInterval(semanticIntervalRef.current)
      semanticIntervalRef.current = null
    }
    semanticProbsRef.current    = _flatProbs()
    lastSemanticTextRef.current = ''
    acousticBufRef.current      = []
    window._sereneConfirmed     = ''
    window._sereneInterim       = ''

    processorRef.current?.disconnect(); processorRef.current = null
    sourceRef.current?.disconnect();    sourceRef.current    = null
    audioCtxRef.current?.close();       audioCtxRef.current  = null
    streamRef.current?.getTracks().forEach(t => t.stop()); streamRef.current = null

    setMicRunning(false)
    setWaveData([])
    setInterimText('')
  }, [])

  // ── START listening ──────────────────────────────────────────────────────
  const startListening = useCallback(async () => {
    if (micRunning || sessionEnded || !sessionStarted || isProcessing) return

    if (recState === 'idle') {
      setConfirmedText('')
      setInterimText('')
      sessionAudRef.current  = []
    }
    acousticProbsRef.current    = _flatProbs()
    semanticProbsRef.current    = _flatProbs()
    lastSemanticTextRef.current = ''
    acousticBufRef.current      = []
    window._sereneConfirmed     = confirmedText
    window._sereneInterim       = ''

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 })
      audioCtxRef.current = ctx
      const source = ctx.createMediaStreamSource(stream)
      sourceRef.current   = source

      const proc = ctx.createScriptProcessor(2048, 1, 1)
      processorRef.current = proc

      proc.onaudioprocess = e => {
        const chunk = new Float32Array(e.inputBuffer.getChannelData(0))

        // ── Waveform display buffer ──────────────────────────────────────────
        waveRawRef.current.push(...chunk)
        if (waveRawRef.current.length > DISPLAY_MAX)
          waveRawRef.current = waveRawRef.current.slice(-DISPLAY_MAX)
        setWaveData([...waveRawRef.current.slice(-3200)])

        // ── Session audio (for voice-message) ────────────────────────────────
        sessionAudRef.current.push(...chunk)

        // ── Pipeline A accumulator → backend librosa ──────────────────────
        // Accumulate samples until we have ~200 ms worth, then fire off
        // a backend call to get real librosa acoustic probs.
        acousticBufRef.current.push(...chunk)
        if (acousticBufRef.current.length >= ACOUSTIC_CHUNK_SAMPLES) {
          const snap = acousticBufRef.current.splice(0, ACOUSTIC_CHUNK_SAMPLES)
          runAcousticPipelineA(snap)
        }
      }
      source.connect(proc)
      proc.connect(ctx.destination)

      // ── Pipeline B — semantic polling every 2 s via Hartmann NLP ───────
      semanticIntervalRef.current = setInterval(async () => {
        const fullText = ((window._sereneConfirmed || '') + ' ' + (window._sereneInterim || '')).trim()
        if (!fullText) return
        const rollingText = fullText.split(/\s+/).slice(-40).join(' ')
        if (rollingText === lastSemanticTextRef.current) return
        lastSemanticTextRef.current = rollingText
        setLastText(rollingText)
        try {
          const res = await api.post('/sessions/detect-emotion', { text: rollingText })
          if (res.data?.probs) {
            semanticProbsRef.current = res.data.probs
            setSemanticProbs(res.data.probs)
            // Recompute fused emotion with latest semantic result
            const fused = fuseEmotions(acousticProbsRef.current, res.data.probs, true)
            setEmotion(fused)
          }
        } catch (_) {}
      }, 2000)

      const rec = buildRecogniser()
      if (rec) {
        recognitionRef.current    = rec
        recognitionActive.current = true
        rec.start()
      }

      setMicRunning(true)
      setRecState('recording')
      setStatus('🎤 Recording — press Stop & Send when done')
    } catch(e) {
      addMsg('system', `Microphone error: ${e.message}`)
    }
  }, [micRunning, sessionEnded, sessionStarted, isProcessing, recState, confirmedText, buildRecogniser, runAcousticPipelineA])

  // ── STOP & SEND ──────────────────────────────────────────────────────────
  const stopAndSend = useCallback(async () => {
    if (!micRunning || isProcessing || !sessionId) return
    setIsProcessing(true)
    setRecState('paused')

    const snap = [...sessionAudRef.current]
    sessionAudRef.current = []

    const liveText = confirmedText.trim()
    stopMicHardware()

    if (snap.length < 8000) {
      addMsg('system', 'No audio captured — please speak first.')
      setIsProcessing(false)
      setRecState('idle')
      setStatus('Ready — press ▶ Start to continue')
      return
    }

    setStatus('⚙️ Transcribing & generating response…')

    try {
      const blob = encodeWAV(snap, 16000)
      const b64  = await blobToB64(blob)
      const { data } = await api.post('/sessions/voice-message', {
        session_id: sessionId,
        audio_b64:  b64,
        emotion,
      })

      const displayText = liveText || data.transcript || ''
      if (displayText) {
        setConfirmedText(displayText)
        addMsg('user', displayText)
      }
      // Backend returns the fused emotion computed from the full clip
      // (real librosa + Hartmann on the complete utterance)
      if (data.emotion) setEmotion(data.emotion)
      if (data.reply)   addMsg('therapist', data.reply)

      setStatus('Done — press ▶ Start to continue speaking')
    } catch(e) {
      const detail = e.response?.data?.detail || e.message
      addMsg('system', `Error: ${detail}`)
      setStatus('Ready — press ▶ Start to continue')
    }

    setIsProcessing(false)
    setRecState('idle')
    setConfirmedText('')
    setInterimText('')
  }, [micRunning, isProcessing, sessionId, emotion, confirmedText, stopMicHardware])

  // ── TERMINATE SESSION ─────────────────────────────────────────────────────
  const terminateSession = useCallback(async () => {
    if (!sessionId || ending) return
    stopMicHardware()
    setEnding(true)
    setRecState('idle')
    setStatus('Saving session…')

    try {
      const chatPayload = messages
        .filter(m => m.role !== 'system')
        .map(m => ({
          role:    m.role === 'therapist' ? 'assistant' : 'user',
          content: m.text,
        }))

      await api.post('/sessions/end', {
        session_id:   sessionId,
        chat_history: chatPayload,
      })

      setSessionEnded(true)
      addMsg('system', 'Session ended. Your summary has been saved to your account. 🌸')
      setStatus('Session complete — summary saved')
    } catch(e) {
      addMsg('system', `Save error: ${e.message}`)
      setStatus('Error saving session')
    }
    setEnding(false)
  }, [sessionId, ending, messages, stopMicHardware])

  const col = EMOTION_COLORS[emotion] || '#a08cb8'
  const fmt = ts => new Date(ts).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })

  const primaryBtnLabel = () => {
    if (isProcessing) return '⏳ Processing…'
    if (micRunning)   return '⏺ Stop & Send'
    return '▶ Start'
  }
  const primaryBtnAction = () => {
    if (isProcessing) return
    if (micRunning)   return stopAndSend()
    return startListening()
  }
  const primaryBtnColor = micRunning ? '#a04848' : '#3a6a50'

  const fmtSecs = s => `${Math.floor(s / 60).toString().padStart(2,'0')}:${(s % 60).toString().padStart(2,'0')}`

  return (
    <div style={s.page}>
      {/* Top bar */}
      <header style={s.topBar}>
        <div style={s.topLeft}>
          <button onClick={() => nav('/')} style={s.backBtn}>← Back</button>
          <div>
            <div style={s.topTitle}>
              <span style={s.topGlyph}>✦</span> Serene — Live Session
            </div>
            <div style={s.topSub}>
              {!sessionStarted ? 'Connecting…'
                : sessionEnded   ? 'Session complete'
                : hasPastMemory  ? '◈ Continuing your journey'
                : '◈ New session'}
            </div>
          </div>
        </div>

        <div style={s.controls}>
          {!sessionEnded && sessionStarted && (
            <button
              onClick={primaryBtnAction}
              disabled={isProcessing || sessionEnded || !sessionStarted}
              style={{
                ...s.ctrlBtn,
                background: primaryBtnColor,
                color: '#ffffff',
                opacity: isProcessing ? 0.55 : 1,
                boxShadow: micRunning ? `0 0 0 3px ${primaryBtnColor}44` : 'none',
              }}
            >
              {micRunning && !isProcessing && <span style={s.recDot} />}
              {primaryBtnLabel()}
            </button>
          )}

          {!sessionEnded && sessionStarted && (
            <button
              onClick={terminateSession}
              disabled={ending}
              style={{
                ...s.ctrlBtn,
                background: 'transparent',
                border: '1.5px solid #5a3a3a',
                color: '#4a2a2a',
                opacity: ending ? 0.45 : 1,
              }}
            >
              {ending ? '⏳ Saving…' : '■ Terminate'}
            </button>
          )}

          {sessionEnded && (
            <button onClick={() => nav('/')} style={{ ...s.ctrlBtn, background: '#a04848', color: '#fff' }}>
              Dashboard →
            </button>
          )}
        </div>
      </header>

      <div style={s.panels}>
        {/* LEFT PANEL — waveform + emotion */}
        <div style={s.leftPanel}>
          <div style={s.leftInner}>

            {/* Emotion badge */}
            <div style={s.panelHead}>
              <span style={s.panelTitle}>Real-Time Voice</span>
              <span style={{
                ...s.emotionTag,
                color: col,
                background: `${col}22`,
                border: `2px solid ${col}66`,
                boxShadow: `0 0 12px ${col}33`,
              }}>
                <span style={{ fontSize: 20 }}>{EMOTION_EMOJI[emotion]}</span>
                <span style={{ fontWeight: 800, fontSize: 13, letterSpacing: '0.10em' }}>
                  {emotion.toUpperCase()}
                </span>
                {micRunning && <span style={{ ...s.recDotInline, background: col }} />}
              </span>
            </div>

            {/* Waveform */}
            <div style={s.canvasWrap}>
              <WaveCanvas data={waveData} color={col} running={micRunning} height={155} />
              {micRunning && (
                <div style={s.recOverlay}>
                  <div style={s.recPulse} />
                  <span style={{ color: '#f0d5d5', fontWeight: 600 }}>{fmtSecs(recSecs)}</span>
                </div>
              )}
              {!micRunning && totalSecs > 0 && (
                <div style={{ ...s.recOverlay, opacity: 0.6 }}>
                  <span style={{ color: '#d4c8dc' }}>Total {fmtSecs(totalSecs)}</span>
                </div>
              )}
            </div>

            {/* Pipeline A + B readout — replaces bare legend */}
            <PipelineReadout
              acousticProbs={acousticProbs}
              semanticProbs={semanticProbs}
              lastText={lastText}
              emotion={emotion}
              running={micRunning}
            />

            {/* Emotion legend */}
            <div style={s.legend}>
              {EMOTION_KEYS.map(k => (
                <span key={k} style={{
                  ...s.legendItem,
                  color:      emotion === k ? EMOTION_COLORS[k] : '#5a4a60',
                  fontWeight: emotion === k ? 700 : 400,
                  opacity:    emotion === k ? 1 : 0.6,
                }}>
                  {EMOTION_EMOJI[k]} {k}
                </span>
              ))}
            </div>

            {/* Live words */}
            <LiveWordsDisplay
              confirmedText={confirmedText}
              interimText={interimText}
              isRecording={micRunning && !isProcessing}
              isProcessing={isProcessing}
            />

            {/* Status hint */}
            <div style={s.statusHint}>
              {isProcessing
                ? <><span style={s.spinnerDot} /> Transcribing &amp; generating response…</>
                : micRunning
                  ? '🎤 Speak freely — press Stop & Send when done'
                  : sessionEnded
                    ? '✓ Session saved to your account'
                    : sessionStarted
                      ? 'Press ▶ Start to speak'
                      : 'Connecting…'}
            </div>
          </div>
        </div>

        <div style={s.divider} />

        {/* RIGHT PANEL — chat */}
        <div style={s.rightPanel}>
          <div style={s.rightHead}>
            <div style={s.therapistMark}>
              <div style={s.therapistAvatar}>✦</div>
              <div>
                <div style={s.therapistName}>Serene</div>
                <div style={s.therapistSub}>
                  AI Therapist · {hasPastMemory ? '◈ Memory active' : 'First session'}
                </div>
              </div>
            </div>
            <div style={{
              ...s.statusPill,
              background: isProcessing ? 'rgba(160,72,72,0.1)' : 'rgba(74,120,64,0.1)',
              color:  isProcessing ? '#8a3030' : '#2a5020',
              border: `1px solid ${isProcessing ? 'rgba(160,72,72,0.25)' : 'rgba(74,120,64,0.25)'}`,
            }}>
              {status}
            </div>
          </div>

          {micRunning && !isProcessing && (
            <div style={s.recBar}>
              <div style={s.recBarDot} />
              <span style={{ color: '#8a3030', fontWeight: 600 }}>
                Recording… {fmtSecs(recSecs)}
              </span>
            </div>
          )}

          <div style={s.chatArea}>
            {messages.map((m, i) => <Bubble key={i} m={m} fmt={fmt} user={user} />)}
            {isProcessing && <TypingBubble />}
            <div ref={chatBottomRef} />
          </div>

          <div style={s.chatHint}>
            ▶ Start to record&nbsp;·&nbsp;⏺ Stop &amp; Send to send&nbsp;·&nbsp;■ Terminate to save &amp; exit
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Bubble components ─────────────────────────────────────────────────────────
function Bubble({ m, fmt, user }) {
  const isUser   = m.role === 'user'
  const isSystem = m.role === 'system'
  if (isSystem) return (
    <div style={bs.system} className="fade-in">
      <span style={{ opacity: 0.5 }}>◦</span> {m.text}
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
      {isUser && <div style={bs.avatarU}>{(user?.name || 'Y')[0].toUpperCase()}</div>}
    </div>
  )
}

function TypingBubble() {
  return (
    <div style={{ ...bs.row, justifyContent: 'flex-start' }} className="fade-in">
      <div style={bs.avatarT}>✦</div>
      <div style={{ ...bs.bubble, ...bs.bubbleT, padding: '14px 18px' }}>
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

// ── Styles ────────────────────────────────────────────────────────────────────
const s = {
  page: {
    height: '100vh', display: 'flex', flexDirection: 'column',
    background: 'var(--cream)', overflow: 'hidden',
  },
  topBar: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '12px 28px', flexWrap: 'wrap', gap: 10,
    background: 'rgba(253,248,242,0.97)',
    backdropFilter: 'blur(20px)',
    borderBottom: '1px solid rgba(180,130,160,0.25)',
    flexShrink: 0, zIndex: 10,
  },
  topLeft:  { display: 'flex', alignItems: 'center', gap: 16 },
  backBtn: {
    background: 'none', border: '1px solid rgba(180,130,160,0.45)',
    borderRadius: 999, padding: '7px 16px', fontSize: 13,
    color: '#3a2a40', cursor: 'pointer', fontFamily: 'var(--font-body)',
    fontWeight: 500,
  },
  topGlyph: { color: 'var(--gold)', marginRight: 6 },
  topTitle: {
    fontFamily: 'var(--font-display)', fontSize: 20, fontWeight: 500,
    color: '#1a1520',
  },
  topSub: { fontSize: 11, color: '#4a3a50', marginTop: 2, fontWeight: 500 },
  controls: { display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' },

  ctrlBtn: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '10px 22px', border: 'none', borderRadius: 999,
    fontSize: 13, fontWeight: 700, cursor: 'pointer',
    fontFamily: 'var(--font-body)', letterSpacing: '0.02em',
    whiteSpace: 'nowrap', transition: 'all 0.2s',
  },
  recDot: {
    width: 8, height: 8, borderRadius: '50%',
    background: '#ffffff', display: 'inline-block',
    animation: 'recPulse 1.2s ease infinite', flexShrink: 0,
  },

  panels:    { flex: 1, display: 'flex', overflow: 'hidden' },
  leftPanel: {
    flex: 1, overflowY: 'auto',
    background: 'linear-gradient(160deg, #fdf8f2 0%, #f5ede0 60%, #ede8f5 100%)',
  },
  leftInner: { padding: '22px 22px 32px', display: 'flex', flexDirection: 'column', gap: 16 },

  panelHead: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  panelTitle: {
    fontFamily: 'var(--font-display)', fontSize: 22, fontWeight: 500,
    color: '#1a1520',
  },
  emotionTag: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '7px 18px', borderRadius: 999,
    transition: 'all 0.35s',
  },
  recDotInline: {
    width: 7, height: 7, borderRadius: '50%',
    display: 'inline-block', animation: 'recPulse 1.2s ease infinite',
  },

  canvasWrap: {
    borderRadius: 16, overflow: 'hidden',
    border: '1px solid rgba(180,130,160,0.3)',
    boxShadow: '0 4px 24px rgba(100,60,80,0.12)',
    position: 'relative',
  },
  recOverlay: {
    position: 'absolute', bottom: 10, right: 14,
    display: 'flex', alignItems: 'center', gap: 6,
    fontSize: 11, background: 'rgba(14,11,18,0.75)',
    padding: '3px 10px', borderRadius: 999, backdropFilter: 'blur(4px)',
  },
  recPulse: {
    width: 7, height: 7, borderRadius: '50%',
    background: '#e07070', animation: 'recPulse 1.2s ease infinite',
  },

  legend: { display: 'flex', flexWrap: 'wrap', gap: '6px 18px' },
  legendItem: { fontSize: 12, transition: 'all 0.3s', fontFamily: 'var(--font-body)', fontWeight: 500 },

  statusHint: {
    display: 'flex', alignItems: 'center', gap: 8,
    fontSize: 12, color: '#3a2a40', fontStyle: 'italic', padding: '4px 0',
    fontWeight: 500,
  },
  spinnerDot: {
    width: 8, height: 8, borderRadius: '50%',
    background: '#8a3030', display: 'inline-block',
    animation: 'pulse 1s ease infinite',
  },

  divider: { width: 1, background: 'rgba(180,130,160,0.25)', flexShrink: 0 },

  rightPanel: {
    width: 460, flexShrink: 0, display: 'flex', flexDirection: 'column',
    background: 'rgba(253,248,242,0.90)', backdropFilter: 'blur(8px)', overflow: 'hidden',
  },
  rightHead: {
    padding: '16px 20px', borderBottom: '1px solid rgba(180,130,160,0.2)',
    display: 'flex', flexDirection: 'column', gap: 10, flexShrink: 0,
  },
  therapistMark:   { display: 'flex', alignItems: 'center', gap: 12 },
  therapistAvatar: {
    width: 40, height: 40, borderRadius: '50%',
    background: 'linear-gradient(135deg, #2d1f2e, #4a2d3e)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 16, color: 'var(--gold)', flexShrink: 0,
  },
  therapistName: {
    fontFamily: 'var(--font-display)', fontSize: 20, fontWeight: 500,
    color: '#1a1520',
  },
  therapistSub: { fontSize: 11, color: '#4a3a50', marginTop: 1, fontWeight: 500 },
  statusPill: {
    padding: '6px 14px', borderRadius: 999, fontSize: 11,
    fontFamily: 'var(--font-body)', letterSpacing: '0.02em',
    alignSelf: 'flex-start', transition: 'all 0.3s', fontWeight: 600,
  },
  recBar: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '7px 20px', fontSize: 12,
    background: 'rgba(160,72,72,0.07)',
    borderBottom: '1px solid rgba(160,72,72,0.12)',
    fontFamily: 'var(--font-body)', flexShrink: 0,
  },
  recBarDot: {
    width: 6, height: 6, borderRadius: '50%',
    background: '#8a3030', animation: 'recPulse 1.2s ease infinite', flexShrink: 0,
  },
  chatArea: {
    flex: 1, overflowY: 'auto', padding: '20px 16px',
    display: 'flex', flexDirection: 'column', gap: 14,
  },
  chatHint: {
    padding: '10px 20px', fontSize: 11, color: '#4a3a50',
    textAlign: 'center', borderTop: '1px solid rgba(180,130,160,0.2)',
    fontFamily: 'var(--font-body)', flexShrink: 0, fontWeight: 500,
  },
}

const bs = {
  system: {
    alignSelf: 'center', fontSize: 11, color: '#3a2a40',
    padding: '5px 16px', background: 'rgba(255,255,255,0.8)',
    borderRadius: 999, display: 'flex', gap: 6, alignItems: 'center',
    fontFamily: 'var(--font-body)', fontWeight: 500,
    border: '1px solid rgba(180,130,160,0.2)',
  },
  row:    { display: 'flex', alignItems: 'flex-end', gap: 8 },
  avatarT: {
    width: 32, height: 32, borderRadius: '50%',
    background: 'linear-gradient(135deg, #2d1f2e, #4a2d3e)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 12, color: 'var(--gold)', flexShrink: 0,
  },
  avatarU: {
    width: 32, height: 32, borderRadius: '50%',
    background: 'linear-gradient(135deg, #c08080, #a04848)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 13, color: 'white', fontWeight: 700, flexShrink: 0,
  },
  bubble:     { maxWidth: '76%', padding: '13px 17px', borderRadius: 18, lineHeight: 1.65 },
  bubbleT:    {
    background: 'rgba(255,255,255,0.97)',
    backdropFilter: 'blur(8px)',
    border: '1px solid rgba(180,130,160,0.3)',
    borderBottomLeftRadius: 5,
    boxShadow: '0 3px 12px rgba(100,60,80,0.08)',
    color: '#1a1520',
  },
  bubbleUser: {
    background: 'linear-gradient(135deg, #2d1f2e 0%, #4a2d3e 100%)',
    borderBottomRightRadius: 5,
    boxShadow: '0 3px 12px rgba(45,31,46,0.22)',
    color: '#f5ede0',
  },
  label: {
    fontSize: 10, fontWeight: 700, letterSpacing: '0.07em',
    marginBottom: 5, textTransform: 'uppercase', opacity: 0.6,
    fontFamily: 'var(--font-body)',
  },
  text:  { fontSize: 14, margin: 0, whiteSpace: 'pre-wrap', color: 'inherit' },
  time:  { fontSize: 9, opacity: 0.45, marginTop: 6, textAlign: 'right', fontFamily: 'var(--font-body)' },
}