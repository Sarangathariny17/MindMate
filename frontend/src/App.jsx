import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './AuthContext.jsx'
import AuthPage from './pages/AuthPage.jsx'
import Dashboard from './pages/Dashboard.jsx'
import SessionPage from './pages/SessionPage.jsx'
import ChatWidget from './ChatWidget.jsx'   // ← ADD THIS

function Guard({ children }) {
  const { user } = useAuth()
  return user ? children : <Navigate to="/auth" replace />
}

function AppShell() {               // ← ADD THIS WRAPPER
  return (
    <>
      <Routes>
        <Route path="/auth"    element={<AuthPage />} />
        <Route path="/"        element={<Guard><Dashboard /></Guard>} />
        <Route path="/session" element={<Guard><SessionPage /></Guard>} />
      </Routes>
      <ChatWidget />                {/* ← ADD THIS */}
    </>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <AppShell />              {/* ← use AppShell instead of Routes directly */}
      </BrowserRouter>
    </AuthProvider>
  )
}