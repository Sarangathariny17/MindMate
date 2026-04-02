import { createContext, useContext, useState } from 'react'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    const name = localStorage.getItem('serene_name')
    return name ? { name } : null
  })

  const login = (data) => {
    localStorage.setItem('serene_token', data.token)
    localStorage.setItem('serene_name', data.name)
    setUser({ name: data.name })
  }

  const logout = () => {
    localStorage.removeItem('serene_token')
    localStorage.removeItem('serene_name')
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}
