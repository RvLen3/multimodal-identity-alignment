import { Routes, Route, Navigate } from "react-router-dom"

import LoadingPage from "./LoadingPage"
import IdentityAlignmentSystem from "./system"

function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/home" />} />
      <Route path="/home" element={<LoadingPage />} />
      <Route path="/system" element={<IdentityAlignmentSystem />} />
    </Routes>
  )
}

export default App