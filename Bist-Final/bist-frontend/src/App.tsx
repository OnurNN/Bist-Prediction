import { BrowserRouter, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import DetailedAnalysisPage from './pages/DetailedAnalysisPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/detailed-analysis" element={<DetailedAnalysisPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;