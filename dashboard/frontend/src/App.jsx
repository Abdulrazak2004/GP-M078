import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { UnitProvider } from './contexts/UnitContext';
import Layout from './components/Layout';
import MapView from './pages/MapView';
import WellMonitor from './pages/WellMonitor';
import WellDesigner from './pages/WellDesigner';

export default function App() {
  return (
    <UnitProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<MapView />} />
            <Route path="/monitor" element={<WellMonitor />} />
            <Route path="/designer" element={<WellDesigner />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </UnitProvider>
  );
}
