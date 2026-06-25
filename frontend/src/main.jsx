import React from 'react';
import ReactDOM from 'react-dom/client';

function App() {
  return (
    <main style={{ fontFamily: 'Inter, system-ui, sans-serif', padding: 32, color: '#e5eef7', background: '#11151d', minHeight: '100vh' }}>
      <h1 style={{ fontSize: 44, marginBottom: 12 }}>Neural Snake Dashboard</h1>
      <p style={{ maxWidth: 760, lineHeight: 1.6, color: '#a9b4c7' }}>
        This React frontend can later show live training metrics, episode charts, saved models, and control the FastAPI backend.
      </p>
      <section style={{ marginTop: 28, display: 'grid', gap: 16, gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
        <Card title="Live Score" value="0" />
        <Card title="Best Score" value="0" />
        <Card title="Epsilon" value="1.000" />
        <Card title="Training Steps" value="0" />
      </section>
    </main>
  );
}

function Card({ title, value }) {
  return (
    <div style={{ background: '#1b2230', border: '1px solid #273042', borderRadius: 20, padding: 20 }}>
      <div style={{ color: '#8fa1bb', fontSize: 14, marginBottom: 8 }}>{title}</div>
      <div style={{ fontSize: 32, fontWeight: 700 }}>{value}</div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
