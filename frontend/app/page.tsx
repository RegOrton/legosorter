"use client";
import { useState, useEffect } from 'react';

export default function Home() {
  const [classification, setClassification] = useState<any>(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [inferenceStatus, setInferenceStatus] = useState<any>(null);
  const [boundingBoxes, setBoundingBoxes] = useState<any[]>([]);
  const [centerDetected, setCenterDetected] = useState(false);

  // Poll inference status every second
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch('http://localhost:8000/inference/status');
        const data = await res.json();
        setInferenceStatus(data);
        if (data.current_prediction) {
          setClassification(data.current_prediction);
        }
        // Update bounding boxes and center detection
        setBoundingBoxes(data.bounding_boxes || []);
        setCenterDetected(data.center_detected || false);
      } catch (e) {
        console.error('Failed to fetch inference status', e);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const classifyNow = async () => {
    setIsClassifying(true);
    try {
      const res = await fetch('http://localhost:8000/inference/classify_now', {
        method: 'POST'
      });
      const data = await res.json();
      setClassification(data);
    } catch (e) {
      console.error('Classification failed', e);
    } finally {
      setIsClassifying(false);
    }
  };

  return (
    <div className="flex flex-col text-zinc-50 h-full">
      <main className="flex-1 p-6 grid grid-cols-1 md:grid-cols-12 gap-6 max-w-[1600px] mx-auto w-full">
        {/* Main Video Feed Area */}
        <section className="col-span-1 md:col-span-8 flex flex-col gap-4">
          <div className="bg-zinc-900/50 rounded-xl border border-zinc-800 aspect-video flex items-center justify-center relative overflow-hidden group shadow-2xl">
            {/* Grid Pattern Overlay */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>

            {/* Video Stream */}
            <img
              src="http://localhost:8000/video/stream"
              alt="Live Webcam Feed"
              className="absolute inset-0 w-full h-full object-cover z-0"
            />

            {/* SVG Overlay for Bounding Boxes and Center Indicator */}
            <svg className="absolute inset-0 w-full h-full z-10" viewBox="0 0 640 480" preserveAspectRatio="xMidYMid slice">
              {/* Center-frame crosshair indicator */}
              <g className={`transition-all duration-300 ${centerDetected ? 'opacity-100' : 'opacity-40'}`}>
                {/* Center crosshair */}
                <circle
                  cx="320"
                  cy="240"
                  r="40"
                  fill="none"
                  stroke={centerDetected ? "#10b981" : "#6b7280"}
                  strokeWidth="2"
                  className={centerDetected ? "animate-pulse" : ""}
                />
                <circle
                  cx="320"
                  cy="240"
                  r="30"
                  fill="none"
                  stroke={centerDetected ? "#10b981" : "#6b7280"}
                  strokeWidth="1"
                  opacity="0.5"
                />
                {/* Crosshair lines */}
                <line x1="320" y1="200" x2="320" y2="220" stroke={centerDetected ? "#10b981" : "#6b7280"} strokeWidth="2" />
                <line x1="320" y1="260" x2="320" y2="280" stroke={centerDetected ? "#10b981" : "#6b7280"} strokeWidth="2" />
                <line x1="280" y1="240" x2="300" y2="240" stroke={centerDetected ? "#10b981" : "#6b7280"} strokeWidth="2" />
                <line x1="340" y1="240" x2="360" y2="240" stroke={centerDetected ? "#10b981" : "#6b7280"} strokeWidth="2" />

                {/* Center dot */}
                <circle
                  cx="320"
                  cy="240"
                  r="3"
                  fill={centerDetected ? "#10b981" : "#6b7280"}
                  className={centerDetected ? "animate-pulse" : ""}
                />
              </g>

              {/* Bounding boxes */}
              {boundingBoxes.map((bbox: any, idx: number) => (
                <g key={idx} className="animate-in fade-in duration-200">
                  {/* Bounding box rectangle */}
                  <rect
                    x={bbox.x}
                    y={bbox.y}
                    width={bbox.width}
                    height={bbox.height}
                    fill="none"
                    stroke={bbox.is_centered ? "#10b981" : "#3b82f6"}
                    strokeWidth="2"
                    className="transition-all duration-300"
                  />

                  {/* Corner accents */}
                  <line x1={bbox.x} y1={bbox.y} x2={bbox.x + 15} y2={bbox.y} stroke={bbox.is_centered ? "#10b981" : "#3b82f6"} strokeWidth="3" />
                  <line x1={bbox.x} y1={bbox.y} x2={bbox.x} y2={bbox.y + 15} stroke={bbox.is_centered ? "#10b981" : "#3b82f6"} strokeWidth="3" />

                  <line x1={bbox.x + bbox.width} y1={bbox.y} x2={bbox.x + bbox.width - 15} y2={bbox.y} stroke={bbox.is_centered ? "#10b981" : "#3b82f6"} strokeWidth="3" />
                  <line x1={bbox.x + bbox.width} y1={bbox.y} x2={bbox.x + bbox.width} y2={bbox.y + 15} stroke={bbox.is_centered ? "#10b981" : "#3b82f6"} strokeWidth="3" />

                  <line x1={bbox.x} y1={bbox.y + bbox.height} x2={bbox.x + 15} y2={bbox.y + bbox.height} stroke={bbox.is_centered ? "#10b981" : "#3b82f6"} strokeWidth="3" />
                  <line x1={bbox.x} y1={bbox.y + bbox.height} x2={bbox.x} y2={bbox.y + bbox.height - 15} stroke={bbox.is_centered ? "#10b981" : "#3b82f6"} strokeWidth="3" />

                  <line x1={bbox.x + bbox.width} y1={bbox.y + bbox.height} x2={bbox.x + bbox.width - 15} y2={bbox.y + bbox.height} stroke={bbox.is_centered ? "#10b981" : "#3b82f6"} strokeWidth="3" />
                  <line x1={bbox.x + bbox.width} y1={bbox.y + bbox.height} x2={bbox.x + bbox.width} y2={bbox.y + bbox.height - 15} stroke={bbox.is_centered ? "#10b981" : "#3b82f6"} strokeWidth="3" />

                  {/* Center point of bbox */}
                  <circle
                    cx={bbox.center_x}
                    cy={bbox.center_y}
                    r="3"
                    fill={bbox.is_centered ? "#10b981" : "#3b82f6"}
                  />

                  {/* Label background */}
                  <rect
                    x={bbox.x}
                    y={bbox.y - 20}
                    width={80}
                    height={18}
                    fill={bbox.is_centered ? "#10b981" : "#3b82f6"}
                    opacity="0.9"
                  />

                  {/* Label text */}
                  <text
                    x={bbox.x + 4}
                    y={bbox.y - 7}
                    fill="white"
                    fontSize="12"
                    fontFamily="monospace"
                    fontWeight="bold"
                  >
                    {bbox.is_centered ? "LOCKED" : "DETECTED"}
                  </text>
                </g>
              ))}
            </svg>

            {/* Overlay UI */}
            <div className="absolute top-4 left-4 flex gap-2 z-20">
              <span className="px-2 py-1 bg-black/50 backdrop-blur text-[10px] text-zinc-400 rounded border border-white/10 font-mono">640x480 @ {inferenceStatus?.fps?.toFixed(1) || '30'} fps</span>
              <span className="px-2 py-1 bg-emerald-500/20 backdrop-blur text-[10px] text-emerald-400 rounded border border-emerald-500/30 font-mono flex items-center gap-1">
                <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse"></span>
                LIVE
              </span>
              {inferenceStatus?.is_running && (
                <span className="px-2 py-1 bg-blue-500/20 backdrop-blur text-[10px] text-blue-400 rounded border border-blue-500/30 font-mono flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse"></span>
                  CLASSIFYING
                </span>
              )}
              {centerDetected && (
                <span className="px-2 py-1 bg-emerald-500/20 backdrop-blur text-[10px] text-emerald-400 rounded border border-emerald-500/30 font-mono flex items-center gap-1 animate-pulse">
                  <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full"></span>
                  TARGET LOCKED
                </span>
              )}
            </div>

            {/* Classification Overlay */}
            {classification && (
              <div className="absolute top-4 right-4 z-20 bg-black/70 backdrop-blur-md rounded-lg border border-zinc-700 p-3 min-w-[200px]">
                <div className="text-xs text-zinc-400 mb-1">Detected:</div>
                <div className="text-lg font-bold text-white mb-1">{classification.class_name}</div>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-300"
                      style={{ width: `${(classification.confidence * 100).toFixed(0)}%` }}
                    ></div>
                  </div>
                  <span className="text-xs text-zinc-300 font-mono">{(classification.confidence * 100).toFixed(1)}%</span>
                </div>
              </div>
            )}

            {/* Detection Status Overlay */}
            <div className="absolute bottom-20 right-4 z-20 bg-black/80 backdrop-blur-md rounded-lg border border-zinc-700 p-3 min-w-[280px] max-h-[400px] overflow-y-auto custom-scrollbar">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xs font-bold text-zinc-300 uppercase tracking-wider flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                    <circle cx="8.5" cy="8.5" r="1.5" />
                    <polyline points="21 15 16 10 5 21" />
                  </svg>
                  Detection Status
                </h3>
                <span className={`px-2 py-0.5 rounded text-[10px] font-mono ${boundingBoxes.length > 0 ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-zinc-700/50 text-zinc-500 border border-zinc-600'}`}>
                  {boundingBoxes.length > 0 ? 'ACTIVE' : 'IDLE'}
                </span>
              </div>

              {/* Algorithm Status */}
              <div className="mb-3 pb-3 border-b border-zinc-700">
                <div className="text-[10px] text-zinc-500 uppercase font-semibold mb-2">Algorithm</div>
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-zinc-400">Method:</span>
                    <span className="text-zinc-200 font-mono">MOG2 BG Sub</span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-zinc-400">Objects Found:</span>
                    <span className={`font-mono font-bold ${boundingBoxes.length > 0 ? 'text-emerald-400' : 'text-zinc-500'}`}>
                      {boundingBoxes.length}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-zinc-400">Center Lock:</span>
                    <span className={`font-mono font-bold ${centerDetected ? 'text-emerald-400' : 'text-zinc-500'}`}>
                      {centerDetected ? '✓ LOCKED' : '✗ NO'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Detected Objects List */}
              {boundingBoxes.length > 0 ? (
                <div>
                  <div className="text-[10px] text-zinc-500 uppercase font-semibold mb-2">Detected Objects</div>
                  <div className="space-y-2 max-h-[200px] overflow-y-auto pr-1">
                    {boundingBoxes.map((bbox: any, idx: number) => (
                      <div
                        key={idx}
                        className={`p-2 rounded border ${bbox.is_centered ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-zinc-800/50 border-zinc-700'} transition-all`}
                      >
                        <div className="flex items-center justify-between mb-1.5">
                          <span className="text-xs font-bold text-white">Object #{idx + 1}</span>
                          {bbox.is_centered && (
                            <span className="px-1.5 py-0.5 bg-emerald-500 text-white text-[9px] font-bold rounded">
                              CENTERED
                            </span>
                          )}
                        </div>
                        <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px]">
                          <div className="flex justify-between">
                            <span className="text-zinc-500">X:</span>
                            <span className="text-zinc-300 font-mono">{bbox.x}px</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-zinc-500">Y:</span>
                            <span className="text-zinc-300 font-mono">{bbox.y}px</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-zinc-500">W:</span>
                            <span className="text-zinc-300 font-mono">{bbox.width}px</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-zinc-500">H:</span>
                            <span className="text-zinc-300 font-mono">{bbox.height}px</span>
                          </div>
                          <div className="flex justify-between col-span-2">
                            <span className="text-zinc-500">Area:</span>
                            <span className="text-zinc-300 font-mono">{bbox.area.toLocaleString()}px²</span>
                          </div>
                          <div className="flex justify-between col-span-2">
                            <span className="text-zinc-500">Center:</span>
                            <span className="text-zinc-300 font-mono">({bbox.center_x}, {bbox.center_y})</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-center py-6">
                  <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="mx-auto mb-2 text-zinc-600">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                    <circle cx="8.5" cy="8.5" r="1.5" />
                    <polyline points="21 15 16 10 5 21" />
                  </svg>
                  <p className="text-xs text-zinc-500">No objects detected</p>
                  <p className="text-[10px] text-zinc-600 mt-1">Place object in view</p>
                </div>
              )}
            </div>

            {/* Bottom overlay with source info */}
            <div className="absolute bottom-4 left-4 z-20">
              <div className="px-2 py-1 bg-black/50 backdrop-blur text-[10px] text-zinc-400 rounded border border-white/10 font-mono">
                HTTP Webcam • host.docker.internal:5000
              </div>
            </div>

            {/* Bounding box count indicator */}
            {boundingBoxes.length > 0 && (
              <div className="absolute bottom-4 right-4 z-20">
                <div className="px-2 py-1 bg-black/50 backdrop-blur text-[10px] text-zinc-400 rounded border border-white/10 font-mono">
                  Objects: {boundingBoxes.length}
                </div>
              </div>
            )}
          </div>

          {/* Secondary Stats / Graphs Row */}
          <div className="grid grid-cols-3 gap-4 h-32">
            <div className="bg-zinc-900/50 rounded-xl border border-zinc-800 p-4 flex flex-col justify-between">
              <span className="text-xs text-zinc-500 uppercase tracking-wider font-semibold">Frames</span>
              <span className="text-2xl font-mono text-zinc-200">{inferenceStatus?.frame_count?.toLocaleString() || '0'}</span>
            </div>
            <div className="bg-zinc-900/50 rounded-xl border border-zinc-800 p-4 flex flex-col justify-between">
              <span className="text-xs text-zinc-500 uppercase tracking-wider font-semibold">Inference FPS</span>
              <span className="text-2xl font-mono text-zinc-200">{inferenceStatus?.fps?.toFixed(1) || '0'} <span className="text-sm text-zinc-500">fps</span></span>
            </div>
            <div className="bg-zinc-900/50 rounded-xl border border-zinc-800 p-4 flex flex-col justify-between">
              <span className="text-xs text-zinc-500 uppercase tracking-wider font-semibold">Confidence</span>
              <span className={`text-2xl font-mono ${classification?.confidence > 0.5 ? 'text-emerald-500' : 'text-yellow-500'}`}>
                {classification ? `${(classification.confidence * 100).toFixed(1)}%` : 'N/A'}
              </span>
            </div>
          </div>
        </section>

        {/* Sidebar Controls */}
        <section className="col-span-1 md:col-span-4 flex flex-col gap-4">
          <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-5 flex flex-col gap-6 shadow-lg">
            <div>
              <h2 className="text-sm font-medium text-zinc-400 mb-4 flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><path d="M12 6v6l4 2" /></svg>
                CLASSIFICATION CONTROL
              </h2>
              <div className="grid grid-cols-1 gap-3">
                <button
                  onClick={classifyNow}
                  disabled={isClassifying}
                  className="group relative overflow-hidden bg-blue-600 hover:bg-blue-500 active:bg-blue-700 disabled:bg-zinc-700 disabled:cursor-not-allowed text-zinc-100 py-4 rounded-lg transition-all border border-blue-500 disabled:border-zinc-600 flex items-center justify-center gap-2 font-medium shadow-lg"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700" />
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></svg>
                  {isClassifying ? 'Classifying...' : 'Classify Now'}
                </button>
              </div>
            </div>

            {/* Machine Control Section */}
            <div className="pt-4 border-t border-zinc-800">
              <h2 className="text-sm font-medium text-zinc-400 mb-4 flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="7" width="20" height="14" rx="2" ry="2" /><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" /></svg>
                MACHINE CONTROL
              </h2>
              <div className="grid grid-cols-2 gap-2">
                {/* Start/Stop Inference */}
                {!inferenceStatus?.is_running ? (
                  <button
                    onClick={async () => {
                      try {
                        await fetch('http://localhost:8000/inference/start', { method: 'POST' });
                      } catch (e) {
                        console.error('Failed to start inference', e);
                      }
                    }}
                    className="col-span-2 bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700 text-white py-3 rounded-lg transition-all border border-emerald-500 flex items-center justify-center gap-2 font-medium"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 3 19 12 5 21 5 3" /></svg>
                    Start Inference
                  </button>
                ) : (
                  <button
                    onClick={async () => {
                      try {
                        await fetch('http://localhost:8000/inference/stop', { method: 'POST' });
                      } catch (e) {
                        console.error('Failed to stop inference', e);
                      }
                    }}
                    className="col-span-2 bg-red-600 hover:bg-red-500 active:bg-red-700 text-white py-3 rounded-lg transition-all border border-red-500 flex items-center justify-center gap-2 font-medium"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="6" y="4" width="4" height="16" /><rect x="14" y="4" width="4" height="16" /></svg>
                    Stop Inference
                  </button>
                )}

                {/* Conveyor Controls */}
                <button
                  className="bg-zinc-700 hover:bg-zinc-600 active:bg-zinc-800 text-zinc-100 py-2.5 rounded-lg transition-all border border-zinc-600 flex items-center justify-center gap-1.5 text-sm font-medium"
                  title="Start Conveyor"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="9 18 15 12 9 6" /></svg>
                  Conveyor
                </button>
                <button
                  className="bg-zinc-700 hover:bg-zinc-600 active:bg-zinc-800 text-zinc-100 py-2.5 rounded-lg transition-all border border-zinc-600 flex items-center justify-center gap-1.5 text-sm font-medium"
                  title="Stop Conveyor"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="6" y="6" width="12" height="12" /></svg>
                  Stop
                </button>

                {/* Diverter Controls */}
                <button
                  className="bg-zinc-700 hover:bg-zinc-600 active:bg-zinc-800 text-zinc-100 py-2.5 rounded-lg transition-all border border-zinc-600 flex items-center justify-center gap-1.5 text-sm font-medium"
                  title="Open Diverter"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" /></svg>
                  Open
                </button>
                <button
                  className="bg-zinc-700 hover:bg-zinc-600 active:bg-zinc-800 text-zinc-100 py-2.5 rounded-lg transition-all border border-zinc-600 flex items-center justify-center gap-1.5 text-sm font-medium"
                  title="Close Diverter"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" /><line x1="12" y1="8" x2="12" y2="16" /></svg>
                  Close
                </button>

                {/* Emergency Stop */}
                <button
                  className="col-span-2 bg-red-700 hover:bg-red-600 active:bg-red-800 text-white py-3 rounded-lg transition-all border-2 border-red-500 flex items-center justify-center gap-2 font-bold shadow-lg"
                  title="Emergency Stop - Stops all motors immediately"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" /></svg>
                  EMERGENCY STOP
                </button>
              </div>
            </div>

            {/* Classification Results */}
            {classification && (
              <div className="pt-2 border-t border-zinc-800">
                <h2 className="text-sm font-medium text-zinc-400 mb-3">DETECTION RESULT</h2>
                <div className="bg-zinc-950 rounded-lg border border-zinc-800 p-4">
                  <div className="text-2xl font-bold text-white mb-2">{classification.class_name}</div>
                  <div className="text-sm text-zinc-400 mb-3">
                    Confidence: <span className={`font-mono ${classification.confidence > 0.5 ? 'text-emerald-400' : 'text-yellow-400'}`}>
                      {(classification.confidence * 100).toFixed(2)}%
                    </span>
                  </div>

                  {classification.all_classes && (
                    <div className="space-y-1.5">
                      <div className="text-xs text-zinc-500 uppercase font-semibold mb-2">All Probabilities:</div>
                      {classification.all_classes.slice(0, 5).map((cls: any, idx: number) => (
                        <div key={idx} className="flex items-center gap-2">
                          <div className="text-xs text-zinc-400 w-24 truncate">{cls.name}</div>
                          <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-blue-500 to-blue-400"
                              style={{ width: `${(cls.probability * 100).toFixed(0)}%` }}
                            ></div>
                          </div>
                          <span className="text-xs text-zinc-500 font-mono w-12 text-right">{(cls.probability * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Logs Panel */}
          <div className="flex-1 bg-zinc-950 rounded-xl border border-zinc-800 p-4 overflow-hidden flex flex-col min-h-[200px]">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-zinc-500">SYSTEM LOGS</span>
              <span className="text-[10px] bg-zinc-900 border border-zinc-800 px-1.5 py-0.5 rounded text-zinc-500">LIVE</span>
            </div>
            <div className="flex-1 font-mono text-[11px] text-zinc-400 overflow-y-auto space-y-1 custom-scrollbar">
              <div className="opacity-50">------------------ INIT ------------------</div>
              <div><span className="text-zinc-600">[{new Date().toLocaleTimeString()}]</span> System modules loaded</div>
              <div><span className="text-zinc-600">[{new Date().toLocaleTimeString()}]</span> <span className="text-emerald-500">Webcam connected</span></div>
              {inferenceStatus?.is_running && (
                <div><span className="text-zinc-600">[{new Date().toLocaleTimeString()}]</span> <span className="text-blue-500">Inference engine running</span></div>
              )}
              {classification && (
                <div><span className="text-zinc-600">[{new Date().toLocaleTimeString()}]</span> Detected: {classification.class_name} ({(classification.confidence * 100).toFixed(1)}%)</div>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
