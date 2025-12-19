"use client";
import { useState, useEffect, useRef } from 'react';

export default function Home() {
  const [classification, setClassification] = useState<any>(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [inferenceStatus, setInferenceStatus] = useState<any>(null);
  const [inferenceMode, setInferenceMode] = useState<'auto' | 'manual'>('auto');
  const [boundingBoxes, setBoundingBoxes] = useState<any[]>([]);
  const [centerDetected, setCenterDetected] = useState(false);
  const [videoKey, setVideoKey] = useState(0); // Key to force video reload
  const videoRef = useRef<HTMLImageElement>(null);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [calibrationStatus, setCalibrationStatus] = useState<any>(null);
  const [frameResolution, setFrameResolution] = useState({ width: 640, height: 480 });
  const [classificationHistory, setClassificationHistory] = useState<any[]>([]);

  // Poll inference status every second
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch('http://localhost:8000/inference/status');
        if (!res.ok) {
          // API offline - set inference status to offline
          setInferenceStatus(null);
          return;
        }
        const data = await res.json();
        setInferenceStatus(data);
        if (data.mode) {
          setInferenceMode(data.mode);
        }
        if (data.current_prediction) {
          setClassification(data.current_prediction);
        }
        // Update bounding boxes and center detection
        setBoundingBoxes(data.bounding_boxes || []);
        setCenterDetected(data.center_detected || false);
      } catch (e) {
        // Silently fail - vision API is offline, which is expected
        setInferenceStatus(null);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Poll calibration status
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch('http://localhost:8000/inference/calibration_status');
        if (res.ok) {
          const data = await res.json();
          setCalibrationStatus(data);
        }
      } catch (e) {
        // Silently fail
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Fetch frame resolution on mount and periodically
  useEffect(() => {
    const fetchResolution = async () => {
      try {
        const res = await fetch('http://localhost:8000/camera/resolution');
        if (res.ok) {
          const data = await res.json();
          setFrameResolution(data);
        }
      } catch (e) {
        // Use defaults if fetch fails
      }
    };

    fetchResolution();
    const interval = setInterval(fetchResolution, 5000);

    return () => clearInterval(interval);
  }, []);

  // Poll classification history
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await fetch('http://localhost:8000/inference/history');
        if (res.ok) {
          const data = await res.json();
          setClassificationHistory(data.history || []);
        }
      } catch (e) {
        // Silently fail
      }
    };

    fetchHistory();
    const interval = setInterval(fetchHistory, 2000); // Poll every 2 seconds

    return () => clearInterval(interval);
  }, []);

  const classifyNow = async () => {
    setIsClassifying(true);
    try {
      const res = await fetch('http://localhost:8000/inference/classify_now', {
        method: 'POST'
      });
      if (!res.ok) {
        throw new Error('Vision API offline');
      }
      const data = await res.json();
      setClassification(data);
    } catch (e) {
      // Vision API is offline - this is expected if not running
    } finally {
      setIsClassifying(false);
    }
  };

  const setMode = async (mode: 'auto' | 'manual') => {
    try {
      await fetch(`http://localhost:8000/inference/mode?mode=${mode}`, { method: 'POST' });
      setInferenceMode(mode);
    } catch (e) {
      console.error('Failed to set mode', e);
    }
  };

  const calibrateBackground = async () => {
    setIsCalibrating(true);
    try {
      const res = await fetch('http://localhost:8000/inference/calibrate', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setCalibrationStatus(data);
      } else {
        console.error('Calibration failed');
      }
    } catch (e) {
      console.error('Failed to calibrate', e);
    } finally {
      setIsCalibrating(false);
    }
  };

  const resetCalibration = async () => {
    try {
      const res = await fetch('http://localhost:8000/inference/recalibrate', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setCalibrationStatus(data);
      }
    } catch (e) {
      console.error('Failed to reset calibration', e);
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
              key={videoKey}
              ref={videoRef}
              src={`http://localhost:8000/video/stream?t=${videoKey}`}
              alt="Live Webcam Feed"
              className="absolute inset-0 w-full h-full object-cover z-0"
              onError={() => {
                // Reload the video stream after 1 second if it fails
                setTimeout(() => setVideoKey(prev => prev + 1), 1000);
              }}
            />

            {/* SVG Overlay for Bounding Boxes and Center Indicator */}
            <svg className="absolute inset-0 w-full h-full z-10" viewBox={`0 0 ${frameResolution.width} ${frameResolution.height}`} preserveAspectRatio="xMidYMid slice">
              {/* Center-frame crosshair indicator */}
              <g className={`transition-all duration-300 ${centerDetected ? 'opacity-100' : 'opacity-40'}`}>
                {/* Center crosshair */}
                <circle
                  cx={frameResolution.width / 2}
                  cy={frameResolution.height / 2}
                  r="40"
                  fill="none"
                  stroke={centerDetected ? "#10b981" : "#6b7280"}
                  strokeWidth="2"
                  className={centerDetected ? "animate-pulse" : ""}
                />
                <circle
                  cx={frameResolution.width / 2}
                  cy={frameResolution.height / 2}
                  r="30"
                  fill="none"
                  stroke={centerDetected ? "#10b981" : "#6b7280"}
                  strokeWidth="1"
                  opacity="0.5"
                />
                {/* Crosshair lines */}
                <line x1={frameResolution.width / 2} y1={frameResolution.height / 2 - 40} x2={frameResolution.width / 2} y2={frameResolution.height / 2 - 20} stroke={centerDetected ? "#10b981" : "#6b7280"} strokeWidth="2" />
                <line x1={frameResolution.width / 2} y1={frameResolution.height / 2 + 20} x2={frameResolution.width / 2} y2={frameResolution.height / 2 + 40} stroke={centerDetected ? "#10b981" : "#6b7280"} strokeWidth="2" />
                <line x1={frameResolution.width / 2 - 40} y1={frameResolution.height / 2} x2={frameResolution.width / 2 - 20} y2={frameResolution.height / 2} stroke={centerDetected ? "#10b981" : "#6b7280"} strokeWidth="2" />
                <line x1={frameResolution.width / 2 + 20} y1={frameResolution.height / 2} x2={frameResolution.width / 2 + 40} y2={frameResolution.height / 2} stroke={centerDetected ? "#10b981" : "#6b7280"} strokeWidth="2" />

                {/* Center dot */}
                <circle
                  cx={frameResolution.width / 2}
                  cy={frameResolution.height / 2}
                  r="3"
                  fill={centerDetected ? "#10b981" : "#6b7280"}
                  className={centerDetected ? "animate-pulse" : ""}
                />
              </g>

              {/* Bounding boxes */}
              {boundingBoxes.map((bbox: any, idx: number) => {
                // Determine color based on validation status
                let strokeColor = '#6b7280'; // gray - invalid
                let labelText = 'INVALID';
                let labelBg = '#6b7280';

                if (bbox.is_stable && bbox.is_centered) {
                  strokeColor = '#10b981'; // emerald - ready
                  labelText = 'READY';
                  labelBg = '#10b981';
                } else if (bbox.is_centered && !bbox.touches_edge && bbox.aspect_ratio_valid) {
                  strokeColor = '#3b82f6'; // blue - stabilizing
                  labelText = 'STABILIZING';
                  labelBg = '#3b82f6';
                } else if (bbox.touches_edge) {
                  strokeColor = '#ef4444'; // red - edge touch
                  labelText = 'EDGE TOUCH';
                  labelBg = '#ef4444';
                }

                return (
                  <g key={idx} className="animate-in fade-in duration-200">
                    {/* Bounding box rectangle */}
                    <rect
                      x={bbox.x}
                      y={bbox.y}
                      width={bbox.width}
                      height={bbox.height}
                      fill="none"
                      stroke={strokeColor}
                      strokeWidth="2"
                      className="transition-all duration-300"
                    />

                    {/* Corner accents */}
                    <line x1={bbox.x} y1={bbox.y} x2={bbox.x + 15} y2={bbox.y} stroke={strokeColor} strokeWidth="3" />
                    <line x1={bbox.x} y1={bbox.y} x2={bbox.x} y2={bbox.y + 15} stroke={strokeColor} strokeWidth="3" />

                    <line x1={bbox.x + bbox.width} y1={bbox.y} x2={bbox.x + bbox.width - 15} y2={bbox.y} stroke={strokeColor} strokeWidth="3" />
                    <line x1={bbox.x + bbox.width} y1={bbox.y} x2={bbox.x + bbox.width} y2={bbox.y + 15} stroke={strokeColor} strokeWidth="3" />

                    <line x1={bbox.x} y1={bbox.y + bbox.height} x2={bbox.x + 15} y2={bbox.y + bbox.height} stroke={strokeColor} strokeWidth="3" />
                    <line x1={bbox.x} y1={bbox.y + bbox.height} x2={bbox.x} y2={bbox.y + bbox.height - 15} stroke={strokeColor} strokeWidth="3" />

                    <line x1={bbox.x + bbox.width} y1={bbox.y + bbox.height} x2={bbox.x + bbox.width - 15} y2={bbox.y + bbox.height} stroke={strokeColor} strokeWidth="3" />
                    <line x1={bbox.x + bbox.width} y1={bbox.y + bbox.height} x2={bbox.x + bbox.width} y2={bbox.y + bbox.height - 15} stroke={strokeColor} strokeWidth="3" />

                    {/* Center point of bbox */}
                    <circle
                      cx={bbox.center_x}
                      cy={bbox.center_y}
                      r="3"
                      fill={strokeColor}
                    />

                    {/* Label background */}
                    <rect
                      x={bbox.x}
                      y={bbox.y - 22}
                      width={Math.max(80, labelText.length * 7)}
                      height={18}
                      fill={labelBg}
                      opacity="0.9"
                    />

                    {/* Label text */}
                    <text
                      x={bbox.x + 4}
                      y={bbox.y - 9}
                      fill="white"
                      fontSize="12"
                      fontFamily="monospace"
                      fontWeight="bold"
                    >
                      {labelText}
                    </text>

                    {/* Stability indicator if available */}
                    {bbox.stability_counter !== undefined && (
                      <text
                        x={bbox.x + 4}
                        y={bbox.y + bbox.height + 15}
                        fill={strokeColor}
                        fontSize="10"
                        fontFamily="monospace"
                        fontWeight="bold"
                      >
                        {bbox.stability_counter}/8
                      </text>
                    )}
                  </g>
                );
              })}
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
              <span className={`px-2 py-1 backdrop-blur text-[10px] rounded border font-mono flex items-center gap-1 ${inferenceMode === 'auto' ? 'bg-purple-500/20 text-purple-400 border-purple-500/30' : 'bg-orange-500/20 text-orange-400 border-orange-500/30'}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${inferenceMode === 'auto' ? 'bg-purple-400' : 'bg-orange-400'}`}></span>
                MODE: {inferenceMode.toUpperCase()}
              </span>
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
                <div className="text-[10px] text-zinc-500 uppercase font-semibold mb-2">Detection Algorithm</div>
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-zinc-400">Method:</span>
                    <span className="text-zinc-200 font-mono">Frame Differencing</span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-zinc-400">Calibration:</span>
                    <span className={`font-mono font-bold ${calibrationStatus?.is_calibrated ? 'text-emerald-400' : 'text-yellow-500'}`}>
                      {calibrationStatus?.is_calibrated ? '✓ OK' : '⚠ PENDING'}
                    </span>
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
                    {boundingBoxes.map((bbox: any, idx: number) => {
                      // Determine color based on validation status
                      let statusColor = 'bg-zinc-800/50 border-zinc-700';
                      let statusLabel = 'INVALID';
                      if (bbox.is_stable && bbox.is_centered) {
                        statusColor = 'bg-emerald-500/20 border-emerald-500/50';
                        statusLabel = 'READY';
                      } else if (bbox.is_centered && !bbox.touches_edge && bbox.aspect_ratio_valid) {
                        statusColor = 'bg-blue-500/20 border-blue-500/30';
                        statusLabel = 'STABILIZING';
                      } else if (bbox.touches_edge) {
                        statusColor = 'bg-red-500/20 border-red-500/30';
                        statusLabel = 'EDGE TOUCH';
                      }

                      return (
                        <div
                          key={idx}
                          className={`p-2 rounded border ${statusColor} transition-all`}
                        >
                          <div className="flex items-center justify-between mb-1.5">
                            <span className="text-xs font-bold text-white">Object #{idx + 1}</span>
                            <span className={`px-1.5 py-0.5 text-white text-[9px] font-bold rounded ${
                              statusLabel === 'READY' ? 'bg-emerald-500' :
                              statusLabel === 'STABILIZING' ? 'bg-blue-500' :
                              statusLabel === 'EDGE TOUCH' ? 'bg-red-500' :
                              'bg-zinc-600'
                            }`}>
                              {statusLabel}
                            </span>
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
                              <span className="text-zinc-500">Aspect:</span>
                              <span className={`font-mono ${bbox.aspect_ratio_valid ? 'text-zinc-300' : 'text-red-400'}`}>
                                {bbox.aspect_ratio?.toFixed(2) || 'N/A'}
                              </span>
                            </div>
                            {bbox.is_stable !== undefined && (
                              <div className="flex justify-between col-span-2">
                                <span className="text-zinc-500">Stability:</span>
                                <span className={`font-mono ${bbox.is_stable ? 'text-emerald-400' : 'text-yellow-400'}`}>
                                  {bbox.stability_counter || 0}/{8}
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                      );
                    })}
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

          {/* Auto-Classification State Indicator */}
          {inferenceStatus?.is_running && inferenceMode === 'auto' && (
            <div className="bg-zinc-900/50 rounded-xl border border-zinc-800 p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-wider flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 2v4" /><path d="m16.2 7.8 2.9-2.9" /><path d="M18 12h4" /><path d="m16.2 16.2 2.9 2.9" /><path d="M12 18v4" /><path d="m4.9 19.1 2.9-2.9" /><path d="M2 12h4" /><path d="m4.9 4.9 2.9 2.9" />
                  </svg>
                  Auto Classification Pipeline
                </h3>
              </div>

              <div className="flex items-center">
                {/* Step 1: Waiting for Part */}
                <div className={`flex-1 flex flex-col items-center p-3 rounded-lg transition-all ${
                  inferenceStatus.auto_state === 'waiting'
                    ? 'bg-zinc-800 border border-zinc-600'
                    : 'opacity-50'
                }`}>
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 ${
                    inferenceStatus.auto_state === 'waiting'
                      ? 'bg-zinc-700 border-2 border-zinc-500'
                      : 'bg-zinc-800 border border-zinc-700'
                  }`}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={inferenceStatus.auto_state === 'waiting' ? 'text-zinc-300 animate-pulse' : 'text-zinc-600'}>
                      <circle cx="12" cy="12" r="10" /><circle cx="12" cy="12" r="6" /><circle cx="12" cy="12" r="2" />
                    </svg>
                  </div>
                  <span className={`text-[10px] font-bold uppercase tracking-wide ${
                    inferenceStatus.auto_state === 'waiting' ? 'text-zinc-300' : 'text-zinc-600'
                  }`}>
                    Waiting for Part
                  </span>
                  {inferenceStatus.auto_state === 'waiting' && (
                    <span className="text-[9px] text-zinc-500 mt-1">Place part in center</span>
                  )}
                </div>

                {/* Connector 1-2 */}
                <div className={`w-12 h-0.5 ${
                  inferenceStatus.auto_state !== 'waiting' ? 'bg-blue-500' : 'bg-zinc-700'
                }`} />

                {/* Step 2: Classifying */}
                <div className={`flex-1 flex flex-col items-center p-3 rounded-lg transition-all ${
                  inferenceStatus.auto_state === 'stabilizing'
                    ? 'bg-blue-500/20 border border-blue-500/50'
                    : 'opacity-50'
                }`}>
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 ${
                    inferenceStatus.auto_state === 'stabilizing'
                      ? 'bg-blue-500/30 border-2 border-blue-500'
                      : inferenceStatus.auto_state === 'classified' || inferenceStatus.auto_state === 'cooldown'
                        ? 'bg-blue-500/20 border border-blue-500/50'
                        : 'bg-zinc-800 border border-zinc-700'
                  }`}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={
                      inferenceStatus.auto_state === 'stabilizing'
                        ? 'text-blue-400 animate-spin'
                        : inferenceStatus.auto_state === 'classified' || inferenceStatus.auto_state === 'cooldown'
                          ? 'text-blue-400'
                          : 'text-zinc-600'
                    }>
                      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                    </svg>
                  </div>
                  <span className={`text-[10px] font-bold uppercase tracking-wide ${
                    inferenceStatus.auto_state === 'stabilizing' ? 'text-blue-400' :
                    inferenceStatus.auto_state === 'classified' || inferenceStatus.auto_state === 'cooldown' ? 'text-blue-400/70' : 'text-zinc-600'
                  }`}>
                    Classifying
                  </span>
                  {inferenceStatus.auto_state === 'stabilizing' && (
                    <span className="text-[9px] text-blue-400/70 mt-1">Processing...</span>
                  )}
                </div>

                {/* Connector 2-3 */}
                <div className={`w-12 h-0.5 ${
                  inferenceStatus.auto_state === 'classified' || inferenceStatus.auto_state === 'cooldown'
                    ? 'bg-emerald-500'
                    : 'bg-zinc-700'
                }`} />

                {/* Step 3: Waiting for Exit */}
                <div className={`flex-1 flex flex-col items-center p-3 rounded-lg transition-all ${
                  inferenceStatus.auto_state === 'classified' || inferenceStatus.auto_state === 'cooldown'
                    ? 'bg-emerald-500/20 border border-emerald-500/50'
                    : 'opacity-50'
                }`}>
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 ${
                    inferenceStatus.auto_state === 'classified' || inferenceStatus.auto_state === 'cooldown'
                      ? 'bg-emerald-500/30 border-2 border-emerald-500'
                      : 'bg-zinc-800 border border-zinc-700'
                  }`}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={
                      inferenceStatus.auto_state === 'classified' || inferenceStatus.auto_state === 'cooldown'
                        ? 'text-emerald-400 animate-pulse'
                        : 'text-zinc-600'
                    }>
                      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><polyline points="22 4 12 14.01 9 11.01" />
                    </svg>
                  </div>
                  <span className={`text-[10px] font-bold uppercase tracking-wide ${
                    inferenceStatus.auto_state === 'classified' || inferenceStatus.auto_state === 'cooldown'
                      ? 'text-emerald-400'
                      : 'text-zinc-600'
                  }`}>
                    Remove Part
                  </span>
                  {(inferenceStatus.auto_state === 'classified' || inferenceStatus.auto_state === 'cooldown') && (
                    <span className="text-[9px] text-emerald-400/70 mt-1">Ready for next</span>
                  )}
                </div>
              </div>
            </div>
          )}

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

          {/* Classification History */}
          {classificationHistory.length > 0 && (
            <div className="bg-zinc-900/50 rounded-xl border border-zinc-800">
              <h2 className="text-sm font-medium text-zinc-400 mb-3 flex items-center gap-2 p-4 pb-0">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></svg>
                CLASSIFICATION HISTORY
              </h2>
              <div className="bg-zinc-950 rounded-lg border border-zinc-800 max-h-[300px] overflow-y-auto m-4 mt-2">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-zinc-900 border-b border-zinc-800">
                    <tr>
                      <th className="text-left p-2 text-zinc-500 font-semibold">Image</th>
                      <th className="text-left p-2 text-zinc-500 font-semibold">Part ID</th>
                      <th className="text-right p-2 text-zinc-500 font-semibold">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {classificationHistory.slice().reverse().map((entry: any, idx: number) => (
                      <tr key={idx} className="border-b border-zinc-800 hover:bg-zinc-900/50 transition-colors">
                        <td className="p-2">
                          <img
                            src={`data:image/jpeg;base64,${entry.thumbnail}`}
                            alt={entry.part_id}
                            className="w-12 h-12 rounded border border-zinc-700 object-contain"
                          />
                        </td>
                        <td className="p-2">
                          <span className="font-mono text-white font-bold">{entry.part_id}</span>
                        </td>
                        <td className="p-2 text-right">
                          <span className={`font-mono font-bold ${entry.confidence > 0.7 ? 'text-emerald-400' : entry.confidence > 0.5 ? 'text-yellow-400' : 'text-red-400'}`}>
                            {(entry.confidence * 100).toFixed(1)}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
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

              {/* Mode Switch */}
              <div className="grid grid-cols-2 gap-2 mt-2 bg-zinc-950/50 p-1 rounded-lg border border-zinc-800">
                <button
                  onClick={() => setMode('auto')}
                  className={`py-2 rounded px-3 text-xs font-medium transition-all flex items-center justify-center gap-2 ${inferenceMode === 'auto'
                      ? 'bg-zinc-800 text-white shadow-sm border border-zinc-700'
                      : 'text-zinc-500 hover:text-zinc-300'
                    }`}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2v4" /><path d="m16.2 7.8 2.9-2.9" /><path d="M18 12h4" /><path d="m16.2 16.2 2.9 2.9" /><path d="M12 18v4" /><path d="m4.9 19.1 2.9-2.9" /><path d="M2 12h4" /><path d="m4.9 4.9 2.9 2.9" /></svg>
                  AUTO
                </button>
                <button
                  onClick={() => setMode('manual')}
                  className={`py-2 rounded px-3 text-xs font-medium transition-all flex items-center justify-center gap-2 ${inferenceMode === 'manual'
                      ? 'bg-zinc-800 text-white shadow-sm border border-zinc-700'
                      : 'text-zinc-500 hover:text-zinc-300'
                    }`}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><circle cx="12" cy="12" r="3" /></svg>
                  MANUAL
                </button>
              </div>
            </div>

            {/* Calibration Section */}
            <div className="pt-4 border-t border-zinc-800">
              <h2 className="text-sm font-medium text-zinc-400 mb-4 flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2v4" /><path d="m16.2 7.8 2.9-2.9" /><path d="M18 12h4" /><path d="m16.2 16.2 2.9 2.9" /><path d="M12 18v4" /><path d="m4.9 19.1 2.9-2.9" /><path d="M2 12h4" /><path d="m4.9 4.9 2.9 2.9" /></svg>
                BACKGROUND CALIBRATION
              </h2>

              {/* Calibration Status */}
              <div className="mb-3 pb-3 border-b border-zinc-700">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-zinc-500">Status:</span>
                  <span className={`text-xs font-mono font-bold px-2 py-0.5 rounded ${calibrationStatus?.is_calibrated ? 'bg-emerald-500/20 text-emerald-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                    {calibrationStatus?.is_calibrated ? '✓ CALIBRATED' : '◯ NOT CALIBRATED'}
                  </span>
                </div>
                <p className="text-[10px] text-zinc-500">
                  {calibrationStatus?.calibration_status || 'No calibration data'}
                </p>
              </div>

              {/* Calibration Controls */}
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={calibrateBackground}
                  disabled={isCalibrating || !inferenceStatus?.is_running}
                  className="col-span-2 group relative overflow-hidden bg-amber-600 hover:bg-amber-500 active:bg-amber-700 disabled:bg-zinc-700 disabled:cursor-not-allowed text-zinc-100 py-3 rounded-lg transition-all border border-amber-500 disabled:border-zinc-600 flex items-center justify-center gap-2 font-medium text-sm"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700" />
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="1" /><path d="M12 8v-2" /><path d="M4.22 4.22l1.42 1.42" /><path d="M1.5 12h2" /><path d="M4.22 19.78l1.42-1.42" /><path d="M12 16v2" /><path d="M19.78 19.78l-1.42-1.42" /><path d="M22.5 12h-2" /><path d="M19.78 4.22l-1.42 1.42" /></svg>
                  {isCalibrating ? 'Calibrating...' : 'Calibrate Now'}
                </button>
                <button
                  onClick={resetCalibration}
                  disabled={!calibrationStatus?.is_calibrated}
                  className="col-span-2 bg-zinc-700 hover:bg-zinc-600 active:bg-zinc-800 disabled:bg-zinc-800 disabled:cursor-not-allowed text-zinc-100 py-2.5 rounded-lg transition-all border border-zinc-600 disabled:border-zinc-700 flex items-center justify-center gap-2 font-medium text-sm"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 4.5a10 10 0 0 1-18.8 4.2" /></svg>
                  Reset Calibration
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
              <div><span className="text-zinc-600">[INIT]</span> System modules loaded</div>
              <div><span className="text-zinc-600">[OK]</span> <span className="text-emerald-500">Webcam connected</span></div>
              {inferenceStatus?.is_running && (
                <div><span className="text-zinc-600">[RUN]</span> <span className="text-blue-500">Inference engine running</span></div>
              )}
              {classification && (
                <div><span className="text-zinc-600">[DETECT]</span> Detected: {classification.class_name} ({(classification.confidence * 100).toFixed(1)}%)</div>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
