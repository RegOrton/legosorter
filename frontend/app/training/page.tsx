"use client";
import { useState, useEffect, useRef } from 'react';

export default function TrainingPage() {
    const [isTraining, setIsTraining] = useState(false);
    const [status, setStatus] = useState<any>(null);
    const [images, setImages] = useState<any>(null);
    const [logs, setLogs] = useState<string[]>([]);
    const [settings, setSettings] = useState<any>(null);
    const [calibrationBg, setCalibrationBg] = useState<any>(null);
    const pollInterval = useRef<NodeJS.Timeout | null>(null);

    const API_URL = "http://localhost:8000";

    const fetchStatus = async () => {
        try {
            const res = await fetch(`${API_URL}/train/status`);
            const data = await res.json();
            setStatus(data);
            setIsTraining(data.is_running);
            if (data.logs) setLogs(data.logs);
            if (data.images) setImages(data.images);
        } catch (e) {
            console.error("Failed to fetch status", e);
        }
    };

    const fetchSettings = async () => {
        try {
            const res = await fetch(`${API_URL}/settings`);
            const data = await res.json();
            setSettings(data);
        } catch (e) {
            console.error("Failed to fetch settings", e);
        }
    };

    const fetchCalibrationBg = async () => {
        try {
            const res = await fetch(`${API_URL}/inference/detector/calibration_bg`);
            if (res.ok) {
                const data = await res.json();
                setCalibrationBg(data);
            } else {
                setCalibrationBg(null);
            }
        } catch (e) {
            console.error("Failed to fetch calibration background", e);
            setCalibrationBg(null);
        }
    };

    useEffect(() => {
        // Fetch settings and calibration on mount
        fetchSettings();
        fetchCalibrationBg();

        // Poll status rapidly for flicker mode (200ms = 5 times per second)
        pollInterval.current = setInterval(fetchStatus, 200);
        return () => {
            if (pollInterval.current) clearInterval(pollInterval.current);
        };
    }, []);

    const startTraining = async () => {
        if (!settings) {
            alert("Settings not loaded yet");
            return;
        }

        try {
            await fetch(`${API_URL}/train/start?epochs=${settings.epochs}&batch_size=${settings.batch_size}&dataset=${settings.dataset}`, { method: 'POST' });
            fetchStatus();
        } catch (e) {
            alert("Failed to start training: " + e);
        }
    };

    const stopTraining = async () => {
        try {
            await fetch(`${API_URL}/train/stop`, { method: 'POST' });
            fetchStatus();
        } catch (e) {
            alert("Failed to stop training: " + e);
        }
    };

    // Calculate progress percentage
    const progressPercent = status?.total_batches > 0
        ? (status.batch_number / status.total_batches) * 100
        : 0;

    return (
        <div className="flex flex-col text-zinc-50 h-full">
            <main className="flex-1 p-6 grid grid-cols-1 md:grid-cols-12 gap-6 max-w-[1600px] mx-auto w-full">
                {/* Left Column: Logs & Viz */}
                <section className="col-span-1 md:col-span-8 flex flex-col gap-6">
                    {/* Checkpoint Status Banner */}
                    {status && (status.checkpoint_loaded !== undefined) && (
                        <div className={`rounded-xl border p-5 ${
                            status.checkpoint_loaded
                                ? 'bg-blue-950/30 border-blue-800'
                                : 'bg-amber-950/30 border-amber-800'
                        }`}>
                            <div className="flex items-start gap-3">
                                <div className="flex-shrink-0 mt-1">
                                    {status.checkpoint_loaded ? (
                                        <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                    ) : (
                                        <svg className="w-5 h-5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                        </svg>
                                    )}
                                </div>
                                <div className="flex-1">
                                    <h3 className={`text-sm font-bold mb-1 ${
                                        status.checkpoint_loaded ? 'text-blue-300' : 'text-amber-300'
                                    }`}>
                                        {status.checkpoint_loaded ? 'CONTINUING FROM CHECKPOINT' : 'TRAINING FROM SCRATCH'}
                                    </h3>
                                    <p className="text-xs text-zinc-400 leading-relaxed">
                                        {status.checkpoint_loaded ? (
                                            <>
                                                Training is <strong className="text-blue-400">improving the existing model</strong> ({status.checkpoint_path}).
                                                All previously learned knowledge is preserved. New parts added to the dataset will be learned
                                                while maintaining recognition of old parts.
                                            </>
                                        ) : (
                                            <>
                                                Training is starting with <strong className="text-amber-400">ImageNet pretrained weights</strong>.
                                                No existing LEGO model checkpoint was found. The model will learn from scratch.
                                            </>
                                        )}
                                    </p>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Progress Bar */}
                    {isTraining && (
                        <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-5">
                            <div className="flex justify-between mb-2">
                                <h3 className="text-sm font-bold text-zinc-400">EPOCH PROGRESS</h3>
                                <span className="text-sm text-zinc-400">
                                    Batch {status?.batch_number || 0} / {status?.total_batches || 0}
                                </span>
                            </div>
                            <div className="w-full bg-zinc-800 rounded-full h-3 overflow-hidden">
                                <div
                                    className="bg-emerald-500 h-full transition-all duration-300"
                                    style={{ width: `${progressPercent}%` }}
                                />
                            </div>
                        </div>
                    )}

                    {/* Loss History Graph */}
                    {status?.loss_history && status.loss_history.length > 0 && (
                        <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-5">
                            <h3 className="text-sm font-bold text-zinc-400 mb-4">LOSS HISTORY</h3>
                            <div className="relative h-48">
                                <svg viewBox="0 0 400 150" className="w-full h-full">
                                    {/* Grid lines */}
                                    {[0, 1, 2, 3, 4].map((i) => (
                                        <line
                                            key={i}
                                            x1="0"
                                            y1={i * 37.5}
                                            x2="400"
                                            y2={i * 37.5}
                                            stroke="#27272a"
                                            strokeWidth="1"
                                        />
                                    ))}

                                    {/* Loss line */}
                                    <polyline
                                        points={status.loss_history.map((point: any, i: number) => {
                                            const x = (i / Math.max(status.loss_history.length - 1, 1)) * 400;
                                            const maxLoss = Math.max(...status.loss_history.map((p: any) => p.loss));
                                            const y = 150 - (point.loss / maxLoss) * 140;
                                            return `${x},${y}`;
                                        }).join(' ')}
                                        fill="none"
                                        stroke="#10b981"
                                        strokeWidth="2"
                                    />

                                    {/* Data points */}
                                    {status.loss_history.map((point: any, i: number) => {
                                        const x = (i / Math.max(status.loss_history.length - 1, 1)) * 400;
                                        const maxLoss = Math.max(...status.loss_history.map((p: any) => p.loss));
                                        const y = 150 - (point.loss / maxLoss) * 140;
                                        return (
                                            <circle
                                                key={i}
                                                cx={x}
                                                cy={y}
                                                r="3"
                                                fill="#10b981"
                                            />
                                        );
                                    })}
                                </svg>
                            </div>
                        </div>
                    )}

                    {/* Timing Breakdown */}
                    {status?.timing_stats && status.timing_stats.total_time > 0 && (
                        <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-5">
                            <h3 className="text-sm font-bold text-zinc-400 mb-4">TIME BREAKDOWN</h3>
                            <div className="space-y-3">
                                <div>
                                    <div className="flex justify-between mb-1">
                                        <span className="text-xs text-zinc-500">Data Generation</span>
                                        <span className="text-xs text-zinc-400">{status.timing_stats.data_generation.toFixed(1)}%</span>
                                    </div>
                                    <div className="w-full bg-zinc-800 rounded-full h-2">
                                        <div
                                            className="bg-blue-500 h-full rounded-full"
                                            style={{ width: `${status.timing_stats.data_generation}%` }}
                                        />
                                    </div>
                                </div>
                                <div>
                                    <div className="flex justify-between mb-1">
                                        <span className="text-xs text-zinc-500">Forward Pass</span>
                                        <span className="text-xs text-zinc-400">{status.timing_stats.forward_pass.toFixed(1)}%</span>
                                    </div>
                                    <div className="w-full bg-zinc-800 rounded-full h-2">
                                        <div
                                            className="bg-purple-500 h-full rounded-full"
                                            style={{ width: `${status.timing_stats.forward_pass}%` }}
                                        />
                                    </div>
                                </div>
                                <div>
                                    <div className="flex justify-between mb-1">
                                        <span className="text-xs text-zinc-500">Backward Pass</span>
                                        <span className="text-xs text-zinc-400">{status.timing_stats.backward_pass.toFixed(1)}%</span>
                                    </div>
                                    <div className="w-full bg-zinc-800 rounded-full h-2">
                                        <div
                                            className="bg-orange-500 h-full rounded-full"
                                            style={{ width: `${status.timing_stats.backward_pass}%` }}
                                        />
                                    </div>
                                </div>
                                <div className="pt-2 border-t border-zinc-800">
                                    <div className="flex justify-between">
                                        <span className="text-xs font-bold text-zinc-400">Total Time</span>
                                        <span className="text-xs font-bold text-emerald-400">{status.timing_stats.total_time.toFixed(2)}s</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    <div className="flex flex-col gap-4">
                        <h2 className="text-xl font-bold tracking-tight">Live Training Feed</h2>

                        {/* Image Preview */}
                        <div className="grid grid-cols-3 gap-4">
                            <div className="flex flex-col gap-2">
                                <span className="text-xs font-bold text-zinc-500 uppercase">Anchor</span>
                                <div className="aspect-square bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden relative">
                                    {images?.anchor ? (
                                        <img src={`data:image/jpeg;base64,${images.anchor}`} className="w-full h-full object-cover" />
                                    ) : (
                                        <div className="absolute inset-0 flex items-center justify-center text-zinc-700 text-xs">Waiting...</div>
                                    )}
                                </div>
                            </div>
                            <div className="flex flex-col gap-2">
                                <span className="text-xs font-bold text-zinc-500 uppercase">Positive</span>
                                <div className="aspect-square bg-zinc-900 border border-emerald-900/50 rounded-lg overflow-hidden relative">
                                    {images?.positive ? (
                                        <img src={`data:image/jpeg;base64,${images.positive}`} className="w-full h-full object-cover" />
                                    ) : (
                                        <div className="absolute inset-0 flex items-center justify-center text-zinc-700 text-xs">Waiting...</div>
                                    )}
                                </div>
                            </div>
                            <div className="flex flex-col gap-2">
                                <span className="text-xs font-bold text-zinc-500 uppercase">Negative</span>
                                <div className="aspect-square bg-zinc-900 border border-red-900/50 rounded-lg overflow-hidden relative">
                                    {images?.negative ? (
                                        <img src={`data:image/jpeg;base64,${images.negative}`} className="w-full h-full object-cover" />
                                    ) : (
                                        <div className="absolute inset-0 flex items-center justify-center text-zinc-700 text-xs">Waiting...</div>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="flex flex-col gap-4 h-full">
                        <h2 className="text-xl font-bold tracking-tight">Logs</h2>

                        <div className="bg-zinc-950 rounded-xl border border-zinc-800 p-4 font-mono text-xs overflow-y-auto h-[500px] flex flex-col-reverse">
                            {logs.length === 0 && <span className="text-zinc-600 italic">No logs yet...</span>}
                            {logs.map((log, i) => (
                                <div key={i} className="py-1 border-b border-zinc-900 text-zinc-300">
                                    <span className="text-zinc-600 mr-2">[{i}]</span>
                                    {log}
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Parts Statistics Table */}
                    {status?.parts_stats && status.parts_stats.length > 0 && (
                        <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-5">
                            <h3 className="text-sm font-bold text-zinc-400 mb-4">PARTS STATISTICS</h3>
                            <div className="bg-zinc-950 rounded-lg border border-zinc-800 overflow-hidden">
                                <div className="max-h-96 overflow-y-auto">
                                    <table className="w-full text-xs">
                                        <thead className="sticky top-0 bg-zinc-900 border-b border-zinc-800">
                                            <tr>
                                                <th className="text-left p-3 text-zinc-500 font-semibold">Part ID</th>
                                                <th className="text-center p-3 text-zinc-500 font-semibold">Views</th>
                                                <th className="text-center p-3 text-zinc-500 font-semibold">Epochs</th>
                                                <th className="text-center p-3 text-zinc-500 font-semibold">Samples</th>
                                                <th className="text-right p-3 text-zinc-500 font-semibold">Avg Loss</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {status.parts_stats.map((part: any, idx: number) => (
                                                <tr key={idx} className="border-b border-zinc-800 hover:bg-zinc-900/50 transition-colors">
                                                    <td className="p-3">
                                                        <span className="font-mono text-emerald-400 font-bold">{part.part_id}</span>
                                                    </td>
                                                    <td className="p-3 text-center">
                                                        <span className="text-zinc-300">{part.views}</span>
                                                    </td>
                                                    <td className="p-3 text-center">
                                                        <span className="text-zinc-300">{part.epochs}</span>
                                                    </td>
                                                    <td className="p-3 text-center">
                                                        <span className="text-zinc-300">{part.samples}</span>
                                                    </td>
                                                    <td className="p-3 text-right">
                                                        <span className={`font-mono font-bold ${
                                                            part.avg_loss < 0.5 ? 'text-emerald-400' :
                                                            part.avg_loss < 1.0 ? 'text-yellow-400' :
                                                            'text-orange-400'
                                                        }`}>
                                                            {part.avg_loss.toFixed(4)}
                                                        </span>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                                <div className="bg-zinc-900 border-t border-zinc-800 p-2 text-xs text-zinc-500">
                                    <span>{status.parts_stats.length} part(s) in training</span>
                                </div>
                            </div>
                        </div>
                    )}
                </section>

                {/* Right Column: Controls */}
                <section className="col-span-1 md:col-span-4 flex flex-col gap-6">
                    {/* Calibration Background Display */}
                    {calibrationBg && (
                        <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-5 shadow-lg">
                            <h2 className="text-sm font-medium text-zinc-400 mb-4">CALIBRATION BACKGROUND</h2>
                            <div className="bg-zinc-950 rounded-lg border border-zinc-800 overflow-hidden">
                                <img
                                    src={`data:image/jpeg;base64,${calibrationBg.calibration_image}`}
                                    alt="Calibration Background"
                                    className="w-full h-auto"
                                />
                            </div>
                            <div className="mt-3 text-xs text-zinc-500">
                                <p>Resolution: {calibrationBg.resolution.width} × {calibrationBg.resolution.height}</p>
                                <p className="mt-1 text-emerald-500">Training uses this background for realistic augmentation</p>
                            </div>
                        </div>
                    )}

                    <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-5 flex flex-col gap-6 shadow-lg">
                        <div>
                            <h2 className="text-sm font-medium text-zinc-400 mb-4">TRAINING CONTROL</h2>

                            <div className="bg-zinc-950 p-4 rounded-lg border border-zinc-800 mb-4">
                                <div className="flex justify-between items-center mb-2">
                                    <span className="text-zinc-500 text-xs uppercase font-bold">Status</span>
                                    <span className={`text-xs font-bold px-2 py-1 rounded ${isTraining ? 'bg-emerald-500/20 text-emerald-400' : 'bg-zinc-800 text-zinc-400'}`}>
                                        {isTraining ? 'RUNNING' : 'IDLE'}
                                    </span>
                                </div>
                                <div className="grid grid-cols-2 gap-4 mt-4">
                                    <div>
                                        <div className="text-xs text-zinc-500">Epoch</div>
                                        <div className="text-xl font-mono">{status?.epoch || 0} <span className="text-zinc-600 text-sm">/ {status?.total_epochs || 0}</span></div>
                                    </div>
                                    <div>
                                        <div className="text-xs text-zinc-500">Loss</div>
                                        <div className={`text-xl font-mono ${status?.loss < 0.1 ? 'text-emerald-400' : 'text-zinc-200'}`}>
                                            {status?.loss?.toFixed(4) || "0.0000"}
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {!isTraining ? (
                                <button
                                    onClick={startTraining}
                                    className="w-full bg-emerald-600 hover:bg-emerald-500 text-white py-3 rounded-lg text-sm font-bold transition-all shadow-lg flex items-center justify-center gap-2"
                                >
                                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 3 19 12 5 21 5 3" /></svg>
                                    Start Training
                                </button>
                            ) : (
                                <button
                                    onClick={stopTraining}
                                    className="w-full bg-red-600 hover:bg-red-500 text-white py-3 rounded-lg text-sm font-bold transition-all shadow-lg flex items-center justify-center gap-2 animate-pulse"
                                >
                                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="6" y="4" width="4" height="16" /><rect x="14" y="4" width="4" height="16" /></svg>
                                    Stop Training
                                </button>
                            )}
                        </div>

                        <div className="h-px bg-zinc-800 my-1" />

                        <div className="text-xs text-zinc-500 p-4 bg-zinc-950 rounded-lg border border-zinc-800 space-y-2">
                            <p>Training will use settings configured in the Settings tab.</p>
                            <p>Backbone: MobileNetV3</p>
                            <p>Loss: TripletMargin</p>
                            <div className="pt-2 mt-2 border-t border-zinc-800">
                                <p className={`font-bold ${status?.checkpoint_loaded ? 'text-blue-400' : 'text-amber-400'}`}>
                                    {status?.checkpoint_loaded ? '⚡ Incremental Training Mode' : '🆕 Fresh Training Mode'}
                                </p>
                            </div>
                        </div>
                    </div>
                </section>
            </main>
        </div>
    );
}
