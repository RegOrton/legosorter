"use client";
import { useState, useEffect, useRef } from 'react';

export default function TrainingPage() {
    const [isTraining, setIsTraining] = useState(false);
    const [status, setStatus] = useState<any>(null);
    const [images, setImages] = useState<any>(null);
    const [logs, setLogs] = useState<string[]>([]);
    const [settings, setSettings] = useState<any>(null);
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

    useEffect(() => {
        // Fetch settings on mount
        fetchSettings();

        // Poll status every second
        pollInterval.current = setInterval(fetchStatus, 1000);
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

    return (
        <div className="flex flex-col text-zinc-50 h-full">
            <main className="flex-1 p-6 grid grid-cols-1 md:grid-cols-12 gap-6 max-w-[1600px] mx-auto w-full">
                {/* Left Column: Logs & Viz */}
                <section className="col-span-1 md:col-span-8 flex flex-col gap-6">
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
                </section>

                {/* Right Column: Controls */}
                <section className="col-span-1 md:col-span-4 flex flex-col gap-6">
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

                        <div className="text-xs text-zinc-500 p-4 bg-zinc-950 rounded-lg border border-zinc-800">
                            <p className="mb-2">Training will use settings configured in the Settings tab.</p>
                            <p>Backbone: MobileNetV3</p>
                            <p>Loss: TripletMargin</p>
                        </div>
                    </div>
                </section>
            </main>
        </div>
    );
}
