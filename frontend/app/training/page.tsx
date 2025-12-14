"use client";
import { useState, useEffect, useRef } from 'react';

export default function TrainingPage() {
    const [isTraining, setIsTraining] = useState(false);
    const [status, setStatus] = useState<any>(null);
    const [images, setImages] = useState<any>(null);
    const [logs, setLogs] = useState<string[]>([]);
    const [dataset, setDataset] = useState<"ldraw" | "ldview" | "rebrickable">("ldraw");
    const [epochs, setEpochs] = useState(10);
    const [batchSize, setBatchSize] = useState(8);
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

    useEffect(() => {
        // Poll status every second
        pollInterval.current = setInterval(fetchStatus, 1000);
        return () => {
            if (pollInterval.current) clearInterval(pollInterval.current);
        };
    }, []);

    const startTraining = async () => {
        try {
            await fetch(`${API_URL}/train/start?epochs=${epochs}&batch_size=${batchSize}&dataset=${dataset}`, { method: 'POST' });
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

                        <div>
                            <h3 className="text-sm font-medium text-zinc-400 mb-3">DATASET SOURCE</h3>
                            <div className="space-y-2">
                                <label className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${dataset === 'ldraw' ? 'bg-emerald-500/10 border-emerald-500/50' : 'bg-zinc-950 border-zinc-800 hover:border-zinc-700'}`}>
                                    <input
                                        type="radio"
                                        name="dataset"
                                        value="ldraw"
                                        checked={dataset === 'ldraw'}
                                        onChange={(e) => setDataset(e.target.value as any)}
                                        disabled={isTraining}
                                        className="w-4 h-4"
                                    />
                                    <div className="flex-1">
                                        <div className="text-sm font-medium text-zinc-200">LDraw Python</div>
                                        <div className="text-xs text-zinc-500">Software renderer, multi-view</div>
                                    </div>
                                </label>

                                <label className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${dataset === 'ldview' ? 'bg-emerald-500/10 border-emerald-500/50' : 'bg-zinc-950 border-zinc-800 hover:border-zinc-700'}`}>
                                    <input
                                        type="radio"
                                        name="dataset"
                                        value="ldview"
                                        checked={dataset === 'ldview'}
                                        onChange={(e) => setDataset(e.target.value as any)}
                                        disabled={isTraining}
                                        className="w-4 h-4"
                                    />
                                    <div className="flex-1">
                                        <div className="text-sm font-medium text-zinc-200">LDView Renders</div>
                                        <div className="text-xs text-zinc-500">Pre-generated realistic 3D</div>
                                    </div>
                                </label>

                                <label className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${dataset === 'rebrickable' ? 'bg-emerald-500/10 border-emerald-500/50' : 'bg-zinc-950 border-zinc-800 hover:border-zinc-700'}`}>
                                    <input
                                        type="radio"
                                        name="dataset"
                                        value="rebrickable"
                                        checked={dataset === 'rebrickable'}
                                        onChange={(e) => setDataset(e.target.value as any)}
                                        disabled={isTraining}
                                        className="w-4 h-4"
                                    />
                                    <div className="flex-1">
                                        <div className="text-sm font-medium text-zinc-200">Rebrickable CGI</div>
                                        <div className="text-xs text-zinc-500">On-the-fly synthesis</div>
                                    </div>
                                </label>
                            </div>
                        </div>

                        <div className="h-px bg-zinc-800 my-1" />

                        <div>
                            <h3 className="text-sm font-medium text-zinc-400 mb-3">TRAINING PARAMETERS</h3>
                            <div className="space-y-3">
                                <div>
                                    <label className="text-xs text-zinc-500 block mb-1">Epochs</label>
                                    <input
                                        type="number"
                                        min="1"
                                        max="100"
                                        value={epochs}
                                        onChange={(e) => setEpochs(parseInt(e.target.value) || 10)}
                                        disabled={isTraining}
                                        className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-2 text-sm text-zinc-200 disabled:opacity-50"
                                    />
                                </div>
                                <div>
                                    <label className="text-xs text-zinc-500 block mb-1">Batch Size</label>
                                    <input
                                        type="number"
                                        min="1"
                                        max="64"
                                        value={batchSize}
                                        onChange={(e) => setBatchSize(parseInt(e.target.value) || 8)}
                                        disabled={isTraining}
                                        className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-2 text-sm text-zinc-200 disabled:opacity-50"
                                    />
                                </div>
                            </div>
                        </div>

                        <div className="h-px bg-zinc-800 my-1" />

                        <div className="text-xs text-zinc-500">
                            <p>Backbone: MobileNetV3</p>
                            <p>Loss: TripletMargin</p>
                        </div>
                    </div>
                </section>
            </main>
        </div>
    );
}
