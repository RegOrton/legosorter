export default function TrainingPage() {
    return (
        <div className="flex flex-col text-zinc-50 h-full">
            <main className="flex-1 p-6 grid grid-cols-1 md:grid-cols-12 gap-6 max-w-[1600px] mx-auto w-full">
                {/* Left Column: Capture & Preview */}
                <section className="col-span-1 md:col-span-8 flex flex-col gap-6">
                    <div className="flex flex-col gap-4">
                        <h2 className="text-xl font-bold tracking-tight">Data Collection</h2>

                        {/* Capture Area */}
                        <div className="bg-zinc-900/50 rounded-xl border border-zinc-800 aspect-video flex items-center justify-center relative overflow-hidden group shadow-2xl">
                            <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>

                            <div className="z-10 text-center flex flex-col items-center gap-3">
                                <div className="h-12 w-12 rounded-full bg-zinc-800 flex items-center justify-center">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-zinc-600"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" /><circle cx="12" cy="13" r="4" /></svg>
                                </div>
                                <div>
                                    <p className="text-zinc-400 font-mono text-sm">/dev/video0</p>
                                    <p className="text-xs text-zinc-600 mt-1">Live Feed Active</p>
                                </div>
                            </div>

                            <div className="absolute bottom-6 flex gap-4">
                                <button className="bg-white text-black hover:bg-zinc-200 px-6 py-2 rounded-full font-bold shadow-lg flex items-center gap-2 transition-all active:scale-95">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><circle cx="12" cy="12" r="3" /></svg>
                                    Capture Frame
                                </button>
                            </div>
                        </div>
                    </div>

                    <div className="flex flex-col gap-4">
                        <h2 className="text-xl font-bold tracking-tight">Recent Captures</h2>
                        <div className="grid grid-cols-4 md:grid-cols-6 gap-3">
                            {[1, 2, 3, 4, 5, 6].map((i) => (
                                <div key={i} className="aspect-square bg-zinc-900 rounded-lg border border-zinc-800 flex items-center justify-center relative group cursor-pointer hover:border-zinc-600 transition-all">
                                    <span className="text-xs text-zinc-600">Img_{i}</span>
                                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                                        <button className="text-xs bg-red-500/20 text-red-500 p-1 rounded border border-red-500/50">Del</button>
                                    </div>
                                </div>
                            ))}
                            <div className="aspect-square bg-zinc-900/50 rounded-lg border border-zinc-800 border-dashed flex items-center justify-center text-zinc-600 hover:text-zinc-400 hover:border-zinc-600 transition-all cursor-pointer">
                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Right Column: Labeling & Controls */}
                <section className="col-span-1 md:col-span-4 flex flex-col gap-6">
                    <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-5 flex flex-col gap-6 shadow-lg">
                        <div>
                            <h2 className="text-sm font-medium text-zinc-400 mb-4 flex items-center gap-2">LABELIG TOOL</h2>
                            <div className="space-y-4">
                                <div className="space-y-1.5">
                                    <label className="text-xs text-zinc-500 font-medium ml-1">Part ID / Number</label>
                                    <input type="text" placeholder="e.g. 3001" className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/50 transition-all" />
                                </div>
                                <div className="space-y-1.5">
                                    <label className="text-xs text-zinc-500 font-medium ml-1">Color Name</label>
                                    <select className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-emerald-500/50 transition-all appearance-none cursor-pointer hover:bg-zinc-900">
                                        <option>Select Color...</option>
                                        <option>Red</option>
                                        <option>Blue</option>
                                        <option>Yellow</option>
                                        <option>Black</option>
                                        <option>White</option>
                                        <option>Light Bluish Gray</option>
                                    </select>
                                </div>

                                <button className="w-full bg-emerald-600 hover:bg-emerald-500 text-white py-2 rounded-lg text-sm font-medium transition-colors mt-2">
                                    Save Label
                                </button>
                            </div>
                        </div>

                        <div className="h-px bg-zinc-800 my-1" />

                        <div>
                            <h2 className="text-sm font-medium text-zinc-400 mb-4">STATS</h2>
                            <div className="grid grid-cols-2 gap-3">
                                <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                                    <div className="text-xs text-zinc-500 mb-1">Total Images</div>
                                    <div className="text-xl font-mono text-zinc-200">2,543</div>
                                </div>
                                <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                                    <div className="text-xs text-zinc-500 mb-1">Classes</div>
                                    <div className="text-xl font-mono text-zinc-200">142</div>
                                </div>
                            </div>
                        </div>

                        <div className="mt-auto pt-4 border-t border-zinc-800">
                            <button className="w-full bg-zinc-100 hover:bg-white text-zinc-900 py-3 rounded-lg text-sm font-bold tracking-wide transition-all shadow-lg flex items-center justify-center gap-2">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" /></svg>
                                Train Model
                            </button>
                            <p className="text-[10px] text-zinc-600 text-center mt-3">
                                Last trained: 2 hours ago (v0.4.1)
                            </p>
                        </div>
                    </div>
                </section>
            </main>
        </div>
    );
}
