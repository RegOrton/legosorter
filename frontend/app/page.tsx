export default function Home() {
  return (
    <div className="flex flex-col text-zinc-50 h-full">


      <main className="flex-1 p-6 grid grid-cols-1 md:grid-cols-12 gap-6 max-w-[1600px] mx-auto w-full">
        {/* Main Video Feed Area */}
        <section className="col-span-1 md:col-span-8 flex flex-col gap-4">
          <div className="bg-zinc-900/50 rounded-xl border border-zinc-800 aspect-video flex items-center justify-center relative overflow-hidden group shadow-2xl">
            {/* Grid Pattern Overlay */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>

            <div className="z-10 text-center flex flex-col items-center gap-3">
              <div className="h-12 w-12 rounded-full bg-zinc-800 flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-zinc-600"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" /><circle cx="12" cy="13" r="3" /></svg>
              </div>
              <div>
                <p className="text-zinc-400 font-mono text-sm">/dev/video0</p>
                <p className="text-xs text-zinc-600 mt-1">Waiting for stream signal...</p>
              </div>
            </div>

            {/* Overlay UI */}
            <div className="absolute top-4 left-4 flex gap-2">
              <span className="px-2 py-1 bg-black/50 backdrop-blur text-[10px] text-zinc-400 rounded border border-white/10 font-mono">1920x1080 @ 30fps</span>
            </div>
          </div>

          {/* Secondary Stats / Graphs Row */}
          <div className="grid grid-cols-3 gap-4 h-32">
            <div className="bg-zinc-900/50 rounded-xl border border-zinc-800 p-4 flex flex-col justify-between">
              <span className="text-xs text-zinc-500 uppercase tracking-wider font-semibold">Processed</span>
              <span className="text-2xl font-mono text-zinc-200">1,204</span>
            </div>
            <div className="bg-zinc-900/50 rounded-xl border border-zinc-800 p-4 flex flex-col justify-between">
              <span className="text-xs text-zinc-500 uppercase tracking-wider font-semibold">Sort Rate</span>
              <span className="text-2xl font-mono text-zinc-200">42 <span className="text-sm text-zinc-500">/min</span></span>
            </div>
            <div className="bg-zinc-900/50 rounded-xl border border-zinc-800 p-4 flex flex-col justify-between">
              <span className="text-xs text-zinc-500 uppercase tracking-wider font-semibold">Accuracy</span>
              <span className="text-2xl font-mono text-emerald-500">98.2%</span>
            </div>
          </div>
        </section>

        {/* Sidebar Controls */}
        <section className="col-span-1 md:col-span-4 flex flex-col gap-4">
          <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-5 flex flex-col gap-6 shadow-lg">
            <div>
              <h2 className="text-sm font-medium text-zinc-400 mb-4 flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2v20" /><path d="M2 12h20" /><path d="m4.93 4.93 14.14 14.14" /><path d="m19.07 4.93-14.14 14.14" /></svg>
                HARDWARE CONTROL
              </h2>
              <div className="grid grid-cols-2 gap-3">
                <button className="col-span-2 group relative overflow-hidden bg-zinc-800 hover:bg-zinc-700 active:bg-zinc-600 text-zinc-100 py-4 rounded-lg transition-all border border-zinc-700 flex items-center justify-center gap-2 font-medium">
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700" />
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 3 19 12 5 21 5 3" /></svg>
                  Start Conveyor
                </button>
                <div className="col-span-2 h-px bg-zinc-800 my-1" />
                <button className="bg-zinc-800 hover:bg-zinc-700 text-xs py-3 rounded-lg border border-zinc-700 text-zinc-300">Jog FWD</button>
                <button className="bg-zinc-800 hover:bg-zinc-700 text-xs py-3 rounded-lg border border-zinc-700 text-zinc-300">Jog REV</button>
              </div>
            </div>

            <div className="pt-2">
              <h2 className="text-sm font-medium text-zinc-400 mb-4">DIVERTER</h2>
              <div className="flex p-1 bg-zinc-950 rounded-lg border border-zinc-800">
                <button className="flex-1 py-2 text-xs font-medium rounded bg-zinc-800 text-zinc-200 shadow-sm transition-all">Left (Bin A)</button>
                <button className="flex-1 py-2 text-xs font-medium rounded text-zinc-500 hover:text-zinc-300 transition-all">Right (Bin B)</button>
              </div>
            </div>

            <div className="mt-auto pt-4 border-t border-zinc-800">
              <button className="w-full bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/20 py-3 rounded-lg text-sm font-bold tracking-wide transition-all uppercase flex items-center justify-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18.36 6.64a9 9 0 1 1-12.73 0" /><line x1="12" y1="2" x2="12" y2="12" /></svg>
                Emergency Stop
              </button>
            </div>
          </div>

          {/* Logs Panel */}
          <div className="flex-1 bg-zinc-950 rounded-xl border border-zinc-800 p-4 overflow-hidden flex flex-col min-h-[200px]">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-zinc-500">SYSTEM LOGS</span>
              <span className="text-[10px] bg-zinc-900 border border-zinc-800 px-1.5 py-0.5 rounded text-zinc-500">LIVE</span>
            </div>
            <div className="flex-1 font-mono text-[11px] text-zinc-400 overflow-y-auto space-y-1 custom-scrollbar">
              <div className="opacity-50">------------------ INIT ------------------</div>
              <div><span className="text-zinc-600">[10:45:01]</span> System modules loaded</div>
              <div><span className="text-zinc-600">[10:45:02]</span> Establishing WebSocket link...</div>
              <div><span className="text-zinc-600">[10:45:02]</span> <span className="text-emerald-500">Connected to 192.168.1.105</span></div>
              <div><span className="text-zinc-600">[10:45:05]</span> <span className="text-yellow-500">WARN: Video feed signal lost</span></div>
              <div><span className="text-zinc-600">[10:45:06]</span> Retrying connection (1/3)...</div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
