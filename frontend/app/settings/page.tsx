"use client";
import { useState, useEffect } from 'react';

export default function SettingsPage() {
    const [dataset, setDataset] = useState<"ldraw" | "ldview" | "rebrickable">("ldraw");
    const [epochs, setEpochs] = useState(10);
    const [batchSize, setBatchSize] = useState(8);
    const [cameraType, setCameraType] = useState<"usb" | "csi" | "http" | "video_file">("usb");
    const [videoFile, setVideoFile] = useState<string | null>(null);
    const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
    const [videos, setVideos] = useState<Array<{filename: string, size: number, modified: number}>>([]);
    const [isUploading, setIsUploading] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [saveMessage, setSaveMessage] = useState<string | null>(null);

    const API_URL = "http://localhost:8000";

    const loadSettings = async () => {
        try {
            const res = await fetch(`${API_URL}/settings`, {
                signal: AbortSignal.timeout(3000)
            });
            if (!res.ok) {
                // Vision API offline - use defaults
                return;
            }
            const data = await res.json();

            if (data.dataset) setDataset(data.dataset);
            if (data.epochs) setEpochs(data.epochs);
            if (data.batch_size) setBatchSize(data.batch_size);
            if (data.camera_type) setCameraType(data.camera_type);
            if (data.video_file !== undefined) setVideoFile(data.video_file);
            if (data.video_playback_speed) setPlaybackSpeed(data.video_playback_speed);
        } catch (e) {
            // Silently fail - vision API is offline, use default settings
        }
    };

    const loadVideos = async () => {
        try {
            const res = await fetch(`${API_URL}/video/list`, {
                signal: AbortSignal.timeout(3000)
            });
            if (!res.ok) return;
            const data = await res.json();
            setVideos(data.videos || []);
        } catch (e) {
            // Silently fail
        }
    };

    const saveSettings = async () => {
        setIsSaving(true);
        setSaveMessage(null);

        try {
            const res = await fetch(`${API_URL}/settings`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset,
                    epochs,
                    batch_size: batchSize,
                    camera_type: cameraType,
                    video_file: videoFile,
                    video_playback_speed: playbackSpeed
                }),
                signal: AbortSignal.timeout(3000)
            });

            if (!res.ok) {
                setSaveMessage("Vision API is offline - settings will be saved when it comes online");
                return;
            }

            const data = await res.json();

            if (res.ok) {
                setSaveMessage("Settings saved successfully!");
                setTimeout(() => setSaveMessage(null), 3000);
            } else {
                setSaveMessage("Failed to save settings: " + data.detail);
            }
        } catch (e) {
            setSaveMessage("Vision API is offline - settings will be saved when it comes online");
        } finally {
            setIsSaving(false);
        }
    };

    const uploadVideo = async (file: File) => {
        setIsUploading(true);
        setSaveMessage(null);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const res = await fetch(`${API_URL}/video/upload`, {
                method: 'POST',
                body: formData,
                signal: AbortSignal.timeout(30000)
            });

            const data = await res.json();

            if (res.ok) {
                setSaveMessage(`Video "${data.filename}" uploaded successfully!`);
                await loadVideos();
                setVideoFile(data.filename);
                setTimeout(() => setSaveMessage(null), 3000);
            } else {
                setSaveMessage("Failed to upload video: " + data.detail);
            }
        } catch (e) {
            setSaveMessage("Failed to upload video: " + (e as Error).message);
        } finally {
            setIsUploading(false);
        }
    };

    const deleteVideo = async (filename: string) => {
        if (!confirm(`Delete video "${filename}"?`)) return;

        setSaveMessage(null);

        try {
            const res = await fetch(`${API_URL}/video/${filename}`, {
                method: 'DELETE',
                signal: AbortSignal.timeout(3000)
            });

            const data = await res.json();

            if (res.ok) {
                setSaveMessage(`Video "${filename}" deleted successfully!`);
                await loadVideos();
                if (videoFile === filename) {
                    setVideoFile(null);
                }
                setTimeout(() => setSaveMessage(null), 3000);
            } else {
                setSaveMessage("Failed to delete video: " + data.detail);
            }
        } catch (e) {
            setSaveMessage("Failed to delete video: " + (e as Error).message);
        }
    };

    const resetSettings = async () => {
        if (!confirm("Reset all settings to defaults?")) return;

        setIsSaving(true);
        setSaveMessage(null);

        try {
            const res = await fetch(`${API_URL}/settings/reset`, {
                method: 'POST',
                signal: AbortSignal.timeout(3000)
            });

            if (!res.ok) {
                setSaveMessage("Vision API is offline - cannot reset settings");
                return;
            }

            const data = await res.json();

            if (res.ok) {
                setSaveMessage("Settings reset to defaults!");
                await loadSettings();
                setTimeout(() => setSaveMessage(null), 3000);
            } else {
                setSaveMessage("Failed to reset settings: " + data.detail);
            }
        } catch (e) {
            setSaveMessage("Vision API is offline - cannot reset settings");
        } finally {
            setIsSaving(false);
        }
    };

    useEffect(() => {
        loadSettings();
        loadVideos();
    }, []);

    return (
        <div className="flex flex-col text-zinc-50 h-full">
            <main className="flex-1 p-6 max-w-[800px] mx-auto w-full">
                <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-6 flex flex-col gap-6 shadow-lg">

                    {/* Dataset Source */}
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
                                    className="w-4 h-4"
                                />
                                <div className="flex-1">
                                    <div className="text-sm font-medium text-zinc-200">Rebrickable CGI</div>
                                    <div className="text-xs text-zinc-500">On-the-fly synthesis</div>
                                </div>
                            </label>
                        </div>
                    </div>

                    <div className="h-px bg-zinc-800" />

                    {/* Training Parameters */}
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
                                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-2 text-sm text-zinc-200"
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
                                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-2 text-sm text-zinc-200"
                                />
                            </div>
                        </div>
                    </div>

                    <div className="h-px bg-zinc-800" />

                    {/* Camera Source */}
                    <div>
                        <h3 className="text-sm font-medium text-zinc-400 mb-3">CAMERA SOURCE</h3>
                        <div className="space-y-2">
                            <label className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${cameraType === 'usb' ? 'bg-blue-500/10 border-blue-500/50' : 'bg-zinc-950 border-zinc-800 hover:border-zinc-700'}`}>
                                <input
                                    type="radio"
                                    name="camera"
                                    value="usb"
                                    checked={cameraType === 'usb'}
                                    onChange={(e) => setCameraType(e.target.value as any)}
                                    className="w-4 h-4"
                                />
                                <div className="flex-1">
                                    <div className="text-sm font-medium text-zinc-200">USB Camera</div>
                                    <div className="text-xs text-zinc-500">Direct USB webcam access</div>
                                </div>
                            </label>

                            <label className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${cameraType === 'csi' ? 'bg-blue-500/10 border-blue-500/50' : 'bg-zinc-950 border-zinc-800 hover:border-zinc-700'}`}>
                                <input
                                    type="radio"
                                    name="camera"
                                    value="csi"
                                    checked={cameraType === 'csi'}
                                    onChange={(e) => setCameraType(e.target.value as any)}
                                    className="w-4 h-4"
                                />
                                <div className="flex-1">
                                    <div className="text-sm font-medium text-zinc-200">CSI Camera</div>
                                    <div className="text-xs text-zinc-500">Raspberry Pi camera module</div>
                                </div>
                            </label>

                            <label className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${cameraType === 'http' ? 'bg-blue-500/10 border-blue-500/50' : 'bg-zinc-950 border-zinc-800 hover:border-zinc-700'}`}>
                                <input
                                    type="radio"
                                    name="camera"
                                    value="http"
                                    checked={cameraType === 'http'}
                                    onChange={(e) => setCameraType(e.target.value as any)}
                                    className="w-4 h-4"
                                />
                                <div className="flex-1">
                                    <div className="text-sm font-medium text-zinc-200">HTTP Camera</div>
                                    <div className="text-xs text-zinc-500">Remote webcam server</div>
                                </div>
                            </label>

                            <label className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${cameraType === 'video_file' ? 'bg-blue-500/10 border-blue-500/50' : 'bg-zinc-950 border-zinc-800 hover:border-zinc-700'}`}>
                                <input
                                    type="radio"
                                    name="camera"
                                    value="video_file"
                                    checked={cameraType === 'video_file'}
                                    onChange={(e) => setCameraType(e.target.value as any)}
                                    className="w-4 h-4"
                                />
                                <div className="flex-1">
                                    <div className="text-sm font-medium text-zinc-200">Video File</div>
                                    <div className="text-xs text-zinc-500">Pre-recorded MP4 video loop</div>
                                </div>
                            </label>
                        </div>
                    </div>

                    {/* Video File Settings */}
                    {cameraType === 'video_file' && (
                        <>
                            <div className="h-px bg-zinc-800" />

                            <div>
                                <h3 className="text-sm font-medium text-zinc-400 mb-3">VIDEO FILE SETTINGS</h3>

                                {/* Upload Video */}
                                <div className="mb-4">
                                    <label className="text-xs text-zinc-500 block mb-2">Upload Video</label>
                                    <input
                                        type="file"
                                        accept="video/mp4,video/avi,video/mov,video/mkv,video/webm"
                                        onChange={(e) => {
                                            const file = e.target.files?.[0];
                                            if (file) uploadVideo(file);
                                        }}
                                        disabled={isUploading}
                                        className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-2 text-sm text-zinc-200 file:mr-4 file:py-1 file:px-3 file:rounded file:border-0 file:text-sm file:bg-emerald-600 file:text-white hover:file:bg-emerald-500 disabled:opacity-50"
                                    />
                                    {isUploading && (
                                        <div className="text-xs text-zinc-500 mt-1">Uploading...</div>
                                    )}
                                </div>

                                {/* Video List */}
                                {videos.length > 0 && (
                                    <div className="mb-4">
                                        <label className="text-xs text-zinc-500 block mb-2">Available Videos</label>
                                        <div className="space-y-2">
                                            {videos.map((video) => (
                                                <div
                                                    key={video.filename}
                                                    className={`flex items-center gap-3 p-3 rounded-lg border transition-all ${videoFile === video.filename ? 'bg-blue-500/10 border-blue-500/50' : 'bg-zinc-950 border-zinc-800'}`}
                                                >
                                                    <input
                                                        type="radio"
                                                        name="video"
                                                        value={video.filename}
                                                        checked={videoFile === video.filename}
                                                        onChange={(e) => setVideoFile(e.target.value)}
                                                        className="w-4 h-4"
                                                    />
                                                    <div className="flex-1 min-w-0">
                                                        <div className="text-sm font-medium text-zinc-200 truncate">{video.filename}</div>
                                                        <div className="text-xs text-zinc-500">
                                                            {(video.size / 1024 / 1024).toFixed(2)} MB
                                                        </div>
                                                    </div>
                                                    <button
                                                        onClick={() => deleteVideo(video.filename)}
                                                        className="px-3 py-1 bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded text-xs transition-all"
                                                    >
                                                        Delete
                                                    </button>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Playback Speed */}
                                <div>
                                    <label className="text-xs text-zinc-500 block mb-1">
                                        Playback Speed: {playbackSpeed.toFixed(1)}x
                                    </label>
                                    <input
                                        type="range"
                                        min="0.1"
                                        max="5"
                                        step="0.1"
                                        value={playbackSpeed}
                                        onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
                                        className="w-full"
                                    />
                                    <div className="flex justify-between text-xs text-zinc-600 mt-1">
                                        <span>0.1x</span>
                                        <span>1x</span>
                                        <span>5x</span>
                                    </div>
                                </div>
                            </div>
                        </>
                    )}

                    <div className="h-px bg-zinc-800" />

                    {/* Save/Reset Buttons */}
                    <div className="flex gap-3">
                        <button
                            onClick={saveSettings}
                            disabled={isSaving}
                            className="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white py-3 rounded-lg text-sm font-bold transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {isSaving ? 'Saving...' : 'Save Settings'}
                        </button>
                        <button
                            onClick={resetSettings}
                            disabled={isSaving}
                            className="px-6 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 py-3 rounded-lg text-sm font-bold transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            Reset
                        </button>
                    </div>

                    {/* Save Message */}
                    {saveMessage && (
                        <div className={`p-3 rounded-lg text-sm ${saveMessage.includes('successfully') ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/50' : 'bg-red-500/10 text-red-400 border border-red-500/50'}`}>
                            {saveMessage}
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}
