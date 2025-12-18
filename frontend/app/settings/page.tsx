"use client";
import { useState, useEffect } from 'react';

type Section = 'training' | 'camera' | 'detector';

export default function SettingsPage() {
    const [activeSection, setActiveSection] = useState<Section>('training');

    // Training settings
    const [dataset, setDataset] = useState<"ldraw" | "ldview" | "rebrickable">("ldraw");
    const [epochs, setEpochs] = useState(10);
    const [batchSize, setBatchSize] = useState(8);

    // Camera settings
    const [cameraType, setCameraType] = useState<"usb" | "csi" | "http" | "video_file">("usb");
    const [videoFile, setVideoFile] = useState<string | null>(null);
    const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
    const [videos, setVideos] = useState<Array<{filename: string, size: number, modified: number}>>([]);

    // Detector settings
    const [minAreaPercent, setMinAreaPercent] = useState(0.1);
    const [maxAreaPercent, setMaxAreaPercent] = useState(15);
    const [diffThreshold, setDiffThreshold] = useState(30);
    const [centerTolerance, setCenterTolerance] = useState(0.15);
    const [edgeMargin, setEdgeMargin] = useState(20);

    // UI state
    const [isUploading, setIsUploading] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [saveMessage, setSaveMessage] = useState<string | null>(null);
    const [debugImage, setDebugImage] = useState<string | null>(null);
    const [detectionInfo, setDetectionInfo] = useState<any>(null);

    const API_URL = "http://localhost:8000";

    const loadSettings = async () => {
        try {
            const res = await fetch(`${API_URL}/settings`, {
                signal: AbortSignal.timeout(3000)
            });
            if (!res.ok) return;
            const data = await res.json();

            if (data.dataset) setDataset(data.dataset);
            if (data.epochs) setEpochs(data.epochs);
            if (data.batch_size) setBatchSize(data.batch_size);
            if (data.camera_type) setCameraType(data.camera_type);
            if (data.video_file !== undefined) setVideoFile(data.video_file);
            if (data.video_playback_speed) setPlaybackSpeed(data.video_playback_speed);
        } catch (e) {
            // Silently fail
        }
    };

    const loadDetectorParams = async () => {
        try {
            const res = await fetch(`${API_URL}/inference/calibration_status`, {
                signal: AbortSignal.timeout(3000)
            });
            if (!res.ok) return;
            const data = await res.json();

            if (data.detector_info?.params) {
                const params = data.detector_info.params;
                setMinAreaPercent((params.min_area_percent || 0.001) * 100);
                setMaxAreaPercent((params.max_area_percent || 0.15) * 100);
                setDiffThreshold(params.diff_threshold || 30);
                setCenterTolerance(params.center_tolerance || 0.15);
                setEdgeMargin(params.edge_margin || 20);
            }
        } catch (e) {
            // Silently fail
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

            setSaveMessage("Settings saved successfully!");
            setTimeout(() => setSaveMessage(null), 3000);
        } catch (e) {
            setSaveMessage("Vision API is offline - settings will be saved when it comes online");
        } finally {
            setIsSaving(false);
        }
    };

    const saveDetectorParams = async () => {
        setIsSaving(true);
        setSaveMessage(null);

        try {
            const url = `${API_URL}/inference/detector/params?min_area_percent=${minAreaPercent / 100}&max_area_percent=${maxAreaPercent / 100}&diff_threshold=${diffThreshold}&center_tolerance=${centerTolerance}&edge_margin=${edgeMargin}`;
            const res = await fetch(url, {
                method: 'POST',
                signal: AbortSignal.timeout(3000)
            });

            if (!res.ok) {
                setSaveMessage("Failed to update detector parameters");
                return;
            }

            setSaveMessage("Detector parameters updated successfully!");
            setTimeout(() => setSaveMessage(null), 3000);
        } catch (e) {
            setSaveMessage("Vision API is offline");
        } finally {
            setIsSaving(false);
        }
    };

    const calibrateBackground = async () => {
        setSaveMessage(null);

        try {
            const res = await fetch(`${API_URL}/inference/calibrate`, {
                method: 'POST',
                signal: AbortSignal.timeout(3000)
            });

            const data = await res.json();

            if (res.ok) {
                setSaveMessage("Background calibrated successfully!");
                setTimeout(() => setSaveMessage(null), 3000);
            } else {
                setSaveMessage("Failed to calibrate: " + data.detail);
            }
        } catch (e) {
            setSaveMessage("Vision API is offline or inference not running");
        }
    };

    const refreshDebugView = async () => {
        setSaveMessage(null);

        try {
            const res = await fetch(`${API_URL}/inference/detector/debug`, {
                signal: AbortSignal.timeout(5000)
            });

            const data = await res.json();

            if (res.ok) {
                setDebugImage('data:image/jpeg;base64,' + data.debug_image);
                setDetectionInfo({
                    status: data.status,
                    bboxCount: data.bounding_boxes.length,
                    centerDetected: data.center_detected
                });
            } else {
                setSaveMessage("Debug view error: " + data.detail);
            }
        } catch (e) {
            setSaveMessage("Could not fetch debug view");
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

            setSaveMessage("Settings reset to defaults!");
            await loadSettings();
            setTimeout(() => setSaveMessage(null), 3000);
        } catch (e) {
            setSaveMessage("Vision API is offline - cannot reset settings");
        } finally {
            setIsSaving(false);
        }
    };

    const resetDetectorParams = () => {
        setMinAreaPercent(0.1);
        setMaxAreaPercent(15);
        setDiffThreshold(30);
        setCenterTolerance(0.15);
        setEdgeMargin(20);
    };

    useEffect(() => {
        loadSettings();
        loadVideos();
        loadDetectorParams();
    }, []);

    // Auto-refresh debug view when on detector tab
    useEffect(() => {
        if (activeSection === 'detector') {
            const interval = setInterval(() => {
                if (debugImage) refreshDebugView();
            }, 500);
            return () => clearInterval(interval);
        }
    }, [activeSection, debugImage]);

    const navItems = [
        { id: 'training' as Section, label: 'Training', icon: '🎓' },
        { id: 'camera' as Section, label: 'Camera', icon: '📷' },
        { id: 'detector' as Section, label: 'Detector', icon: '🔍' },
    ];

    return (
        <div className="flex h-screen bg-zinc-950 text-zinc-50">
            {/* Left Sidebar Navigation */}
            <aside className="w-64 bg-zinc-900 border-r border-zinc-800 flex flex-col">
                <div className="p-6">
                    <h1 className="text-xl font-bold text-zinc-100">Settings</h1>
                    <p className="text-xs text-zinc-500 mt-1">Configure system parameters</p>
                </div>

                <nav className="flex-1 px-3">
                    {navItems.map((item) => (
                        <button
                            key={item.id}
                            onClick={() => setActiveSection(item.id)}
                            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg mb-1 transition-all text-left ${
                                activeSection === item.id
                                    ? 'bg-emerald-600 text-white shadow-lg'
                                    : 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
                            }`}
                        >
                            <span className="text-xl">{item.icon}</span>
                            <span className="font-medium">{item.label}</span>
                        </button>
                    ))}
                </nav>

                <div className="p-4 border-t border-zinc-800">
                    <div className="text-xs text-zinc-600">
                        LEGO Sorter v1.0
                    </div>
                </div>
            </aside>

            {/* Main Content Area */}
            <main className="flex-1 overflow-y-auto">
                <div className="max-w-4xl mx-auto p-8">
                    {/* Training Section */}
                    {activeSection === 'training' && (
                        <div className="space-y-6">
                            <div>
                                <h2 className="text-2xl font-bold text-zinc-100 mb-2">Training Settings</h2>
                                <p className="text-sm text-zinc-500">Configure dataset and training parameters</p>
                            </div>

                            <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-6">
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

                            <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-6">
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

                            <div className="flex gap-3">
                                <button
                                    onClick={saveSettings}
                                    disabled={isSaving}
                                    className="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white py-3 rounded-lg text-sm font-bold transition-all shadow-lg disabled:opacity-50"
                                >
                                    {isSaving ? 'Saving...' : 'Save Settings'}
                                </button>
                                <button
                                    onClick={resetSettings}
                                    disabled={isSaving}
                                    className="px-6 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 py-3 rounded-lg text-sm font-bold transition-all disabled:opacity-50"
                                >
                                    Reset
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Camera Section */}
                    {activeSection === 'camera' && (
                        <div className="space-y-6">
                            <div>
                                <h2 className="text-2xl font-bold text-zinc-100 mb-2">Camera Settings</h2>
                                <p className="text-sm text-zinc-500">Configure camera source and video input</p>
                            </div>

                            <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-6">
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

                            {cameraType === 'video_file' && (
                                <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-6">
                                    <h3 className="text-sm font-medium text-zinc-400 mb-3">VIDEO FILE SETTINGS</h3>

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
                            )}

                            <div className="flex gap-3">
                                <button
                                    onClick={saveSettings}
                                    disabled={isSaving}
                                    className="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white py-3 rounded-lg text-sm font-bold transition-all shadow-lg disabled:opacity-50"
                                >
                                    {isSaving ? 'Saving...' : 'Save Settings'}
                                </button>
                                <button
                                    onClick={resetSettings}
                                    disabled={isSaving}
                                    className="px-6 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 py-3 rounded-lg text-sm font-bold transition-all disabled:opacity-50"
                                >
                                    Reset
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Detector Section */}
                    {activeSection === 'detector' && (
                        <div className="space-y-6">
                            <div>
                                <h2 className="text-2xl font-bold text-zinc-100 mb-2">Detector Tuning</h2>
                                <p className="text-sm text-zinc-500">Calibrate and adjust object detection parameters</p>
                            </div>

                            <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-6">
                                <h3 className="text-sm font-medium text-zinc-400 mb-3">CALIBRATION</h3>
                                <p className="text-sm text-zinc-500 mb-4">
                                    Position your camera to show ONLY the background (no LEGO bricks), then click calibrate.
                                </p>
                                <button
                                    onClick={calibrateBackground}
                                    className="w-full bg-blue-600 hover:bg-blue-500 text-white py-3 rounded-lg text-sm font-bold transition-all shadow-lg"
                                >
                                    📷 Calibrate Background
                                </button>
                            </div>

                            <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-6">
                                <h3 className="text-sm font-medium text-zinc-400 mb-4">DETECTION PARAMETERS</h3>

                                <div className="space-y-5">
                                    <div>
                                        <label className="flex justify-between text-sm text-zinc-300 mb-2">
                                            <span>Minimum Area (% of frame)</span>
                                            <span className="font-mono text-emerald-400">{minAreaPercent.toFixed(1)}%</span>
                                        </label>
                                        <input
                                            type="range"
                                            min="0.01"
                                            max="5"
                                            step="0.1"
                                            value={minAreaPercent}
                                            onChange={(e) => setMinAreaPercent(parseFloat(e.target.value))}
                                            className="w-full"
                                        />
                                        <p className="text-xs text-zinc-600 mt-1">
                                            Objects smaller than this are ignored. 0.1% = good default for noise filtering.
                                        </p>
                                    </div>

                                    <div>
                                        <label className="flex justify-between text-sm text-zinc-300 mb-2">
                                            <span>Maximum Area (% of frame)</span>
                                            <span className="font-mono text-emerald-400">{maxAreaPercent}%</span>
                                        </label>
                                        <input
                                            type="range"
                                            min="1"
                                            max="100"
                                            step="1"
                                            value={maxAreaPercent}
                                            onChange={(e) => setMaxAreaPercent(parseInt(e.target.value))}
                                            className="w-full"
                                        />
                                        <p className="text-xs text-zinc-600 mt-1">
                                            Objects larger than this % of the frame are ignored. 15-30% = good range.
                                        </p>
                                    </div>

                                    <div>
                                        <label className="flex justify-between text-sm text-zinc-300 mb-2">
                                            <span>Difference Threshold</span>
                                            <span className="font-mono text-emerald-400">{diffThreshold}</span>
                                        </label>
                                        <input
                                            type="range"
                                            min="5"
                                            max="100"
                                            step="5"
                                            value={diffThreshold}
                                            onChange={(e) => setDiffThreshold(parseInt(e.target.value))}
                                            className="w-full"
                                        />
                                        <p className="text-xs text-zinc-600 mt-1">
                                            Lower = more sensitive. Use lower values for subtle color differences.
                                        </p>
                                    </div>

                                    <div>
                                        <label className="flex justify-between text-sm text-zinc-300 mb-2">
                                            <span>Center Tolerance</span>
                                            <span className="font-mono text-emerald-400">{centerTolerance.toFixed(2)}</span>
                                        </label>
                                        <input
                                            type="range"
                                            min="0.05"
                                            max="0.5"
                                            step="0.05"
                                            value={centerTolerance}
                                            onChange={(e) => setCenterTolerance(parseFloat(e.target.value))}
                                            className="w-full"
                                        />
                                        <p className="text-xs text-zinc-600 mt-1">
                                            Fraction of frame center considered "centered" (0.15 = ±15%)
                                        </p>
                                    </div>

                                    <div>
                                        <label className="flex justify-between text-sm text-zinc-300 mb-2">
                                            <span>Edge Margin (pixels)</span>
                                            <span className="font-mono text-emerald-400">{edgeMargin}</span>
                                        </label>
                                        <input
                                            type="range"
                                            min="5"
                                            max="100"
                                            step="5"
                                            value={edgeMargin}
                                            onChange={(e) => setEdgeMargin(parseInt(e.target.value))}
                                            className="w-full"
                                        />
                                        <p className="text-xs text-zinc-600 mt-1">
                                            Distance from edge where object is considered touching.
                                        </p>
                                    </div>
                                </div>
                            </div>

                            <div className="flex gap-3">
                                <button
                                    onClick={saveDetectorParams}
                                    disabled={isSaving}
                                    className="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white py-3 rounded-lg text-sm font-bold transition-all shadow-lg disabled:opacity-50"
                                >
                                    {isSaving ? 'Applying...' : 'Apply Parameters'}
                                </button>
                                <button
                                    onClick={resetDetectorParams}
                                    className="px-6 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 py-3 rounded-lg text-sm font-bold transition-all"
                                >
                                    Reset
                                </button>
                            </div>

                            <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-6">
                                <div className="flex items-center justify-between mb-4">
                                    <h3 className="text-sm font-medium text-zinc-400">DEBUG VIEW</h3>
                                    <button
                                        onClick={refreshDebugView}
                                        className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg text-sm font-medium transition-all"
                                    >
                                        🔄 Refresh
                                    </button>
                                </div>

                                {detectionInfo && (
                                    <div className="mb-4 p-4 bg-zinc-950 rounded-lg border border-zinc-800">
                                        <div className="grid grid-cols-3 gap-4 text-sm">
                                            <div>
                                                <div className="text-zinc-500 text-xs">Status</div>
                                                <div className="text-zinc-200 font-medium">{detectionInfo.status}</div>
                                            </div>
                                            <div>
                                                <div className="text-zinc-500 text-xs">Bounding Boxes</div>
                                                <div className="text-zinc-200 font-medium">{detectionInfo.bboxCount}</div>
                                            </div>
                                            <div>
                                                <div className="text-zinc-500 text-xs">Center Detected</div>
                                                <div className="text-zinc-200 font-medium">
                                                    {detectionInfo.centerDetected ? '✅ Yes' : '❌ No'}
                                                </div>
                                            </div>
                                        </div>
                                        {detectionInfo.bboxCount === 0 && (
                                            <div className="mt-3 text-xs text-yellow-500">
                                                ⚠️ No objects detected. Try adjusting the parameters above.
                                            </div>
                                        )}
                                    </div>
                                )}

                                {debugImage ? (
                                    <img
                                        src={debugImage}
                                        alt="Debug visualization"
                                        className="w-full rounded-lg border border-zinc-800"
                                    />
                                ) : (
                                    <div className="w-full h-64 bg-zinc-950 rounded-lg border border-zinc-800 flex items-center justify-center text-zinc-600">
                                        Click Refresh to view detection debug output
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Save Message */}
                    {saveMessage && (
                        <div className={`p-4 rounded-lg text-sm ${saveMessage.includes('successfully') || saveMessage.includes('Calibrated') ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/50' : 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/50'}`}>
                            {saveMessage}
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}
