"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { toast } from "sonner";
import Recorder from "@/app/components/Recorder"
import TextDisplay from "@/app/components/TextDisplay"
import ControlDock from "@/app/components/ControlDock"

import * as HandsModule from "@mediapipe/hands";

const MIN_DETECTION_CONFIDENCE = 0.5;
const MIN_TRACKING_CONFIDENCE = 0.5;

type RecorderState = "default" | "initializing" | "recording" | "ready" | "loading" | "error" | "show-text";

export default function GestureRecorder() {
    const [state, setState] = useState<RecorderState>("default");
    const [translatedText, setTranslatedText] = useState<string>("");
    const [error, setError] = useState<string>("");
    const [highlightedIndex, setHighlightedIndex] = useState<number>(-1);
    const [isPlaying, setIsPlaying] = useState(false);

    const [handsDetected, setHandsDetected] = useState<number>(0);
    const [keypointsCount, setKeypointsCount] = useState<number>(0);

    const videoRef = useRef<HTMLVideoElement>(null);
    const streamRef = useRef<MediaStream | null>(null);

    const handsRef = useRef<HandsModule.Hands | null>(null);
    const animationFrameIdRef = useRef<number | null>(null);
    const recordedKeypointsRef = useRef<number[][]>([]);
    
    const textContainerRef = useRef<HTMLDivElement>(null);
    const wordsRef = useRef<string[]>([]);

    const processResults = useCallback((results: HandsModule.Results) => {
        let numHandsDetected = 0;
        const NUM_HAND_FEATURES = 21 * 3;

        let left_hand_kps = Array(NUM_HAND_FEATURES).fill(0.0);
        let right_hand_kps = Array(NUM_HAND_FEATURES).fill(0.0);

        if (results.multiHandLandmarks && results.multiHandedness) {
            numHandsDetected = results.multiHandLandmarks.length;
            for (let i = 0; i < numHandsDetected; i++) {
                const landmarks = results.multiHandLandmarks[i];
                const handedness = (results.multiHandedness[i] as any).label;
                const wristLm = landmarks[0];
                if (wristLm) {
                    const { x: wristX, y: wristY, z: wristZ } = wristLm;
                    const normalized = landmarks.flatMap(lm => [lm.x - wristX, lm.y - wristY, lm.z - wristZ]);
                    if (handedness === "Left") left_hand_kps = normalized;
                    else if (handedness === "Right") right_hand_kps = normalized;
                }
            }
        }
        
        setHandsDetected(numHandsDetected);

        if (state === "recording" && numHandsDetected > 0) {
            const combinedKeypoints = [...left_hand_kps, ...right_hand_kps];
            recordedKeypointsRef.current.push(combinedKeypoints);
            setKeypointsCount(recordedKeypointsRef.current.length);
        }
    }, [state]);

    const animationLoop = useCallback(async () => {
        if (!videoRef.current || videoRef.current.paused || videoRef.current.ended) {
            animationFrameIdRef.current = null;
            return;
        }
        if (handsRef.current) {
            await handsRef.current.send({ image: videoRef.current });
        }
        animationFrameIdRef.current = requestAnimationFrame(animationLoop);
    }, []);

    useEffect(() => {
        const initializeMediaPipe = async () => {
            try {
                handsRef.current = new HandsModule.Hands({ 
                    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
                });

                handsRef.current.setOptions({
                    maxNumHands: 2,
                    modelComplexity: 1,
                    minDetectionConfidence: MIN_DETECTION_CONFIDENCE,
                    minTrackingConfidence: MIN_TRACKING_CONFIDENCE
                });
                
                handsRef.current.onResults(processResults);
                console.log('MediaPipe Hands model initialized successfully.');
            } catch (error) {
                console.error('Failed to initialize MediaPipe Hands model:', error);
                toast.error('Failed to initialize detection model.');
            }
        };
        initializeMediaPipe();
    }, [processResults]);

    const startRecording = useCallback(async () => {
        setError(""); setTranslatedText(""); setHighlightedIndex(-1); setIsPlaying(false);
        setHandsDetected(0); setKeypointsCount(0);
        recordedKeypointsRef.current = [];
        setState("initializing");

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } }, audio: false });
            streamRef.current = stream;
            if (!videoRef.current) throw new Error("Video element not available.");
            videoRef.current.srcObject = stream;

            await new Promise<void>(resolve => { if(videoRef.current) videoRef.current.onloadeddata = () => resolve() });
            await videoRef.current.play();

            if (!handsRef.current) throw new Error('MediaPipe Hands model not ready.');
            
            animationFrameIdRef.current = requestAnimationFrame(animationLoop);

            setState("recording");
            toast.info("Recording started!");
        } catch (err) {
            const errorMessage = (err instanceof Error ? err.message : String(err));
            console.error('Start recording error:', err); setError(errorMessage); setState("error");
            toast.error(`Error: ${errorMessage}`);
        }
    }, [animationLoop]);

    const stopRecording = useCallback(() => {
        if (animationFrameIdRef.current) {
            cancelAnimationFrame(animationFrameIdRef.current);
            animationFrameIdRef.current = null;
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
        setState("ready");
        console.log(`Recording stopped. Total keypoint frames: ${recordedKeypointsRef.current.length}`);
    }, []);

    const sendRecording = useCallback(async () => {
        if (recordedKeypointsRef.current.length === 0) {
            toast.error("No keypoints to send."); return;
        }
        setState("loading");
        try {
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080/predict_gesture";
            const res = await fetch(apiUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ keypoints: recordedKeypointsRef.current }),
            });
            if (!res.ok) {
                const errorData = await res.json().catch(() => ({ message: "Server error" }));
                throw new Error(errorData.message || "Server responded with an error.");
            }
            const data = await res.json();
            const newText = data.translatedText || "No translation.";
            setTranslatedText(newText); wordsRef.current = newText.split(/\s+/).filter(Boolean);
            setState("show-text");
            toast.success("Gesture recognized!");
        } catch (err: any) {
            console.error("Error sending keypoints for prediction:", err);
            setError(err.message);
            setState("error");
            toast.error(`Prediction failed: ${err.message}`);
        }
    }, []);

    const playTranslatedText = useCallback(() => {
        if (!translatedText || typeof window.speechSynthesis === 'undefined') return;
        if (isPlaying) { window.speechSynthesis.cancel(); setIsPlaying(false); setHighlightedIndex(-1); return; }
        setIsPlaying(true); setHighlightedIndex(0);
        const utterance = new SpeechSynthesisUtterance(translatedText);
        utterance.rate = 1.0; utterance.pitch = 1.0;
        utterance.onboundary = (event) => {
            if (event.name === "word" && wordsRef.current && event.charIndex !== undefined) {
                let currentWordIndex = 0; let charCount = 0;
                for (let i = 0; i < wordsRef.current.length; i++) {
                    charCount += wordsRef.current[i].length + 1;
                    if (event.charIndex < charCount) { currentWordIndex = i; break; }
                }
                setHighlightedIndex(currentWordIndex);
            }
        };
        utterance.onend = () => { setIsPlaying(false); setHighlightedIndex(-1); };
        utterance.onerror = (event) => {
            console.error("SpeechSynthesisUtterance error:", event.error);
            setIsPlaying(false); setHighlightedIndex(-1); toast.error("Text-to-speech failed.");
        };
        const speakWhenVoicesReady = () => {
            const voices = window.speechSynthesis.getVoices();
            if (voices.length > 0) {
                const bestVoice = voices.find(voice => voice.lang.startsWith('en'));
                if (bestVoice) utterance.voice = bestVoice;
                window.speechSynthesis.speak(utterance);
            } else { console.warn("No voices loaded. Speaking with default."); window.speechSynthesis.speak(utterance); }
        };
        if (typeof window !== 'undefined' && window.speechSynthesis.getVoices().length === 0 && 'onvoiceschanged' in window.speechSynthesis) {
            window.speechSynthesis.onvoiceschanged = () => speakWhenVoicesReady();
        } else if (typeof window !== 'undefined') { 
            speakWhenVoicesReady(); 
        }
    }, [isPlaying, translatedText]);

    const copyToClipboard = useCallback(() => {
        if (!translatedText || typeof navigator === 'undefined') return;
        navigator.clipboard.writeText(translatedText)
            .then(() => toast.success("Copied to clipboard"))
            .catch((err) => { console.error("Failed to copy text:", err); toast.error("Failed to copy"); });
    }, [translatedText]);

    const resetRecorder = useCallback(() => {
        if (typeof window.speechSynthesis !== 'undefined') window.speechSynthesis.cancel();
        stopRecording();
        setTranslatedText(""); setError(""); setHighlightedIndex(-1); setIsPlaying(false);
        setHandsDetected(0); setKeypointsCount(0);
        setState("default");
    }, [stopRecording]);

    return (
        <div className="relative flex flex-col w-full h-[calc(100vh-2rem)] max-h-screen overflow-hidden">
            <div className="flex-1 flex items-center justify-center p-4 overflow-hidden">
                {(state === "recording" || state === "initializing") && (
                    <div className="relative w-full max-w-7xl aspect-video rounded-lg shadow-lg bg-black">
                        <video ref={videoRef} className="w-full h-full object-cover transform -scale-x-100 rounded-lg" muted autoPlay playsInline />
                        <div className="absolute top-4 left-4 bg-black bg-opacity-80 text-white p-3 rounded text-sm z-50 font-mono">
                            <div>Status: <span className="text-green-400">{state}</span></div>
                            <div>Hands: <span className="text-blue-400">{handsDetected}</span></div>
                            <div>Frames: <span className="text-yellow-400">{keypointsCount}</span></div>
                        </div>
                    </div>
                )}
                {state === "show-text" && (
                    <TextDisplay ref={textContainerRef} translatedText={translatedText} highlightedIndex={highlightedIndex} words={wordsRef.current} onCopyToClipboard={copyToClipboard} />
                )}
                {state === "error" && error && (
                    <div className="flex flex-col items-center justify-center mt-4 text-center">
                        <p className="text-red-600 mb-4">{error}</p>

                        <button onClick={resetRecorder} className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded-md text-sm">Reset</button>
                    </div>
                )}
                {(state === "default" || state === "ready" || state === "loading") && (
                    <Recorder state={state} onReset={resetRecorder} onRetry={resetRecorder} />
                )}
            </div>
            <ControlDock state={state} isTextAvailable={!!translatedText} isPlaying={isPlaying} onStartRecording={startRecording} onStopRecording={stopRecording} onResetRecorder={resetRecorder} onSendRecording={sendRecording} onPlayTranslatedText={playTranslatedText} />
        </div>
    )
}