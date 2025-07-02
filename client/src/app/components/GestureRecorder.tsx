"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { toast } from "sonner";
import Recorder from "@/app/components/Recorder"
import TextDisplay from "@/app/components/TextDisplay"
import ControlDock from "@/app/components/ControlDock"

import * as HolisticModule from "@mediapipe/holistic";

const MIN_DETECTION_CONFIDENCE = 0.5;
const MIN_TRACKING_CONFIDENCE = 0.5;
const POSE_LANDMARKS_TO_EXTRACT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

type RecorderState = "default" | "initializing" | "recording" | "ready" | "loading" | "error" | "show-text";

export default function GestureRecorder() {
    const [state, setState] = useState<RecorderState>("default");
    const [translatedText, setTranslatedText] = useState<string>("");
    const [error, setError] = useState<string>("");
    const [highlightedIndex, setHighlightedIndex] = useState<number>(-1);
    const [isPlaying, setIsPlaying] = useState(false);

    const [poseDetected, setPoseDetected] = useState(false);
    const [handsDetected, setHandsDetected] = useState<number>(0);
    const [keypointsCount, setKeypointsCount] = useState<number>(0);

    const videoRef = useRef<HTMLVideoElement>(null);
    const streamRef = useRef<MediaStream | null>(null);

    const holisticRef = useRef<HolisticModule.Holistic | null>(null);
    const animationFrameIdRef = useRef<number | null>(null);
    const recordedKeypointsRef = useRef<number[][]>([]);
    
    const textContainerRef = useRef<HTMLDivElement>(null);
    const wordsRef = useRef<string[]>([]);

    const processResults = useCallback((results: HolisticModule.Results) => {
        let isPoseDetected = false, numHandsDetected = 0;

        const NUM_POSE_FEATURES = POSE_LANDMARKS_TO_EXTRACT.length * 3;
        const NUM_HAND_FEATURES = 21 * 3;

        let pose_kps = Array(NUM_POSE_FEATURES).fill(0.0);
        let left_hand_kps = Array(NUM_HAND_FEATURES).fill(0.0);
        let right_hand_kps = Array(NUM_HAND_FEATURES).fill(0.0);

        if (results.poseLandmarks) {
            isPoseDetected = true;
            const refPoint = results.poseLandmarks[HolisticModule.POSE_LANDMARKS.NOSE];
            if(refPoint) {
                const { x: refX, y: refY, z: refZ } = refPoint;
                pose_kps = POSE_LANDMARKS_TO_EXTRACT.flatMap(i => {
                    const lm = results.poseLandmarks[i];
                    return lm ? [lm.x - refX, lm.y - refY, lm.z - refZ] : [0, 0, 0];
                });
            }
        }
        
        const handLandmarks = [results.leftHandLandmarks, results.rightHandLandmarks];
        const handedness = ["Left", "Right"];
        handLandmarks.forEach((landmarks, i) => {
            if (landmarks) {
                numHandsDetected++;
                const wristLm = landmarks[0];
                if (wristLm) {
                    const { x: wristX, y: wristY, z: wristZ } = wristLm;
                    const normalized = landmarks.flatMap(lm => [lm.x - wristX, lm.y - wristY, lm.z - wristZ]);
                    if (handedness[i] === "Left") left_hand_kps = normalized;
                    else right_hand_kps = normalized;
                }
            }
        });
        
        setPoseDetected(isPoseDetected);
        setHandsDetected(numHandsDetected);

        if (state === "recording" && (isPoseDetected || numHandsDetected > 0)) {
            const combinedKeypoints = [...pose_kps, ...left_hand_kps, ...right_hand_kps];
            recordedKeypointsRef.current.push(combinedKeypoints);
            setKeypointsCount(recordedKeypointsRef.current.length);
        }
    }, [state]);

    const animationLoop = useCallback(async () => {
        if (!videoRef.current || videoRef.current.paused || videoRef.current.ended) {
            animationFrameIdRef.current = null;
            return;
        }
        if (holisticRef.current) {
            await holisticRef.current.send({ image: videoRef.current });
        }
        animationFrameIdRef.current = requestAnimationFrame(animationLoop);
    }, []);

    useEffect(() => {
        const initializeMediaPipe = async () => {
            try {
                holisticRef.current = new HolisticModule.Holistic({ 
                    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
                });

                holisticRef.current.setOptions({
                    modelComplexity: 1,
                    smoothLandmarks: true,
                    minDetectionConfidence: MIN_DETECTION_CONFIDENCE,
                    minTrackingConfidence: MIN_TRACKING_CONFIDENCE
                });
                
                holisticRef.current.onResults(processResults);
                console.log('MediaPipe Holistic model initialized successfully.');
            } catch (error) {
                console.error('Failed to initialize MediaPipe Holistic model:', error);
                toast.error('Failed to initialize detection model.');
            }
        };
        initializeMediaPipe();
    }, [processResults]);

    const startRecording = useCallback(async () => {
        setError(""); setTranslatedText(""); setHighlightedIndex(-1); setIsPlaying(false);
        setPoseDetected(false); setHandsDetected(0); setKeypointsCount(0);
        recordedKeypointsRef.current = [];
        setState("initializing");

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } }, audio: false });
            streamRef.current = stream;
            if (!videoRef.current) throw new Error("Video element not available.");
            videoRef.current.srcObject = stream;

            await new Promise<void>(resolve => { if(videoRef.current) videoRef.current.onloadeddata = () => resolve() });
            await videoRef.current.play();

            if (!holisticRef.current) throw new Error('MediaPipe Holistic model not ready.');
            
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
        setPoseDetected(false); setHandsDetected(0); setKeypointsCount(0);
        setState("default");
    }, [stopRecording]);

    return (
        <div className="relative flex flex-col w-full h-[calc(100vh-2rem)] max-h-screen overflow-hidden">
            <div className="flex-1 p-4 overflow-auto">
                {state !== "show-text" ? (
                    <Recorder state={state} onReset={resetRecorder} onRetry={resetRecorder} />
                ) : (
                    <TextDisplay ref={textContainerRef} translatedText={translatedText} highlightedIndex={highlightedIndex} words={wordsRef.current} onCopyToClipboard={copyToClipboard} />
                )}
                {state === "error" && error && (
                    <div className="flex flex-col items-center justify-center mt-4">
                        <p className="text-red-600 mb-4 text-center">{error}</p>
                        <button onClick={resetRecorder} className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded-md text-sm">Reset</button>
                    </div>
                )}
                {(state === "recording" || state === "initializing") && (
                    <div className="fixed top-4 left-4 bg-black bg-opacity-80 text-white p-3 rounded text-sm z-50 font-mono">
                        <div>Status: <span className="text-green-400">{state}</span></div>
                        <div>Pose: <span className={poseDetected ? "text-green-400" : "text-red-400"}>{poseDetected ? "Detected" : "None"}</span></div>
                        <div>Hands: <span className="text-blue-400">{handsDetected}</span></div>
                        <div>Frames: <span className="text-yellow-400">{keypointsCount}</span></div>
                    </div>
                )}
            </div>
            {(state === 'initializing' || state === 'recording') && (
                <div className="absolute bottom-20 right-4 z-10 md:block">
                    <div className="md:w-48 md:h-32 xl:w-72 xl:h-48 overflow-hidden rounded-lg border border-blue-200 shadow-lg bg-black">
                        <video ref={videoRef} className="w-full h-full object-cover transform -scale-x-100" muted autoPlay playsInline />
                    </div>
                </div>
            )}
            <ControlDock state={state} isTextAvailable={!!translatedText} isPlaying={isPlaying} onStartRecording={startRecording} onStopRecording={stopRecording} onResetRecorder={resetRecorder} onSendRecording={sendRecording} onPlayTranslatedText={playTranslatedText} />
        </div>
    )
}
