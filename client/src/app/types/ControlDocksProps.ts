export interface ControlDockProps {
    state: "default" | "initializing" | "recording" | "ready" | "loading" | "error" | "show-text";
    isTextAvailable: boolean;
    isPlaying: boolean;
    onStartRecording: () => void;
    onStopRecording: () => void;
    onResetRecorder: () => void;
    onSendRecording: () => void;
    onPlayTranslatedText: () => void;
}
