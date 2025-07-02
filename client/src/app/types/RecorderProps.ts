export type RecorderState = "default" | "initializing" | "recording" | "ready" | "loading" | "error" | "show-text";

export interface RecorderProps {
    onReset: () => void;
    onRetry: () => void;
    state: RecorderState;
}
