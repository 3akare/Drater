import { ControlDockProps } from "@/app/types/ControlDocksProps";
import { Play, Send, Square, Trash, Camera, X } from "lucide-react";
import { cn } from "@/app/lib/utils";

const ControlDock = ({
  state,
  isTextAvailable,
  isPlaying,
  onStartRecording,
  onStopRecording,
  onResetRecorder,
  onSendRecording,
  onPlayTranslatedText
}: ControlDockProps) => {
  const renderFirstButtonIcon = () => {
    if (state === "recording") return <Square className="w-6 h-6" />;
    else if (state === "ready") return <Trash className="w-6 h-6" />;
    else return <Camera className="w-6 h-6" />;
  };

  const handleFirstButtonClick = () => {
    if (state === "recording") onStopRecording();
    else if (state === "ready") onResetRecorder();
    else onStartRecording();
  };

  return (
    <div className="sticky bottom-0 left-0 right-0 flex justify-center pb-4 md:pb-6 pt-2">
      <div className="bg-white/90 backdrop-blur-md border border-blue-200 rounded-full px-4 md:px-6 py-2 md:py-3 shadow-lg flex items-center gap-4 md:gap-8">
        {/* First Button (Camera/Stop/Trash) */}
        <button
          onClick={handleFirstButtonClick}
          disabled={["loading", "show-text"].includes(state)}
          className={cn(
            "p-2 md:p-3 rounded-full transition-all transform hover:scale-110",
            state === "recording"
              ? "bg-blue-100 text-blue-600 shadow-md border border-blue-200"
              : state === "ready"
                ? "bg-gray-100 text-gray-600 shadow-md border border-gray-200"
                : "bg-gray-50 hover:bg-gray-100 text-blue-600 border border-gray-200",
            ["loading", "show-text"].includes(state) && "opacity-50 cursor-not-allowed",
          )}
          aria-label={
            state === "recording" ? "Stop recording" : state === "ready" ? "Delete recording" : "Start recording"
          }
        >
          {renderFirstButtonIcon()}
        </button>

        {/* Play Button */}
        <button
          onClick={onPlayTranslatedText}
          disabled={!isTextAvailable || ["recording", "loading"].includes(state)}
          className={cn(
            "p-2 md:p-3 rounded-full transition-all transform hover:scale-110",
            isPlaying
              ? "bg-blue-100 text-blue-600 shadow-md border border-blue-200"
              : "bg-gray-50 hover:bg-gray-100 text-blue-600 border border-gray-200",
            (!isTextAvailable || ["recording", "loading"].includes(state)) && "opacity-50 cursor-not-allowed",
          )}
          aria-label={isPlaying ? "Restart speech" : "Play translated text"}
        >
          <Play className="w-5 h-5 md:w-6 md:h-6" />
        </button>
        {/* Send Button */}
        <button
          onClick={state === "ready" ? onSendRecording : onResetRecorder}
          disabled={["recording", "loading", "default"].includes(state)}
          className={cn(
            "p-2 md:p-3 rounded-full transition-all transform hover:scale-110",
            state === "ready"
              ? "bg-blue-100 text-blue-600 shadow-md border border-blue-200"
              : state === "show-text" || state === "error"
                ? "bg-gray-50 hover:bg-gray-100 text-blue-600 border border-gray-200"
                : "bg-gray-50 text-gray-400 border border-gray-200",
            ["recording", "loading", "default"].includes(state) && "opacity-50 cursor-not-allowed",
          )}
          aria-label={state === "ready" ? "Send recording" : "Reset"}
        >
          {state === "ready" ? <Send className="w-5 h-5 md:w-6 md:h-6" /> : <X className="w-5 h-5 md:w-6 md:h-6" />}
        </button>
      </div>
    </div>
  );
};

export default ControlDock;