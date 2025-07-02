import { RecorderProps } from "@/app/types/RecorderProps";
import { Loader2 } from "lucide-react";
import { forwardRef } from "react";

const Recorder = forwardRef<HTMLDivElement, RecorderProps>(({ onReset, onRetry, state }, ref) => {
  return (
    <div ref={ref} className="flex flex-col items-center justify-center h-full">
      {state === "default" && (
        <div className="flex flex-col items-center justify-center h-full">
          <p className="text-xl text-center text-gray-900">Click the video button to begin</p>
        </div>
      )}
      {state === "recording" && (
        <div className="flex flex-col items-center justify-center h-full">
          <div className="text-center">
            <div className="inline-flex items-center gap-2 bg-blue-50 text-blue-600 px-4 py-2 rounded-full border border-blue-200">
              <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
              <p className="font-medium">Recording...</p>
            </div>
          </div>
        </div>
      )}
      {state === "ready" && (
        <div className="flex flex-col items-center justify-center h-full">
          <p className="text-xl text-center mb-4 text-gray-900">Ready to send</p>
          <p className="text-base text-center text-blue-600">
            Click Send to upload the video or Delete to discard the recording.
          </p>
        </div>
      )}
      {state === "loading" && (
        <div className="flex flex-col items-center justify-center h-full">
          <div className="text-center">
            <Loader2 className="w-10 h-10 text-blue-500 animate-spin mx-auto mb-4" />
            <p className="text-lg text-black">Processing your recording...</p>
          </div>
        </div>
      )}
      {state === "error" && (
        <div className="flex flex-col items-center justify-center h-full">
          <div className="text-center">
            <p className="text-red-600 text-xl mb-2">An error occurred</p>
            <div className="flex gap-2 justify-center">
              <button
                onClick={onRetry}
                className="px-4 py-2 bg-gray-100 text-gray-900 rounded-md text-sm hover:bg-gray-200 border border-gray-300"
              >
                Retry
              </button>
              <button
                onClick={onReset}
                className="px-4 py-2 bg-red-50 text-red-600 rounded-md text-sm hover:bg-red-100 border border-red-200"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )
      }
    </div>
  );
});

export default Recorder;
