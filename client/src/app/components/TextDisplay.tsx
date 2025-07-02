import { TextDisplayProps } from "@/app/types/TextDisplayProps";
import { forwardRef } from "react";
import { Copy } from "lucide-react";
import { cn } from "@/app/lib/utils";

const TextDisplay = forwardRef<HTMLDivElement, TextDisplayProps>(({
    translatedText,
    highlightedIndex,
    words,
    onCopyToClipboard,
}, ref) => {
    if (!translatedText) return null;
    return (
        <div ref={ref} className="relative flex-1 p-6 overflow-auto leading-relaxed font-mono bg-white/50">
            <div className="text-2xl text-center leading-relaxed md:text-left md:text-xl ">
                {words.map((word, index) => (
                    <span key={index}
                        className={cn(
                            "transition-colors duration-200",
                            index === highlightedIndex ? "text-blue-600 font-medium" : "text-gray-400",
                        )}>
                        {word}{" "}
                    </span>
                ))}
            </div>
            <button
                onClick={onCopyToClipboard}
                className="absolute top-3 right-3 p-2 rounded-md hover:bg-gray-100 text-blue-600 border border-gray-200"
                aria-label="Copy to clipboard"
            >
                <Copy className="w-5 h-5" />
            </button>
        </div>
    );
});

export default TextDisplay;