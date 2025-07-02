export interface TextDisplayProps {
    translatedText: string;
    highlightedIndex: number;
    words: string[];
    onCopyToClipboard: () => void;
}
