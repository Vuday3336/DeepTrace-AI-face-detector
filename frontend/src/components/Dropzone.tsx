import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import { validateUpload } from "../lib/validation";

interface Props {
  onFile: (file: File) => void;
  onInvalid: (message: string) => void;
  disabled?: boolean;
}

export function Dropzone({ onFile, onInvalid, disabled = false }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const accept = useCallback(
    (file: File | undefined) => {
      if (!file || disabled) return;
      const problem = validateUpload(file);
      if (problem) onInvalid(problem);
      else onFile(file);
    },
    [disabled, onFile, onInvalid],
  );

  // Paste an image from the clipboard anywhere on the page
  useEffect(() => {
    const onPaste = (event: ClipboardEvent) => {
      const item = Array.from(event.clipboardData?.items ?? []).find((i) => i.kind === "file");
      accept(item?.getAsFile() ?? undefined);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [accept]);

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    accept(event.dataTransfer.files[0]);
  };

  return (
    <div
      role="button"
      tabIndex={0}
      aria-disabled={disabled}
      aria-label="Upload an image: drop a file, click to browse, or paste"
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && !disabled && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-14 text-center transition
        ${dragging ? "border-accent bg-accent/5" : "border-line hover:border-slate-500"}
        ${disabled ? "pointer-events-none opacity-50" : ""}`}
    >
      <svg aria-hidden viewBox="0 0 24 24" className="h-10 w-10 text-accent" fill="none" stroke="currentColor" strokeWidth="1.6">
        <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M12 4v12m0-12l-4 4m4-4l4 4" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <p className="text-base text-slate-200">
        Drop a photo here, <span className="text-accent underline">browse</span>, or paste
      </p>
      <p className="text-xs text-slate-500">JPG, PNG or WebP · up to 10 MB · every face in the photo is analysed</p>
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        data-testid="file-input"
        onChange={(e) => {
          accept(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
    </div>
  );
}
