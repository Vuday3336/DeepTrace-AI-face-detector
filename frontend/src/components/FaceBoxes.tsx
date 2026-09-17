import type { Face, Label } from "../api/types";

const STROKE: Record<Label, string> = { REAL: "#34d399", AI_GENERATED: "#fb7185", UNCERTAIN: "#fbbf24" };

interface Props {
  src: string;
  width: number;
  height: number;
  faces: Face[];
  selected: number;
  onSelect: (index: number) => void;
}

/** The uploaded photo with numbered, clickable face boxes (SVG scales with the image). */
export function FaceBoxes({ src, width, height, faces, selected, onSelect }: Props) {
  const stroke = Math.max(2, Math.round(Math.max(width, height) / 250));
  return (
    <div className="relative overflow-hidden rounded-lg bg-black">
      <img src={src} alt="Uploaded photo" className="block h-auto w-full" />
      <svg viewBox={`0 0 ${width} ${height}`} className="absolute inset-0 h-full w-full" aria-label="Detected faces">
        {faces.map((face) => {
          const [x1, y1, x2, y2] = face.bbox;
          const active = face.face_index === selected;
          const color = STROKE[face.label];
          const tag = Math.max(18, Math.round(Math.max(width, height) / 30));
          return (
            <g key={face.face_index} onClick={() => onSelect(face.face_index)} className="cursor-pointer">
              <rect
                x={x1}
                y={y1}
                width={x2 - x1}
                height={y2 - y1}
                fill={active ? `${color}22` : "transparent"}
                stroke={color}
                strokeWidth={active ? stroke * 1.8 : stroke}
                rx={stroke * 2}
              />
              <rect x={x1} y={Math.max(0, y1 - tag)} width={tag} height={tag} fill={color} rx={stroke} />
              <text
                x={x1 + tag / 2}
                y={Math.max(0, y1 - tag) + tag * 0.72}
                textAnchor="middle"
                fontSize={tag * 0.65}
                fontWeight="700"
                fill="#0b1220"
              >
                {face.face_index + 1}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
