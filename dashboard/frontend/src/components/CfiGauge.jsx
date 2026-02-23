import { getCfiColor } from '../utils/colors';

export default function CfiGauge({ value, size = 120 }) {
  const color = getCfiColor(value);
  const clampedValue = Math.min(100, Math.max(0, value));
  const percentage = clampedValue / 100;

  // Semi-circular gauge
  const radius = (size - 16) / 2;
  const circumference = Math.PI * radius;
  const offset = circumference * (1 - percentage);

  const label =
    clampedValue <= 25 ? 'SAFE' :
    clampedValue <= 50 ? 'WATCH' :
    clampedValue <= 75 ? 'ELEVATED' : 'CRITICAL';

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size / 2 + 20} viewBox={`0 0 ${size} ${size / 2 + 20}`}>
        {/* Background arc */}
        <path
          d={`M 8 ${size / 2 + 8} A ${radius} ${radius} 0 0 1 ${size - 8} ${size / 2 + 8}`}
          fill="none"
          stroke="#2F343C"
          strokeWidth="8"
          strokeLinecap="round"
        />
        {/* Value arc */}
        <path
          d={`M 8 ${size / 2 + 8} A ${radius} ${radius} 0 0 1 ${size - 8} ${size / 2 + 8}`}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 0.6s ease-out, stroke 0.3s' }}
        />
        {/* Value text */}
        <text
          x={size / 2}
          y={size / 2 - 2}
          textAnchor="middle"
          fill={color}
          fontSize="24"
          fontWeight="700"
          fontFamily="'JetBrains Mono', monospace"
        >
          {Math.round(clampedValue)}
        </text>
        {/* Label */}
        <text
          x={size / 2}
          y={size / 2 + 16}
          textAnchor="middle"
          fill="#ABB3BF"
          fontSize="10"
          fontWeight="600"
          letterSpacing="2"
        >
          {label}
        </text>
      </svg>
    </div>
  );
}
