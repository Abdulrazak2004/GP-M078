import { useCallback, useEffect, useRef, useState } from 'react';
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react';

const SPEEDS = [1, 10, 100, 1000];

export default function TimeSlider({ totalSteps, currentStep, onStepChange, playing, onPlayPause }) {
  const [speed, setSpeed] = useState(100);
  const intervalRef = useRef(null);

  // Playback loop
  useEffect(() => {
    if (playing && totalSteps > 0) {
      const ms = Math.max(16, 1000 / speed);
      intervalRef.current = setInterval(() => {
        onStepChange(prev => {
          const next = prev + 1;
          if (next >= totalSteps) {
            onPlayPause(false);
            return totalSteps - 1;
          }
          return next;
        });
      }, ms);
    }
    return () => clearInterval(intervalRef.current);
  }, [playing, speed, totalSteps, onStepChange, onPlayPause]);

  const stepBack = useCallback(() => {
    onStepChange(s => Math.max(0, s - 10));
  }, [onStepChange]);

  const stepForward = useCallback(() => {
    onStepChange(s => Math.min(totalSteps - 1, s + 10));
  }, [onStepChange, totalSteps]);

  const dayNumber = currentStep * 30;
  const years = (dayNumber / 365.25).toFixed(1);

  return (
    <div className="bg-bg-surface rounded-lg border border-bg-border p-3">
      <div className="flex items-center gap-3">
        {/* Controls */}
        <button onClick={stepBack} className="text-text-muted hover:text-text-primary transition-colors">
          <SkipBack size={16} />
        </button>
        <button
          onClick={() => onPlayPause(!playing)}
          className="w-8 h-8 flex items-center justify-center rounded-full bg-accent-blue/20 text-accent-blue hover:bg-accent-blue/30 transition-colors"
        >
          {playing ? <Pause size={14} /> : <Play size={14} className="ml-0.5" />}
        </button>
        <button onClick={stepForward} className="text-text-muted hover:text-text-primary transition-colors">
          <SkipForward size={16} />
        </button>

        {/* Slider */}
        <div className="flex-1 mx-2">
          <input
            type="range"
            min={0}
            max={Math.max(0, totalSteps - 1)}
            value={currentStep}
            onChange={e => onStepChange(Number(e.target.value))}
            className="w-full h-1 bg-bg-border rounded-full appearance-none cursor-pointer
                       [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3
                       [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full
                       [&::-webkit-slider-thumb]:bg-accent-blue [&::-webkit-slider-thumb]:cursor-pointer"
          />
        </div>

        {/* Day counter */}
        <div className="text-right min-w-[100px]">
          <span className="text-sm font-mono tabular-nums text-text-primary">
            Day {dayNumber.toLocaleString()}
          </span>
          <span className="text-xs text-text-muted ml-1">({years} yr)</span>
        </div>

        {/* Speed control */}
        <div className="flex items-center gap-1 border-l border-bg-border pl-3">
          {SPEEDS.map(s => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className={`px-2 py-0.5 text-xs rounded transition-colors ${
                speed === s
                  ? 'bg-accent-blue/20 text-accent-blue'
                  : 'text-text-muted hover:text-text-secondary'
              }`}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
