from typing import List

import librosa
import numpy as np
import soundfile as sf


def load_prompt_wav(prompt_wav: str, sampling_rate: int) -> np.ndarray:
    """Load a wav file and resample if needed. Returns (C, T) float32 array."""
    audio, prompt_sampling_rate = sf.read(prompt_wav, always_2d=True)
    wave = audio.T.astype(np.float32)  # (C, T)

    if prompt_sampling_rate != sampling_rate:
        wave = librosa.resample(wave, orig_sr=prompt_sampling_rate, target_sr=sampling_rate)

    return wave


def remove_silence(
    audio: np.ndarray,
    sampling_rate: int,
    only_edge: bool = False,
    trail_sil: float = 0,
) -> np.ndarray:
    """
    Remove silences longer than 1 second, and edge silences longer than 0.1 seconds.

    Parameters:
        audio: numpy array with shape (C, T).
        sampling_rate: sampling rate of the audio.
        only_edge: If true, only remove edge silences.
        trail_sil: duration of added trailing silence in ms.
    """
    mono = audio.mean(axis=0) if audio.ndim > 1 else audio

    if not only_edge:
        intervals = librosa.effects.split(mono, top_db=50, ref=1.0)
        if len(intervals) > 0:
            min_silence = sampling_rate
            merged = [list(intervals[0])]
            for start, end in intervals[1:]:
                if start - merged[-1][1] < min_silence:
                    merged[-1][1] = end
                else:
                    merged.append([start, end])

            keep = int(sampling_rate)
            segments = []
            prev_e = 0
            for start, end in merged:
                s = max(prev_e, start - keep)
                e = min(audio.shape[-1], end + keep)
                segments.append(audio[..., s:e])
                prev_e = e
            audio = np.concatenate(segments, axis=-1)
            mono = audio.mean(axis=0) if audio.ndim > 1 else audio

    _, (trim_start, trim_end) = librosa.effects.trim(mono, top_db=50, ref=1.0, frame_length=512, hop_length=128)
    keep_edge = int(0.1 * sampling_rate)
    start = max(0, trim_start - keep_edge)
    end = min(audio.shape[-1], trim_end + keep_edge)
    audio = audio[..., start:end]

    if trail_sil > 0:
        trail_samples = int(trail_sil * sampling_rate / 1000)
        silence = np.zeros((*audio.shape[:-1], trail_samples), dtype=audio.dtype)
        audio = np.concatenate([audio, silence], axis=-1)

    return audio.copy()


def cross_fade_concat(
    chunks: List[np.ndarray], fade_duration: float = 0.1, sample_rate: int = 24000
) -> np.ndarray:
    """Concatenate audio chunks with cross-fading. Each chunk has shape (C, T)."""
    if len(chunks) <= 1:
        return chunks[0] if chunks else np.array([])

    fade_samples = int(fade_duration * sample_rate)

    if fade_samples <= 0:
        return np.concatenate(chunks, axis=-1)

    final = chunks[0]

    for next_chunk in chunks[1:]:
        k = min(fade_samples, final.shape[-1], next_chunk.shape[-1])

        if k <= 0:
            final = np.concatenate([final, next_chunk], axis=-1)
            continue

        fade = np.linspace(1, 0, k, dtype=np.float32)[np.newaxis]
        final = np.concatenate(
            [
                final[..., :-k],
                final[..., -k:] * fade + next_chunk[..., :k] * (1 - fade),
                next_chunk[..., k:],
            ],
            axis=-1,
        )

    return final
