import torch
import torchaudio
import soundfile as sf
import numpy as np
from typing import List

# Punctuation set for chunking
punctuation = {";", ":", ",", ".", "!", "?", "；", "：", "，", "。", "！", "？"}


def load_prompt_wav(prompt_wav: str, sampling_rate: int):
    """
    Load the waveform with torchaudio and resampling if needed.

    Parameters:
        prompt_wav: path of the prompt wav.
        sampling_rate: target sampling rate.

    Returns:
        Loaded prompt waveform with target sampling rate,
        PyTorch tensor of shape (C, T)
    """
    audio, prompt_sampling_rate = sf.read(prompt_wav, always_2d=True)
    prompt_wav = torch.from_numpy(audio.T).float()  # (C, T)

    if prompt_sampling_rate != sampling_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=prompt_sampling_rate, new_freq=sampling_rate
        )
        prompt_wav = resampler(prompt_wav)
    return prompt_wav


def chunk_tokens_punctuation(tokens_list: List[str], max_tokens: int = 100):
    """
    Splits the input tokens list into chunks according to punctuations,
        each with a maximum number of tokens.

    Args:
        token_list (list of str): The list of tokens to be split.
        max_tokens (int): The maximum number of tokens per chunk.

    Returns:
        List[str]: A list of text chunks.
    """

    # 1. Split the tokens according to punctuations.
    sentences = []
    current_sentence = []
    for token in tokens_list:
        # If the first token of current sentence is punctuation or blank,
        # append it to the end of the previous sentence.
        if (
            len(current_sentence) == 0
            and len(sentences) != 0
            and (token in punctuation or token == " ")
        ):
            sentences[-1].append(token)
        # Otherwise, append the current token to the current sentence.
        else:
            current_sentence.append(token)
            # Split the sentence in positions of punctuations.
            if token in punctuation:
                sentences.append(current_sentence)
                current_sentence = []
    # Assume the last few tokens are also a sentence
    if len(current_sentence) != 0:
        sentences.append(current_sentence)

    # 2. Merge short sentences.
    chunks = []
    current_chunk = []
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_tokens:
            current_chunk.extend(sentence)
        else:
            if len(current_chunk) > 0:
                chunks.append(current_chunk)
            current_chunk = sentence

    if len(current_chunk) > 0:
        chunks.append(current_chunk)

    return chunks


def get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute the intermediate time steps for sampling.

    Args:
        t_start: The starting time of the sampling (default is 0).
        t_end: The starting time of the sampling (default is 1).
        num_step: The number of sampling.
        t_shift: shift the t toward smaller numbers so that the sampling
            will emphasize low SNR region. Should be in the range of (0, 1].
            The shifting will be more significant when the number is smaller.
        device: A torch device.
    Returns:
        The time step with the shape (num_step + 1,).
    """

    timesteps = torch.linspace(t_start, t_end, num_step + 1).to(device)

    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)

    return timesteps


def cross_fade_concat(
    chunks: List[torch.Tensor], fade_duration: float = 0.1, sample_rate: int = 24000
) -> torch.Tensor:
    """
    Concatenates audio chunks with cross-fading between consecutive chunks.

    Args:
        chunks: List of audio tensors, each with shape (C, T) where
                C = number of channel, T = time dimension (samples)
        fade_duration: Duration of cross-fade in seconds
        sample_rate: Audio sample rate in Hz

    Returns:
        Concatenated audio tensor with shape (N, T_total)
    """
    # Handle edge cases: empty input or single chunk
    if len(chunks) <= 1:
        return chunks[0] if chunks else torch.tensor([])

    # Calculate total fade samples from duration and sample rate
    fade_samples = int(fade_duration * sample_rate)

    # Use simple concatenation if fade duration is non-positive
    if fade_samples <= 0:
        return torch.cat(chunks, dim=-1)

    # Initialize final tensor with the first chunk
    final = chunks[0]

    # Iterate through remaining chunks to apply cross-fading
    for next_chunk in chunks[1:]:
        # Calculate safe fade length (cannot exceed either chunk's duration)
        k = min(fade_samples, final.shape[-1], next_chunk.shape[-1])

        # Fall back to simple concatenation if safe fade length is invalid
        if k <= 0:
            final = torch.cat([final, next_chunk], dim=-1)
            continue

        # Create fade curve (1 -> 0) with shape (1, k) for broadcasting
        fade = torch.linspace(1, 0, k, device=final.device)[None]

        # Concatenate three parts:
        # 1. Non-overlapping part of previous audio
        # 2. Cross-faded overlapping region
        # 3. Non-overlapping part of next audio
        final = torch.cat(
            [
                final[..., :-k],  # All samples except last k from previous
                final[..., -k:] * fade
                + next_chunk[..., :k] * (1 - fade),  # Cross-fade region
                next_chunk[..., k:],  # All samples except first k from next
            ],
            dim=-1,
        )

    return final

