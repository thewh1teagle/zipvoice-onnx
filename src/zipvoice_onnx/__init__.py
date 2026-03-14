from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import json
import logging
import numpy as np
import torch
from typing import List
from dataclasses import field

from .model import OnnxModel, sample
from .vocoder import get_vocoder, VocosFbank, rms_norm
from .tokenizer import Tokenizer
from .utils import (
    load_prompt_wav,
    chunk_tokens_punctuation,
    cross_fade_concat,
)
from .audio_processing import remove_silence


@dataclass
class ZipVoiceOptions:
    text_encoder_path: str
    fm_decoder_path: str
    text_encoder_int8_path: str = ""
    fm_decoder_int8_path: str = ""
    model_json_path: str = ""
    tokens_path: str = ""
    vocos_model_path: Optional[str] = None
    onnx_providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])


class ZipVoice:
    def __init__(self, options: ZipVoiceOptions, num_thread: int = 4, seed: int = 666):
        self.options = options
        self.num_thread = num_thread
        self.seed = seed
        
        # Validate paths
        text_encoder_path = Path(options.text_encoder_path)
        fm_decoder_path = Path(options.fm_decoder_path)
        model_json_path = Path(options.model_json_path) if options.model_json_path else None
        tokens_path = Path(options.tokens_path) if options.tokens_path else None
        
        if not text_encoder_path.is_file():
            raise FileNotFoundError(f"{text_encoder_path} does not exist")
        if not fm_decoder_path.is_file():
            raise FileNotFoundError(f"{fm_decoder_path} does not exist")
        
        # Set up threading
        torch.set_num_threads(num_thread)
        torch.set_num_interop_threads(num_thread)
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize tokenizer
        if tokens_path:
            if not tokens_path.is_file():
                raise FileNotFoundError(f"{tokens_path} does not exist")
            self.tokenizer = Tokenizer(token_file=str(tokens_path))
        else:
            raise ValueError("tokens_path must be provided")
        
        # Load model config
        if model_json_path:
            if not model_json_path.is_file():
                raise FileNotFoundError(f"{model_json_path} does not exist")
            with open(model_json_path, "r") as f:
                self.model_config = json.load(f)
        else:
            # Default config
            self.model_config = {
                "feature": {
                    "type": "vocos",
                    "sampling_rate": 24000,
                }
            }
        
        # Initialize model
        self.model = OnnxModel(str(text_encoder_path), str(fm_decoder_path), num_thread=num_thread, onnx_providers=options.onnx_providers)
        
        # Initialize vocoder and feature extractor
        self.vocoder = get_vocoder(options.vocos_model_path)
        self.vocoder.eval()
        
        if self.model_config["feature"]["type"] == "vocos":
            self.feature_extractor = VocosFbank()
        else:
            raise NotImplementedError(
                f"Unsupported feature type: {self.model_config['feature']['type']}"
            )
        
        self.sampling_rate = self.model_config["feature"]["sampling_rate"]
        
    @torch.inference_mode()
    def create(
        self,
        ref_wav: str,
        ref_phonemes: str,
        target_phonemes: str,
        speed: float = 1.0,
        num_steps: int | None = None,
        guidance_scale: float = 1.0,
        t_shift: float = 0.5,
        target_rms: float = 0.1,
        feat_scale: float = 0.1,
        remove_long_sil: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech audio from phonemes using a reference audio sample.
        
        Args:
            ref_wav: Path to reference audio file
            ref_phonemes: Phonemes string for the reference audio
            target_phonemes: Phonemes string to generate
            speed: Speed control factor
            num_steps: Number of steps for decoding (default: 16)
            guidance_scale: Scale for classifier-free guidance (default: 1.0)
            t_shift: Time shift parameter (default: 0.5)
            target_rms: Target RMS for waveform normalization (default: 0.1)
            feat_scale: Scale for features (default: 0.1)
            remove_long_sil: Whether to remove long silences (default: False)
            
        Returns:
            Tuple of (audio_samples, sampling_rate) where audio_samples is a numpy array
        """
        if num_steps is None:
            num_steps = 8
        
        # Load and process prompt wav
        prompt_wav = load_prompt_wav(ref_wav, sampling_rate=self.sampling_rate)
        
        # Remove edge and long silences in the prompt wav.
        # Add 0.2s trailing silence to avoid leaking prompt to generated speech.
        prompt_wav = remove_silence(
            prompt_wav, self.sampling_rate, only_edge=False, trail_sil=200
        )
        
        prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)
        
        prompt_duration = prompt_wav.shape[-1] / self.sampling_rate
        
        if prompt_duration > 20:
            logging.warning(
                f"Given prompt wav is too long ({prompt_duration}s). "
                f"Please provide a shorter one (1-3 seconds is recommended)."
            )
        elif prompt_duration > 10:
            logging.warning(
                f"Given prompt wav is long ({prompt_duration}s). "
                f"It will lead to slower inference speed and possibly worse speech quality."
            )
        
        # Extract features from prompt wav
        prompt_features = self.feature_extractor.extract(prompt_wav, sampling_rate=self.sampling_rate)
        
        prompt_features = prompt_features.unsqueeze(0) * feat_scale
        
        # Tokenize text (str tokens), punctuations will be preserved.
        tokens_str = self.tokenizer.texts_to_tokens([target_phonemes])[0]
        prompt_tokens_str = self.tokenizer.texts_to_tokens([ref_phonemes])[0]
        
        # chunk text so that each len(prompt wav + generated wav) is around 25 seconds.
        token_duration = (prompt_wav.shape[-1] / self.sampling_rate) / (
            len(prompt_tokens_str) * speed
        )
        max_tokens = int((25 - prompt_duration) / token_duration)
        chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)
        
        # Tokenize text (int tokens)
        chunked_tokens = self.tokenizer.tokens_to_token_ids(chunked_tokens_str)
        prompt_tokens = self.tokenizer.tokens_to_token_ids([prompt_tokens_str])
        
        # Start predicting features
        chunked_features = []
        for tokens in chunked_tokens:
            # Generate features
            pred_features = sample(
                model=self.model,
                tokens=[tokens],
                prompt_tokens=prompt_tokens,
                prompt_features=prompt_features,
                speed=speed,
                t_shift=t_shift,
                guidance_scale=guidance_scale,
                num_step=num_steps,
            )
            
            # Postprocess predicted features
            pred_features = pred_features.permute(0, 2, 1) / feat_scale  # (B, C, T)
            chunked_features.append(pred_features)
        
        # Start vocoder processing
        chunked_wavs = []
        for pred_features in chunked_features:
            wav = self.vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)
            # Adjust wav volume if necessary
            if prompt_rms < target_rms:
                wav = wav * prompt_rms / target_rms
            chunked_wavs.append(wav)
        
        # Merge chunked wavs
        final_wav = cross_fade_concat(
            chunked_wavs, fade_duration=0.1, sample_rate=self.sampling_rate
        )
        final_wav = remove_silence(
            final_wav, self.sampling_rate, only_edge=(not remove_long_sil), trail_sil=0
        )
        
        # Convert to numpy array
        audio_samples = final_wav.cpu().numpy()
        
        # If mono, return as 1D array; if stereo, return as 2D array (channels, time)
        if audio_samples.shape[0] == 1:
            audio_samples = audio_samples[0]
        
        return audio_samples, self.sampling_rate
