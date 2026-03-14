from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List
import json
import logging
import random
import numpy as np

from .model import OnnxModel, sample
from .vocoder import OnnxVocoder, VocosFbank, rms_norm
from .tokenizer import Tokenizer, chunk_tokens_punctuation
from .audio import load_prompt_wav, remove_silence, cross_fade_concat


@dataclass
class ZipVoiceOptions:
    text_encoder_path: str
    fm_decoder_path: str
    vocoder_path: str
    text_encoder_int8_path: str = ""
    fm_decoder_int8_path: str = ""
    model_json_path: str = ""
    tokens_path: str = ""
    onnx_providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])


class ZipVoice:
    def __init__(self, options: ZipVoiceOptions, num_thread: int = 4, seed: int = 666):
        self.options = options
        self.num_thread = num_thread
        self.seed = seed

        text_encoder_path = Path(options.text_encoder_path)
        fm_decoder_path = Path(options.fm_decoder_path)
        model_json_path = Path(options.model_json_path) if options.model_json_path else None
        tokens_path = Path(options.tokens_path) if options.tokens_path else None

        if not text_encoder_path.is_file():
            raise FileNotFoundError(f"{text_encoder_path} does not exist")
        if not fm_decoder_path.is_file():
            raise FileNotFoundError(f"{fm_decoder_path} does not exist")

        random.seed(seed)
        np.random.seed(seed)

        if tokens_path:
            if not tokens_path.is_file():
                raise FileNotFoundError(f"{tokens_path} does not exist")
            self.tokenizer = Tokenizer(token_file=str(tokens_path))
        else:
            raise ValueError("tokens_path must be provided")

        if model_json_path:
            if not model_json_path.is_file():
                raise FileNotFoundError(f"{model_json_path} does not exist")
            with open(model_json_path, "r") as f:
                self.model_config = json.load(f)
        else:
            self.model_config = {
                "feature": {
                    "type": "vocos",
                    "sampling_rate": 24000,
                }
            }

        self.model = OnnxModel(str(text_encoder_path), str(fm_decoder_path), num_thread=num_thread, onnx_providers=options.onnx_providers)
        self.vocoder = OnnxVocoder(options.vocoder_path, num_thread=num_thread, onnx_providers=options.onnx_providers)

        if self.model_config["feature"]["type"] == "vocos":
            self.feature_extractor = VocosFbank()
        else:
            raise NotImplementedError(f"Unsupported feature type: {self.model_config['feature']['type']}")

        self.sampling_rate = self.model_config["feature"]["sampling_rate"]

    def create(
        self,
        ref_wav: str,
        ref_phonemes: str,
        target_phonemes: str,
        speed: float = 1.0,
        num_steps: int = 8,
        guidance_scale: float = 1.0,
        t_shift: float = 0.5,
        target_rms: float = 0.1,
        feat_scale: float = 0.1,
        remove_long_sil: bool = False,
    ) -> Tuple[np.ndarray, int]:
        prompt_wav = load_prompt_wav(ref_wav, sampling_rate=self.sampling_rate)
        prompt_wav = remove_silence(prompt_wav, self.sampling_rate, only_edge=False, trail_sil=200)
        prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

        prompt_duration = prompt_wav.shape[-1] / self.sampling_rate

        if prompt_duration > 20:
            logging.warning(f"Given prompt wav is too long ({prompt_duration}s). Please provide a shorter one (1-3 seconds is recommended).")
        elif prompt_duration > 10:
            logging.warning(f"Given prompt wav is long ({prompt_duration}s). It will lead to slower inference speed and possibly worse speech quality.")

        prompt_features = self.feature_extractor.extract(prompt_wav, sampling_rate=self.sampling_rate)
        prompt_features = prompt_features[np.newaxis] * feat_scale  # (1, T, n_mels)

        tokens_str = self.tokenizer.texts_to_tokens([target_phonemes])[0]
        prompt_tokens_str = self.tokenizer.texts_to_tokens([ref_phonemes])[0]

        token_duration = (prompt_wav.shape[-1] / self.sampling_rate) / (len(prompt_tokens_str) * speed)
        max_tokens = int((25 - prompt_duration) / token_duration)
        chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)

        chunked_tokens = self.tokenizer.tokens_to_token_ids(chunked_tokens_str)
        prompt_tokens = self.tokenizer.tokens_to_token_ids([prompt_tokens_str])

        chunked_wavs = []
        for tokens in chunked_tokens:
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
            pred_features = (np.transpose(pred_features, (0, 2, 1)) / feat_scale).astype(np.float32)  # (B, n_mels, T)
            wav = self.vocoder.decode(pred_features)  # (1, T_audio)
            wav = np.clip(wav, -1, 1)
            if prompt_rms < target_rms:
                wav = wav * prompt_rms / target_rms
            chunked_wavs.append(wav)

        final_wav = cross_fade_concat(chunked_wavs, fade_duration=0.1, sample_rate=self.sampling_rate)
        final_wav = remove_silence(final_wav, self.sampling_rate, only_edge=(not remove_long_sil), trail_sil=0)

        if final_wav.shape[0] == 1:
            final_wav = final_wav[0]

        return final_wav, self.sampling_rate
