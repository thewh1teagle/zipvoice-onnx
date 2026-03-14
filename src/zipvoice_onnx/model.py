from typing import List

import numpy as np
import onnxruntime as ort


def get_ort_session_options(num_thread: int) -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = num_thread
    opts.intra_op_num_threads = num_thread
    return opts


def get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
) -> np.ndarray:
    timesteps = np.linspace(t_start, t_end, num_step + 1, dtype=np.float32)
    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)
    return timesteps


class OnnxModel:
    def __init__(
        self,
        text_encoder_path: str,
        fm_decoder_path: str,
        num_thread: int = 1,
        onnx_providers: List[str] = ["CPUExecutionProvider"],
    ):
        self.session_opts = get_ort_session_options(num_thread)
        self.onnx_providers = onnx_providers

        self.init_text_encoder(text_encoder_path)
        self.init_fm_decoder(fm_decoder_path)

    def init_text_encoder(self, model_path: str):
        self.text_encoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=self.onnx_providers,
        )
        self._te_in = [i.name for i in self.text_encoder.get_inputs()]
        self._te_out = [self.text_encoder.get_outputs()[0].name]

    def init_fm_decoder(self, model_path: str):
        self.fm_decoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=self.onnx_providers,
        )
        self._fm_in = [i.name for i in self.fm_decoder.get_inputs()]
        self._fm_out = [self.fm_decoder.get_outputs()[0].name]
        meta = self.fm_decoder.get_modelmeta().custom_metadata_map
        self.feat_dim = int(meta["feat_dim"])

    def run_text_encoder(self, tokens, prompt_tokens, prompt_features_len, speed) -> np.ndarray:
        out = self.text_encoder.run(
            self._te_out,
            dict(zip(self._te_in, [tokens, prompt_tokens, prompt_features_len, speed])),
        )
        return out[0]

    def run_fm_decoder(self, t, x, text_condition, speech_condition, guidance_scale) -> np.ndarray:
        out = self.fm_decoder.run(
            self._fm_out,
            dict(zip(self._fm_in, [t, x, text_condition, speech_condition, guidance_scale])),
        )
        return out[0]


def sample(
    model: OnnxModel,
    tokens: List[List[int]],
    prompt_tokens: List[List[int]],
    prompt_features: np.ndarray,
    speed: float = 1.0,
    t_shift: float = 0.5,
    guidance_scale: float = 1.0,
    num_step: int = 16,
) -> np.ndarray:
    assert len(tokens) == len(prompt_tokens) == 1
    tokens_arr = np.array(tokens, dtype=np.int64)
    prompt_tokens_arr = np.array(prompt_tokens, dtype=np.int64)
    prompt_features_len = np.array(prompt_features.shape[1], dtype=np.int64)
    speed_arr = np.array(speed, dtype=np.float32)

    text_condition = model.run_text_encoder(
        tokens_arr, prompt_tokens_arr, prompt_features_len, speed_arr
    )

    batch_size, num_frames, _ = text_condition.shape
    assert batch_size == 1
    feat_dim = model.feat_dim

    timesteps = get_time_steps(t_start=0.0, t_end=1.0, num_step=num_step, t_shift=t_shift)
    x = np.random.randn(batch_size, num_frames, feat_dim).astype(np.float32)
    pad_len = num_frames - prompt_features.shape[1]
    speech_condition = np.pad(prompt_features, ((0, 0), (0, pad_len), (0, 0)))
    guidance_scale_arr = np.array(guidance_scale, dtype=np.float32)

    for step in range(num_step):
        v = model.run_fm_decoder(
            t=np.array(timesteps[step], dtype=np.float32),
            x=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            guidance_scale=guidance_scale_arr,
        )
        x = x + v * (timesteps[step + 1] - timesteps[step])

    x = x[:, int(prompt_features_len):, :]
    return x
