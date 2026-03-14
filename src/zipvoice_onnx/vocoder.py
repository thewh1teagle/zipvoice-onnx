import librosa
import numpy as np
import onnxruntime as ort

from .model import get_ort_session_options


class VocosFbank:
    """Feature extractor for computing mel spectrograms compatible with Vocos vocoder."""

    def __init__(
        self,
        sampling_rate: int = 24000,
        n_mels: int = 100,
        n_fft: int = 1024,
        hop_length: int = 256,
        num_channels: int = 1,
    ):
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_channels = num_channels

        self.mel_filters = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            norm=None,
            htk=True,
        ).astype(np.float32)  # (n_mels, n_fft//2+1)

    def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Extract mel spectrogram features from audio samples.

        Args:
            samples: numpy array with shape (C, T)
            sampling_rate: Sampling rate of the audio.

        Returns:
            numpy array with shape (T, n_mels)
        """
        assert sampling_rate == self.sampling_rate

        if samples.ndim == 1:
            samples = samples[np.newaxis]

        if self.num_channels == 1:
            if samples.shape[0] == 2:
                samples = samples.mean(axis=0, keepdims=True)
        else:
            assert samples.shape[0] == 2, samples.shape

        mels = []
        for ch in samples:
            stft = librosa.stft(ch, n_fft=self.n_fft, hop_length=self.hop_length, center=True, window="hann")
            mel = self.mel_filters @ np.abs(stft)  # (n_mels, T)
            mels.append(mel)
        mel = np.stack(mels, axis=0)  # (C, n_mels, T)

        logmel = np.log(np.clip(mel, a_min=1e-7, a_max=None))
        logmel = logmel.reshape(-1, logmel.shape[-1]).T  # (T, n_mels)

        return logmel.astype(np.float32)


class OnnxVocoder:
    def __init__(self, model_path: str, num_thread: int = 1, onnx_providers: list = ["CPUExecutionProvider"]):
        sess_opts = get_ort_session_options(num_thread)
        self.session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=onnx_providers)

        meta = self.session.get_modelmeta().custom_metadata_map
        self.n_fft = int(meta.get("n_fft", 1024))
        self.hop_length = int(meta.get("hop_length", 256))
        self.win_length = int(meta.get("win_length", 1024))

    def decode(self, mel: np.ndarray) -> np.ndarray:
        """
        Args:
            mel: (1, n_mels, T)
        Returns:
            (1, T_audio)
        """
        mag, x, y = self.session.run(None, {self.session.get_inputs()[0].name: mel})
        complex_spec = mag[0] * (x[0] + 1j * y[0])  # (n_fft//2+1, T)
        wav = librosa.istft(complex_spec, hop_length=self.hop_length, win_length=self.win_length, window="hann", center=True)
        return wav[np.newaxis].astype(np.float32)  # (1, T_audio)


def rms_norm(audio: np.ndarray, target_rms: float):
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < target_rms:
        audio = audio * target_rms / rms
    return audio, float(rms)
