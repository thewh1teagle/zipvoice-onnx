import onnxruntime as ort
import torch
import torchaudio


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
        
        # Use torchaudio transforms to match training setup
        self.fbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1,
        )

    def extract(
        self,
        samples: torch.Tensor,
        sampling_rate: int,
    ) -> torch.Tensor:
        """
        Extract mel spectrogram features from audio samples.

        Args:
            samples: PyTorch tensor with shape (C, T) where C is channels, T is time.
            sampling_rate: Sampling rate of the audio.

        Returns:
            PyTorch tensor with shape (T, n_mels) containing log mel spectrogram features.
        """
        # Check for sampling rate compatibility
        assert sampling_rate == self.sampling_rate, (
            f"Mismatched sampling rate: extractor expects {self.sampling_rate}, "
            f"got {sampling_rate}"
        )
        
        # Ensure samples have the right shape: (C, T)
        if len(samples.shape) == 1:
            samples = samples.unsqueeze(0)
        
        # Handle multi-channel audio
        if self.num_channels == 1:
            if samples.shape[0] == 2:
                samples = samples.mean(dim=0, keepdims=True)
        else:
            assert samples.shape[0] == 2, samples.shape

        # Compute mel spectrogram using torchaudio
        mel = self.fbank(samples)  # (1, n_mels, T) or (2, n_mels, T)
        logmel = mel.clamp(min=1e-7).log()

        # Reshape to (T, n_mels) or (T, 2 * n_mels)
        logmel = logmel.reshape(-1, logmel.shape[-1]).t()  # (time, n_mels) or (time, 2 * n_mels)

        return logmel


def get_vocoder(onnx_model_path: str):
    """
    Get an ONNX vocoder instance.
    
    Args:
        onnx_model_path: Path to ONNX vocoder model
    
    Returns:
        OnnxVocoder instance
    """
    return OnnxVocoder(onnx_model_path)


def rms_norm(prompt_wav: torch.Tensor, target_rms: float):
    """
    Normalize the rms of prompt_wav is it is smaller than target rms.

    Parameters:
        prompt_wav: PyTorch tensor with shape (C, T).
        target_rms: target rms value

    Returns:
        prompt_wav: normalized prompt wav with shape (C, T).
        promt_rms: rms of original prompt wav. Will be used to
            re-normalize the generated wav.
    """
    prompt_rms = torch.sqrt(torch.mean(torch.square(prompt_wav)))
    if prompt_rms < target_rms:
        prompt_wav = prompt_wav * target_rms / prompt_rms
    return prompt_wav, prompt_rms


class OnnxVocoder:
    """
    ONNX-compatible Vocos vocoder wrapper.
    
    The ONNX model outputs magnitude, cos(phase), and sin(phase) instead of audio
    (since ONNX doesn't support complex numbers). This class reconstructs the audio
    by converting to complex spectrogram and applying ISTFT.
    
    Args:
        onnx_model_path: Path to the ONNX vocoder model file
        n_fft: FFT window size (default: 1024)
        hop_length: Hop length for STFT (default: 256)
        sampling_rate: Audio sampling rate (default: 24000)
        num_thread: Number of threads for ONNX inference (default: 1)
    """
    
    def __init__(
        self,
        onnx_model_path: str,
        n_fft: int = 1024,
        hop_length: int = 256,
        sampling_rate: int = 24000,
        num_thread: int = 1,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.win_length = n_fft
        
        # Initialize ONNX runtime session
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = num_thread
        session_opts.intra_op_num_threads = num_thread
        
        self.session = ort.InferenceSession(
            onnx_model_path,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )
        
        # Get input name from ONNX model
        self.input_name = self.session.get_inputs()[0].name
        
        # Pre-compute window for ISTFT
        self.window = torch.hann_window(self.win_length)
    
    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Decode mel spectrogram to audio waveform.
        
        Args:
            mel: Mel spectrogram tensor with shape (B, n_mels, T) where
                 B is batch size, n_mels is number of mel bins (100), T is time frames.
        
        Returns:
            Audio waveform tensor with shape (B, 1, T_audio) where T_audio is the
            reconstructed audio length.
        """
        # Ensure mel is in the correct format: (B, n_mels, T)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # Add batch dimension
        
        # Convert to numpy for ONNX inference
        mel_np = mel.float().numpy()
        
        # Run ONNX inference
        outputs = self.session.run(
            None,
            {self.input_name: mel_np}
        )
        
        # Extract outputs: mag, cos_phase, sin_phase
        mag, cos_phase, sin_phase = outputs
        
        # Convert to torch tensors
        mag = torch.from_numpy(mag)
        cos_phase = torch.from_numpy(cos_phase)
        sin_phase = torch.from_numpy(sin_phase)
        
        # Reconstruct complex spectrogram: S = mag * (cos + i*sin)
        # ONNX outputs are in shape (B, n_bins, T), where n_bins = n_fft // 2 + 1
        complex_spec = mag * (cos_phase + 1j * sin_phase)
        
        # Apply ISTFT to reconstruct audio
        # complex_spec shape: (B, n_bins, T)
        # Output shape: (B, T_audio)
        audio = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
        )
        
        # Add channel dimension: (B, T_audio) -> (B, 1, T_audio)
        audio = audio.unsqueeze(1)
        
        return audio
    
    def eval(self):
        """Set to evaluation mode (for compatibility with nn.Module interface)."""
        pass
    