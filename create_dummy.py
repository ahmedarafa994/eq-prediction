import torch
import torchaudio

# Create a 4 second 1-channel dummy audio file
waveform = torch.randn(1, 22050 * 4)
torchaudio.save("dummy.wav", waveform, 22050)
print("Saved dummy.wav")
