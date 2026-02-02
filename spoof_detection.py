import torch
import librosa
import numpy as np
import os
from model.AASIST import Model
from utils import create_optimizer
import soundfile as sf

# -------------------------
# Config (matches training)
# -------------------------
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "model\weights\AASIST-L.pth"


# -------------------------
# Load Audio
# -------------------------
# def load_audio(path, sr=16000, max_len=64600):

#     wav, _ = librosa.load(path, sr=sr)

#     # Pad / cut to fixed length
#     if len(wav) < max_len:
#         wav = np.pad(wav, (0, max_len - len(wav)))
#     else:
#         wav = wav[:max_len]

#     return wav


def load_audio(path, sr=16000, max_len=64600):
    
    
    os.makedirs("processed_wav", exist_ok=True)

    filename = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join("processed_wav", filename + "_proc.wav")
    
    try:
        # Fast path (WAV, FLAC)
        wav, file_sr = sf.read(path, dtype="float32")

        if file_sr != sr:
            wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)

        

    except RuntimeError:
        # Fallback (MP3, M4A)
        wav, _ = librosa.load(path, sr=sr)

    # Convert stereo → mono
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    # Pad / cut
    if len(wav) < max_len:
        wav = np.pad(wav, (0, max_len - len(wav)))
    else:
        wav = wav[:max_len]

    # Save processed audio
    sf.write(out_path, wav, sr)
    
    return wav




def load_model():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # MODEL_PATH = "pretrained/AASIST.pth"

    # Official AASIST pretrained config
    
    
    d_args_l = {
        "nb_samp": 64600,
        "first_conv": 128,

        "filts": [
            70,
            [1, 32],
            [32, 32],
            [32, 24],
            [24, 24],
            [24, 24],
            [24, 24],
        ],

        "gat_dims": [24, 32],   # ⚠️ MUST be 24
        "pool_ratios": [0.5, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0],
    }
    
    d_args = {
            "nb_samp": 64600,
            "first_conv": 128,

            "filts": [
                70,
                [1, 32],
                [32, 32],
                [32, 64],
                [64, 64],
            ],

            "gat_dims": [64, 32],

            "pool_ratios": [0.5, 0.7, 0.5, 0.5],

            "temperatures": [2.0, 2.0, 100.0, 100.0],
    }


    # Build model
    model = Model(d_args_l)   # ✅ Will match now

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Some checkpoints wrap state_dict
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)   # ✅ Will match now

    model.to(DEVICE)
    model.eval()

    return model


# -------------------------
# Predict
# -------------------------
def predict(model, audio_path):

    wav = load_audio(audio_path)

    x = torch.tensor(wav, dtype=torch.float32)
    x = x.unsqueeze(0).to(DEVICE)  # [1, T]

    with torch.no_grad():
        out, _ = model(x)     # unpack tuple (logits, features)
        prob = torch.softmax(out, dim=1)


    spoof_prob = prob[0][1].item()
    genuine_prob = prob[0][0].item()

    return genuine_prob, spoof_prob


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print("Usage: python infer_single.py your_audio.wav")
        exit(1)

    audio_file = sys.argv[1]

    model = load_model()

    genuine, spoof = predict(model, audio_file)

    print("\n===== RESULT =====")
    print(f"Genuine (Human): {genuine:.4f}")
    print(f"Spoof (AI/Fake): {spoof:.4f}")

    if spoof > genuine:
        print("Prediction: 🚨 SPOOF / AI-GENERATED")
    else:
        print("Prediction: ✅ REAL HUMAN VOICE")
