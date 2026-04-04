"""
Domain-specific feature extractors for signal, text, and image data.
"""

import numpy as np

try:
    from scipy import signal
    from scipy.stats import entropy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def extract_signal_features(audio_waveform, sr=22050):
    """Extract domain-specific features from audio/signal data."""
    features = {}
    features["rms_energy"] = np.sqrt(np.mean(audio_waveform ** 2))
    features["zero_crossing_rate"] = np.mean(np.diff(np.sign(audio_waveform)) != 0)
    features["peak_to_peak"] = np.max(audio_waveform) - np.min(audio_waveform)
    features["crest_factor"] = np.max(np.abs(audio_waveform)) / (features["rms_energy"] + 1e-10)

    if HAS_SCIPY:
        freqs, psd = signal.welch(audio_waveform, sr)
        features["spectral_centroid"] = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
        features["spectral_bandwidth"] = np.sqrt(
            np.sum((freqs - features["spectral_centroid"]) ** 2 * psd) / (np.sum(psd) + 1e-10)
        )
        features["spectral_entropy"] = entropy(psd + 1e-10)
        features["dominant_frequency"] = freqs[np.argmax(psd)]
    return features


def extract_text_features(texts, max_tfidf_features=5000):
    """Extract features from text data.

    Returns dict with tfidf, char_tfidf, and stats arrays.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(
        max_features=max_tfidf_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_features = tfidf.fit_transform(texts)

    char_tfidf = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), max_features=1000
    )
    char_features = char_tfidf.fit_transform(texts)

    stats = []
    for text in texts:
        words = text.split()
        n_words = len(words) if words else 0
        stats.append([
            len(text),
            n_words,
            len(set(words)) / n_words if n_words else 0,
            np.mean([len(w) for w in words]) if words else 0,
            text.count("!"),
            text.count("?"),
        ])
    return {
        "tfidf": tfidf_features,
        "char_tfidf": char_features,
        "stats": np.array(stats),
    }


def extract_image_features(image):
    """Extract hand-crafted features from images.

    Image should be a 2D or 3D numpy array.
    """
    from skimage.feature import hog, local_binary_pattern
    from skimage.color import rgb2gray

    if len(image.shape) == 3:
        gray = rgb2gray(image)
    else:
        gray = image
    if gray.max() > 1.0:
        gray = gray / 255.0

    features = {}
    hog_features = hog(
        gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)
    )
    features["hog"] = hog_features
    lbp = local_binary_pattern(gray, P=24, R=3, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7
    features["lbp_hist"] = hist
    features["mean"] = np.mean(gray)
    features["std"] = np.std(gray)
    edges = np.abs(np.diff(gray, axis=0)).sum() + np.abs(np.diff(gray, axis=1)).sum()
    features["edge_density"] = edges / gray.size
    return features
