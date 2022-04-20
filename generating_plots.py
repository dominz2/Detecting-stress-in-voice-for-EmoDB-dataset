import numpy as np
import json
import librosa
import librosa.display
import matplotlib.pyplot as plt


class Sound:

    def __init__(self, path, frame_size=1024, hop_length=512, name="Unknown"):
        self.path = path
        self.sound, self.sr = librosa.load(path)
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.time = self.amplitude_envelope()[0]
        self.name = name

    def plot_amplitude(self):
        """Display plot of amplitude"""
        plt.figure(figsize=(15, 17))
        librosa.display.waveplot(self.sound, alpha=0.5)
        plt.title(self.name)
        plt.ylim((-1, 1))
        plt.show()

    def plot_amplitude_envelope(self):
        plt.figure(figsize=(15, 17))
        ae_sound = self.amplitude_envelope()[1]
        plt.plot(self.time, ae_sound, color="r")
        plt.title(self.name)
        plt.ylim((-1, 1))
        plt.show()

    def plot_spectral_centroid(self):
        plt.figure(figsize=(15, 17))
        sc_sound = self.spectral_centroid()
        plt.plot(self.time, sc_sound, color="r")
        plt.title(self.name)
        plt.show()

    def plot_band_width(self):
        plt.figure(figsize=(15, 17))
        bw_sound = self.band_width()
        plt.plot(self.time, bw_sound, color="r")
        plt.title(self.name)
        plt.show()

    def plot_root_mean_square_energy(self):
        plt.figure(figsize=(15, 17))
        rms_sound = self.root_mean_square_energy()
        plt.plot(self.time, rms_sound, color="r")
        plt.title(self.name)
        plt.show()

    def plot_zero_crossing_rate(self):
        plt.figure(figsize=(15, 17))
        zero_crossing_rate = self.zero_crossing_rate()
        plt.plot(self.time, zero_crossing_rate, color="r")
        plt.title(self.name)
        plt.show()

    def plot_magnitude_spectrum(self, f_ratio=1.0):
        plt.figure(figsize=(18, 5))
        frequency = np.linspace(0, self.sr, len(self.magnitude_spectrum()))
        num_frequency_bins = int(len(frequency) * f_ratio)
        plt.plot(frequency[:num_frequency_bins], self.magnitude_spectrum()[:num_frequency_bins])
        plt.xlabel("frequency [Hz]")
        plt.title(self.name)
        plt.show()

    def plot_MFCC(self, n_mfcc=13):
        mfccs = librosa.feature.mfcc(self.sound, n_mfcc=n_mfcc, sr=self.sr)
        # visualise mfccs
        plt.figure(figsize=(25, 10))
        librosa.display.specshow(mfccs, x_axis="time", sr=self.sr)
        plt.colorbar(format='%+2f')
        plt.show()

    def plot_mel_banks(self, n_mels=10):
        filter_banks = librosa.filters.mel(n_fft=self.frame_size, sr=self.sr, n_mels=n_mels)
        plt.figure(figsize=(25, 10))
        librosa.display.specshow(filter_banks, sr=self.sr, x_axis="linear")
        plt.colorbar(format="%+2.f")
        plt.show()

    def plot_mel_spectrogram(self, n_mels=10):
        mel_spectrogram = librosa.feature.melspectrogram(self.sound, sr=self.sr,
                                                         n_fft=self.frame_size, hop_length=self.hop_length,
                                                         n_mels=n_mels)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        plt.figure(figsize=(25, 10))
        librosa.display.specshow(log_mel_spectrogram, x_axis="time", y_axis="mel", sr=self.sr)
        plt.colorbar(format="%+2.f")
        plt.show()

    def plot_stft(self):
        """Visualising short time Furier transform"""
        stft = librosa.stft(self.sound, n_fft=self.frame_size, hop_length=self.hop_length)
        stft = librosa.power_to_db(np.abs(stft) ** 2)

        plt.figure(figsize=(25, 10))
        librosa.display.specshow(stft, sr=self.sr, hop_length=self.hop_length, x_axis="time", y_axis="log")
        plt.colorbar(format="%+2.f")
        plt.show()

    def amplitude_envelope(self):
        amplitude_envelope = []
        for i in range(0, len(self.sound), self.hop_length):
            current_frame_amplitude_envelope = max(self.sound[i:i + self.frame_size])
            amplitude_envelope.append(current_frame_amplitude_envelope)
        ae_sound = np.array(amplitude_envelope)
        frames = range(0, ae_sound.size)
        time = librosa.frames_to_time(frames, sr=22050, hop_length=self.hop_length)
        return time, ae_sound

    def spectral_centroid(self):
        sc_sound = librosa.feature.spectral_centroid(y=self.sound, sr=self.sr, n_fft=self.frame_size,
                                                     hop_length=self.hop_length)
        return sc_sound[0]

    def band_width(self):
        bw_sound = librosa.feature.spectral_bandwidth(y=self.sound, sr=self.sr, n_fft=self.frame_size,
                                                      hop_length=self.hop_length)[0]
        return bw_sound

    def root_mean_square_energy(self):
        rms_sound = librosa.feature.rms(y=self.sound, frame_length=self.frame_size,
                                        hop_length=self.hop_length)
        return rms_sound[0]

    def zero_crossing_rate(self):
        zcr_sound = librosa.feature.zero_crossing_rate(y=self.sound, frame_length=self.frame_size,
                                                       hop_length=self.hop_length)
        return zcr_sound[0]

    def magnitude_spectrum(self):
        ft = np.fft.fft(self.sound)
        magnitude_spectrum = np.abs(ft)
        return magnitude_spectrum


if __name__ == "__main__":
    DATA_PATH = "debussy.wav"

    sax = Sound(DATA_PATH)
    # sax.plot_amplitude()
    sax.plot_magnitude_spectrum(f_ratio=.1)
    plt.show()

