import threading

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import librosa
import numpy as np
import soundcard as sc
import soundfile as sf


DELTA_WIDTH = 9
HOP_LENGTH = 512
MIN_SAMPLE_MS = 100
MSE_FRAME_LENGTH = 2048


def get_db(y):
    # Calculate MSE energy per frame
    mse = librosa.feature.rmse(y=y,
                               frame_length=MSE_FRAME_LENGTH,
                               hop_length=HOP_LENGTH) ** 2

    # Convert power to decibels
    return librosa.power_to_db(mse.squeeze(), ref=-100)


def is_silent(y, threshold_db):
    return np.max(get_db(y)) < threshold_db


def pca(features, components=6):
    """Dimension reduction via Principal Component Analysis (PCA)"""
    pca = PCA(n_components=components)
    transformed = pca.fit(features).transform(features)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(transformed)
    return scaler.transform(transformed), pca, scaler


def mfcc_features(y, sr, n_mels=128, n_mfcc=13):
    """Extract MFCCs (Mel-Frequency Cepstral Coefficients)"""
    # Analyze only first second
    y = y[0:sr]

    # Calculate MFCCs (Mel-Frequency Cepstral Coefficients)
    mel_spectrum = librosa.feature.melspectrogram(y,
                                                  sr=sr,
                                                  n_mels=n_mels)
    log_spectrum = librosa.amplitude_to_db(mel_spectrum,
                                           ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_spectrum,
                                sr=sr,
                                n_mfcc=n_mfcc)

    if mfcc.shape[-1] < DELTA_WIDTH:
        raise RuntimeError('MFCC vector does not contain enough time steps')

    if not mfcc.any():
        return np.zeros(n_mfcc * 3)

    # Standardize feature for equal variance
    delta_mfcc = librosa.feature.delta(mfcc, width=DELTA_WIDTH)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2, width=DELTA_WIDTH)
    feature_vector = np.concatenate((
        np.mean(mfcc, 1),
        np.mean(delta_mfcc, 1),
        np.mean(delta2_mfcc, 1)))
    feature_vector = (
        feature_vector - np.mean(feature_vector)
    ) / np.std(feature_vector)

    return feature_vector


def slice_audio(y, onsets, sr=44100, offset=0, top_db=5,
                trim=True):
    frames = []

    min_frames = (sr // 1000) * MIN_SAMPLE_MS

    for i in range(len(onsets) - 1):
        # Take audio from onset start to next onset
        onset_start = onsets[i] * HOP_LENGTH
        onset_end = onsets[i + 1] * HOP_LENGTH

        # Ignore too short samples
        if onset_end - onset_start < min_frames:
            continue

        if trim:
            # Trim silence
            y_trim, trim_indexes = librosa.effects.trim(
                y[onset_start:onset_end], ref=np.mean, top_db=top_db)

            if len(y_trim) < min_frames:
                continue

            # Set new slice relative to onset position
            start = onset_start + trim_indexes[0]
            end = onset_start + trim_indexes[1]

            if end - start < min_frames:
                continue
        else:
            start = onset_start
            end = onset_end

        frames.append([y[start:end], start + offset, end + offset])

    return frames


def detect_onsets(y, sr=44100, db_threshold=-50):
    # Get the frame->beat strength profile
    onset_envelope = librosa.onset.onset_strength(y=y,
                                                  sr=sr,
                                                  hop_length=HOP_LENGTH,
                                                  aggregate=np.median)

    # Locate note onset events
    onsets = librosa.onset.onset_detect(y=y,
                                        sr=sr,
                                        onset_envelope=onset_envelope,
                                        hop_length=HOP_LENGTH,
                                        backtrack=True)

    # Convert frames to time
    times = librosa.frames_to_time(np.arange(len(onset_envelope)),
                                   sr=sr,
                                   hop_length=HOP_LENGTH)

    # Filter out onsets which signals are too low
    db = get_db(y)
    onsets = [o for o in onsets if db[o] > db_threshold]

    return onsets, times[onsets]


def all_inputs():
    return sc.all_microphones()


def all_outputs():
    return sc.all_speakers()


class AudioIO():
    """Records data from an audio source."""

    def __init__(self, ctx, samplerate=44100, buffersize=1024,
                 device_in=0, channel_in=0,
                 device_out=0, channel_out=0,
                 volume=1.0):
        self.ctx = ctx
        self.samplerate = samplerate
        self.buffersize = buffersize

        # Parameters to be changed during live performance
        self._volume = volume

        # Select audio devices and its channels
        inputs = all_inputs()
        outputs = all_outputs()

        if len(inputs) - 1 < device_in:
            raise IndexError(
                'No input device with index {} given'.format(device_in))

        if len(outputs) - 1 < device_out:
            raise IndexError(
                'No output device with index {} given'.format(device_out))

        self._input = sc.get_microphone(inputs[device_in].id)
        self._output = sc.get_speaker(outputs[device_out].id)

        if self._input.channels - 1 < channel_in:
            raise IndexError(
                'No input channel with index {} given'.format(channel_in))

        if self._output.channels - 1 < channel_out:
            raise IndexError(
                'No output channel with index {} given'.format(channel_out))

        ctx.log('Input device "{}" @ channel {}'.format(
            self._input.name, channel_in))
        ctx.log('Output device "{}" @ channel {}'.format(
            self._output.name, channel_out))

        self._input_ch = channel_in
        self._output_ch = channel_out

        # Prepare reading thread
        self._lock = threading.Lock()
        self._frames = np.array([])
        self.is_running = False

        # Prepare writing thread
        self._buffer = np.array([])

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        with self._lock:
            self._volume = value

    def read_frames(self):
        with self._lock:
            frames = self._frames
            self.flush()
            return frames

    def flush(self):
        self._frames = np.array([])

    def record(self):
        with self._input.recorder(self.samplerate,
                                  channels=[self._input_ch]) as mic:
            while self.is_running:
                data = mic.record(self.buffersize)
                if len(data) > 0:
                    # Reshape the vector as it looks different on Linux
                    data = np.reshape(data, self.buffersize)
                    try:
                        self._frames = np.concatenate((self._frames, data))
                    except ValueError:
                        self.ctx.elog(
                            'Something went wrong during audio recording!',
                            data.shape,
                            self._frames.shape)
                        continue

    def play_buffer(self):
        try:
            with self._output.player(self.samplerate,
                                     channels=[self._output_ch]) as speaker:
                speaker.play(self._buffer * self._volume)
        except TypeError:
            self.ctx.elog('Something went wrong during audio playback!')

    def play(self, wav_path):
        with self._lock:
            data, _ = sf.read(wav_path)
            self._buffer = data
        threading.Thread(target=self.play_buffer).start()

    def start(self):
        with self._lock:
            self.is_running = True
        threading.Thread(target=self.record).start()

    def stop(self):
        with self._lock:
            self.is_running = False
