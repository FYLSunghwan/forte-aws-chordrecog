import json
import logger
import logging
import mir_eval
import pretty_midi as pm
import youtube_dl
import torch
import librosa
import os
from model import *

idx2chord = ['C', 'C:min', 'C#', 'C#:min', 'D', 'D:min', 'D#', 'D#:min', 'E', 'E:min', 'F', 'F:min', 'F#',
             'F#:min', 'G', 'G:min', 'G#', 'G#:min', 'A', 'A:min', 'A#', 'A#:min', 'B', 'B:min', 'N']

root_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
quality_list = ['min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7', 'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4']

def idx2voca_chord():
    idx2voca_chord = {}
    idx2voca_chord[169] = 'N'
    idx2voca_chord[168] = 'X'
    for i in range(168):
        root = i // 14
        root = root_list[root]
        quality = i % 14
        quality = quality_list[quality]
        if i % 14 != 1:
            chord = root + ':' + quality
        else:
            chord = root
        idx2voca_chord[i] = chord
    return idx2voca_chord


logger = logging.getLogger(__name__)

def audio_file_to_features(audio_file, config):
    original_wav, sr = librosa.load(audio_file, sr=config.mp3['song_hz'], mono=True)
    currunt_sec_hz = 0
    while len(original_wav) > currunt_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len']:
        start_idx = int(currunt_sec_hz)
        end_idx = int(currunt_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len'])
        tmp = librosa.cqt(original_wav[start_idx:end_idx], sr=sr, n_bins=config.feature['n_bins'], bins_per_octave=config.feature['bins_per_octave'], hop_length=config.feature['hop_length'])
        if start_idx == 0:
            feature = tmp
        else:
            feature = np.concatenate((feature, tmp), axis=1)
        currunt_sec_hz = end_idx
    tmp = librosa.cqt(original_wav[currunt_sec_hz:], sr=sr, n_bins=config.feature['n_bins'], bins_per_octave=config.feature['bins_per_octave'], hop_length=config.feature['hop_length'])
    feature = np.concatenate((feature, tmp), axis=1)
    feature = np.log(np.abs(feature) + 1e-6)
    feature_per_second = config.mp3['inst_len'] / config.model['timestep']
    song_length_second = len(original_wav)/config.mp3['song_hz']
    return feature, feature_per_second, song_length_second


def download_yt(link):
    ydl_opts = {
      'format': 'bestaudio/best',
      'postprocessors': [{
          'key': 'FFmpegExtractAudio',
          'preferredcodec': 'mp3',
          'preferredquality': '192',
        }],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(link,download=False)
        if 'entries' in result:
            video = result['entries'][0]
            return 0
        else:
            video = result
            path = "{}-{}.mp3".format(video["title"],video['id'])
            ydl.download([link])
            path = path.replace('/', '_')
            return path

class ModelHandler(object):
    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.error = None
        self._context = None
        self.device = None
        self.model = None
        self.mean = None
        self.std = None
        self.voca = False
        self.initialized = False
        self.idx_to_chord = None
        self.audio_path = None
        self.n_timestep = None
        self.config = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.config = HParams.load("run_config.yaml")
        
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        self._context = context
        
        self.config.feature['large_voca'] = True
        self.config.model['num_chords'] = 170
        self.idx_to_chord = idx2voca_chord()
        
        self.model = BTC_model(config=self.config.model).to(self.device)
        checkpoint = torch.load(os.path.join(model_dir, "btc_model_large_voca.pt"))
        self.mean = checkpoint['mean']
        self.std = checkpoint['std']
        self.n_timestep = self.config.model['timestep']
        self.model.load_state_dict(checkpoint['model'])
        
        self.initialized = True
        
        logger.debug('Model loaded successfully')

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        Download mp3 from youtubel link
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """

        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        
        url = text['url']
        
        path = download_yt(url)
        return path
        

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Load mp3
        feature, feature_per_second, song_length_second = audio_file_to_features(model_input, self.config)
        logger.info("audio file loaded and feature computation success : %s" % model_input)

        # Majmin type chord recognition
        feature = feature.T
        feature = (feature - self.mean) / self.std
        time_unit = feature_per_second
        n_timestep = self.n_timestep

        num_pad = n_timestep - (feature.shape[0] % n_timestep)
        feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
        num_instance = feature.shape[0] // n_timestep

        start_time = 0.0
        lines = []
        with torch.no_grad():
            self.model.eval()
            feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(self.device)
            for t in range(num_instance):
                self_attn_output, _ = self.model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
                prediction, _ = self.model.output_layer(self_attn_output)
                prediction = prediction.squeeze()
                for i in range(n_timestep):
                    if t == 0 and i == 0:
                        prev_chord = prediction[i].item()
                        continue
                    if prediction[i].item() != prev_chord:
                        lines.append({
                            'start_time' : start_time,
                            'end_time' : time_unit * (n_timestep * t + i),
                            'chord_name' : self.idx_to_chord[prev_chord]
                        })
                        start_time = time_unit * (n_timestep * t + i)
                        prev_chord = prediction[i].item()
                    if t == num_instance - 1 and i + num_pad == n_timestep:
                        if start_time != time_unit * (n_timestep * t + i):
                            lines.append({
                                'start_time' : start_time,
                                'end_time' : time_unit * (n_timestep * t + i),
                                'chord_name' : self.idx_to_chord[prev_chord]
                            })
                        break
        lines = {"res" : lines}
        lines = json.dumps(lines)
        return [lines]

    def postprocess(self, inference_output):
        """
        Return predict result in batch.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        return inference_output

    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)

_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)


