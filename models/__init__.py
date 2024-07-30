# from .wav2vec import ContrastWav2Vec2Model
from .wav2vec_tdnn import ContrastWav2Vec2Model
from .whisper import ConstrastWhisperEncoder

model_list = {
    'facebook/wav2vec2-base-960h': ContrastWav2Vec2Model,
    'facebook/wav2vec2-large-960h-lv60': ContrastWav2Vec2Model,
    'openai/whisper-small': ConstrastWhisperEncoder
}

def load_model(model_name):
    return model_list[model_name]
