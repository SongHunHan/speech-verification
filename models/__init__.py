from .wav2vec import ContrastWav2Vec2Model
from .wav2vec_tdnn import ContrastWav2Vec2TDNNModel
from .whisper import ConstrastWhisperEncoder

model_list = {
    # 'facebook/wav2vec2-base-960h': ContrastWav2Vec2Model,
    'facebook/wav2vec2-base-960h': ContrastWav2Vec2TDNNModel,
    'openai/whisper-small': ConstrastWhisperEncoder
}

def load_model(model_name):
    return model_list[model_name]
