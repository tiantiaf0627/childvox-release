import torch
import logging
import torchaudio
import sys, os, pdb
import torch.nn.functional as F

from pathlib import Path

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model', 'childvox'))

from whisper_audio import WhisperWrapper
from babyhubert_audio import BabyHuBERTWrapper

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


if __name__ == '__main__':

    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    # Define the model
    # Note that ensemble across folds yields better performance than a single fold
    model = BabyHuBERTWrapper.from_pretrained(
        "tiantiaf/childvox-speechmaturity-babyhubert", fold_idx=1
    ).to(device)
    model.eval()

    # Labels recovered from the pushed config — no need to hardcode the ordering
    label_list = model.label_list

    # Our training data uses 1-second segments, so prepare your audio as
    # a maximum of 1 second, 16kHz and mono channel
    max_audio_length = 1 * 16000
    data = torch.zeros([1, 160000]).float().to(device)[:, :max_audio_length]

    # data, sr = torchaudio.load("your_audio.wav")
    # data = data.float().to(device)[:, :max_audio_length]

    logits = model(data)

    # Probability over the five vocal-maturity classes
    maturity_prob = F.softmax(logits, dim=1)
    print(label_list[torch.argmax(maturity_prob).detach().cpu().item()])
    print(maturity_prob.detach().cpu().numpy())