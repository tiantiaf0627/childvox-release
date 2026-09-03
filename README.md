## ChildVox: A Speech, Audio, and Large Audio-Language Model Benchmark in Understanding and Characterizing Sound across Childhood

<p align="center">
  📄 <a href="https://arxiv.org/abs/2605.29257"><strong>[Preprint Paper]</strong></a> &nbsp;|&nbsp;
  🤗
  <a href="https://huggingface.co/collections/tiantiaf/childvox"><strong>[ChildVox Collection]</strong></a> &nbsp; | &nbsp;
  <a href="https://huggingface.co/collections/tiantiaf/childvox-whisper-large"><strong>[Whisper-Large v3 Models]</strong></a> &nbsp; | &nbsp;
  <a href="https://huggingface.co/collections/tiantiaf/childvox-whisper-base"><strong>[Whisper-Base Models]</strong></a> &nbsp; | &nbsp;
  <a href="https://huggingface.co/collections/tiantiaf/childvox-babyhubert"><strong>[BabyHuBERT Models]</strong></a> &nbsp;
</p>

#### This repo presents ChildVox, a benchmark for understanding and characterizing the diverse acoustic signals through which children communicate, using audio, speech, and large audio-language models.

Accepted to **EMNLP 2026 Main**.

Unlike prior child-speech work that focuses primarily on ASR, ChildVox follows the full developmental trajectory *from birth through school age*, covering **physiological sounds, non-linguistic vocalizations, canonical syllables, and spoken language**. ChildVox integrates **more than 20 sub-tasks across 17 child-centered audio and speech datasets** into a consistent evaluation protocol, and benchmarks self-supervised, ASR-oriented, and large audio-language models (LALMs) side by side.

Benchmark tasks span physiological sound classification (murmur, crackle, wheeze, respiratory condition), vocalization event classification (child sound events, cry cause), canonical syllable and vocal development classification, speech quality assessment (pronunciation, fluency, prosody, articulation, intelligibility, emotion), adult-child speaker diarization, and word- and phoneme-level ASR.

### Download Repo
```bash
git clone git@github.com:tiantiaf0627/childvox-release.git
```

### Installation
```bash
conda create -n childvox python=3.10
cd childvox-release
pip install -e .
```

### Quick Example - BabyHuBERT Child Vocal Development (Speech Maturity) Classification
```python
# Load libraries
import torch
import torch.nn.functional as F
from src.model.childvox.babyhubert_audio import BabyHuBERTWrapper

# Label List
maturity_list = [
    "Canonical",
    "Non-Canonical",
    "Crying",
    "Laughing",
    "Junk"
]

# Find device
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Load model from Huggingface
# We provide a model per cross-validation fold, and specify the fold from 1, 2, 3, 4, 5
model = BabyHuBERTWrapper.from_pretrained(
    "tiantiaf/childvox-speechmaturity-babyhubert", fold_idx=1
).to(device)
model.eval()

# Load data, here just zeros as the example
# The child vocalization segments used in training are short, so we cap the input at 1 second
# So you need to prepare your audio to a maximum of 1 second, 16kHz and mono channel
max_audio_length = 1 * 16000
data = torch.zeros([1, 16000]).float().to(device)[:, :max_audio_length]
logits, embeddings = model(data, return_feature=True)

# Probability and output
maturity_prob = F.softmax(logits, dim=1)
print(maturity_list[torch.argmax(maturity_prob).detach().cpu().item()])
```

`Canonical` denotes mature syllables containing a consonant-vowel transition, while `Non-Canonical` denotes immature vocalizations such as isolated vowels or consonants. `Junk` covers segments that are not child vocalizations (e.g., noise, adult speech, or unintelligible audio).

#### Given that ChildVox includes private and license-restricted corpora (e.g. NLS, ADOS2-Mod3, MyST), we only release models trained on publicly accessible datasets. Below are the models we currently put out.

### Whisper-Large v3 Models

| Model Name | Data | Pre-trained Model | Use LoRa | LoRa Rank Size | Max Audio Length | Output |
|---|---|---|---|---|---|---|
| [tiantiaf/childvox-circor-whisper-large](https://huggingface.co/tiantiaf/childvox-circor-whisper-large) | CirCor | whisper-large-v3 | Yes | 64 | 10s | Absent, Unknown, Present |
| [tiantiaf/childvox-speechmaturity-whisper-large](https://huggingface.co/tiantiaf/childvox-speechmaturity-whisper-large) | SpeechMaturity | whisper-large-v3 | Yes | 64 | 1s | Canonical, Non-Canonical, Crying, Laughing, Junk |
| [tiantiaf/childvox-babblecor-whisper-large](https://huggingface.co/tiantiaf/childvox-babblecor-whisper-large) | BabbleCor | whisper-large-v3 | Yes | 64 | 1s | Canonical, Non-Canonical, Crying, Laughing, Junk |
| [tiantiaf/childvox-ReCANVo-whisper-large](https://huggingface.co/tiantiaf/childvox-ReCANVo-whisper-large) | ReCANVo | whisper-large-v3 | Yes | 64 | 5s | Delighted, Dysregulated, Frustrated, Request, Self-talk, Social |
| [tiantiaf/childvox-percept_r-whisper-large](https://huggingface.co/tiantiaf/childvox-percept_r-whisper-large) | PERCEPT-R | whisper-large-v3 | Yes | 64 | 2s | Rhotic, Derhotic |
| [tiantiaf/childvox-speechocean762-accuracy-whisper-large](https://huggingface.co/tiantiaf/childvox-speechocean762-accuracy-whisper-large) | SpeechOcean762 | whisper-large-v3 | Yes | 64 | 10s | Poor or Understandable, Good, Excellent |
| [tiantiaf/childvox-speechocean762-prosody-whisper-large](https://huggingface.co/tiantiaf/childvox-speechocean762-prosody-whisper-large) | SpeechOcean762 | whisper-large-v3 | Yes | 64 | 10s | Poor intonation, Nearly correct intonation, Correct intonation |

### Whisper-Base Models

| Model Name | Data | Pre-trained Model | Use LoRa | LoRa Rank Size | Max Audio Length | Output |
|---|---|---|---|---|---|---|
| [tiantiaf/childvox-circor-whisper-base](https://huggingface.co/tiantiaf/childvox-circor-whisper-base) | CirCor | whisper-base | Yes | 64 | 10s | Absent, Unknown, Present |
| [tiantiaf/childvox-speechmaturity-whisper-base](https://huggingface.co/tiantiaf/childvox-speechmaturity-whisper-base) | SpeechMaturity | whisper-base | Yes | 64 | 1s | Canonical, Non-Canonical, Crying, Laughing, Junk |
| [tiantiaf/childvox-babblecor-whisper-base](https://huggingface.co/tiantiaf/childvox-babblecor-whisper-base) | BabbleCor | whisper-base | Yes | 64 | 1s | Canonical, Non-Canonical, Crying, Laughing, Junk |
| [tiantiaf/childvox-ReCANVo-whisper-base](https://huggingface.co/tiantiaf/childvox-ReCANVo-whisper-base) | ReCANVo | whisper-base | Yes | 64 | 5s | Delighted, Dysregulated, Frustrated, Request, Self-talk, Social |
| [tiantiaf/childvox-percept_r-whisper-base](https://huggingface.co/tiantiaf/childvox-percept_r-whisper-base) | PERCEPT-R | whisper-base | Yes | 64 | 2s | Rhotic, Derhotic |
| [tiantiaf/childvox-speechocean762-accuracy-whisper-base](https://huggingface.co/tiantiaf/childvox-speechocean762-accuracy-whisper-base) | SpeechOcean762 | whisper-base | Yes | 64 | 10s | Poor or Understandable, Good, Excellent |
| [tiantiaf/childvox-speechocean762-prosody-whisper-base](https://huggingface.co/tiantiaf/childvox-speechocean762-prosody-whisper-base) | SpeechOcean762 | whisper-base | Yes | 64 | 10s | Poor intonation, Nearly correct intonation, Correct intonation |

### BabyHuBERT Models

BabyHuBERT is fine-tuned **without LoRA**, using a learnable weighted combination of representations from all hidden layers.

| Model Name | Data | Pre-trained Model | Use LoRa | Max Audio Length | Output |
|---|---|---|---|---|---|
| [tiantiaf/childvox-circor-babyhubert](https://huggingface.co/tiantiaf/childvox-circor-babyhubert) | CirCor | BabyHuBERT | No | 10s | Absent, Unknown, Present |
| [tiantiaf/childvox-speechmaturity-babyhubert](https://huggingface.co/tiantiaf/childvox-speechmaturity-babyhubert) | SpeechMaturity | BabyHuBERT | No | 1s | Canonical, Non-Canonical, Crying, Laughing, Junk |
| [tiantiaf/childvox-babblecor-babyhubert](https://huggingface.co/tiantiaf/childvox-babblecor-babyhubert) | BabbleCor | BabyHuBERT | No | 1s | Canonical, Non-Canonical, Crying, Laughing, Junk |
| [tiantiaf/childvox-ReCANVo-babyhubert](https://huggingface.co/tiantiaf/childvox-ReCANVo-babyhubert) | ReCANVo | BabyHuBERT | No | 5s | Delighted, Dysregulated, Frustrated, Request, Self-talk, Social |
| [tiantiaf/childvox-percept_r-babyhubert](https://huggingface.co/tiantiaf/childvox-percept_r-babyhubert) | PERCEPT-R | BabyHuBERT | No | 2s | Rhotic, Derhotic |
| [tiantiaf/childvox-speechocean762-accuracy-babyhubert](https://huggingface.co/tiantiaf/childvox-speechocean762-accuracy-babyhubert) | SpeechOcean762 | BabyHuBERT | No | 10s | Poor or Understandable, Good, Excellent |
| [tiantiaf/childvox-speechocean762-prosody-babyhubert](https://huggingface.co/tiantiaf/childvox-speechocean762-prosody-babyhubert) | SpeechOcean762 | BabyHuBERT | No | 10s | Poor intonation, Nearly correct intonation, Correct intonation |

### Limitations

- **Language and demographic coverage.** Most speech-category datasets in ChildVox are English, and the ASR evaluation is restricted to the English subset, so conclusions may not generalize to children speaking other languages. Demographic factors (educational background, developmental status) are often undocumented across the source corpora.
- **Annotation variability.** Affective vocalization (ReCANVo), cry-cause (Donate-a-Cry), and canonical-syllable labeling (SpeechMaturity, BabbleCor) involve inherently subjective categories with inter-rater disagreement, so reported scores may reflect a ceiling imposed by annotation reliability.
- **Restricted set of foundation models.** ChildVox does not evaluate every recent open-source LALM (e.g. GAMA, SALMONN, Step-Audio, Kimi-Audio), and the proprietary comparison is limited to two Gemini Flash models in a zero-shot setting with a single prompting approach.

#### Responsible Use: Child speech and physiological sound data is highly sensitive. Users should respect the privacy and consent of the children and families whose recordings are processed, obtain approval from the appropriate ethics/IRB body, and adhere to the relevant laws and regulations in their jurisdictions when using ChildVox.

❌ **Out-of-Scope Use**
- Clinical or diagnostic applications (e.g., screening for developmental or language disorders)
- Individual-level developmental assessment without expert human review
- Surveillance
- Privacy-invasive applications
- No commercial use

## If you have any questions, please contact: Tiantian Feng (tiantiaf@usc.edu)

#### If you like our work or use the models in your work, kindly cite the following. We appreciate your recognition!
```
@article{feng2026childvox,
  title={ChildVox: A Speech, Audio, and Large Audio-Language Model Benchmark in Understanding and Characterizing Sound across Childhood},
  author={Feng, Tiantian and Xu, Anfeng and Shi, Xuan and Kommineni, Aditya and Siam, Shakhrul Iman and Micheletti, Megan and Shi, Zhonghao and Tager-Flusberg, Helen and Zhang, Mi and Perry, Lynn K. and Lord, Catherine and Messinger, Daniel and Narayanan, Shrikanth},
  journal={arXiv preprint arXiv:2605.29257},
  year={2026}
}
```