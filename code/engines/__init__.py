"""Training / evaluation loops (kept OUT of the run scripts)."""
from .finetune import train_one_epoch, evaluate
from .pretrain import train_one_epoch as train_one_epoch_pretrain
from .visualize import run_attention_vis
from .waveform import evaluate_waveforms
# Stage-3 supervised waveform training reuses the generic supervised loop with
# a waveform-regression criterion (model -> [B, T], target -> [B, T]).
from .finetune import train_one_epoch as train_one_epoch_waveform
