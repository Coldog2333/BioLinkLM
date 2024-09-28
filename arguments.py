from dataclasses import dataclass, field
from distutils.util import strtobool
from typing import Optional
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    pretrained_save_dir: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default="decoder")
    model_max_length: int = field(default=None, metadata={"help": "Maximum sequence length."})


@dataclass
class DataArguments:
    data_root_dir: str = field(default=None, metadata={"help": "Directory of the dataset."})
    train_data_path: str = field(default=None, metadata={"help": "Path to the train data."})
    validation_data_path: str = field(default=None, metadata={"help": "Path to the valid data."})
    test_data_path: str = field(default=None, metadata={"help": "Path to the test data."})
    method: str = field(default=None, metadata={"help": "method. Options: [linked, standard]"})

    # downstream
    cache_file: str = field(default=None, metadata={"help": "Cache file path."})
    dataset_name_or_path: str = field(default="")
    truncation_strategy: Optional[str] = field(default=None)
    objective: Optional[str] = field(default="seq2seq")

    # lm
    prepend_bos: bool = field(default=True, metadata={"help": "Whether to prepend BOS token."})
    append_eos: bool = field(default=True, metadata={"help": "Whether to append EOS token."})
    extra_train_data_path: str = field(default=None, metadata={"help": "Path to the extra data in general domain."})
    mix_ratio: float = field(default=0.0, metadata={"help": "Ratio of the mixture of extra data in general domain."})
    mlm_probability: float = field(default=0.15, metadata={"help": "Probability of masking tokens for MLM."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    stage: str = field(default="train")
    cache_dir: Optional[str] = field(default=None)
    # optimizer
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for Adam optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for Adam optimizer"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    # scheduler
    num_cycles: float = field(default=0.5, metadata={"help": "Number of cycles for cosine scheduler."})

    train_from_scratch: bool = field(default=False, metadata={"help": "Whether to train from scratch."})
    resume_from_checkpoint: bool = field(default=False, metadata={"help": "Whether to resume from checkpoint."})
    curriculum_learning: bool = field(default=False, metadata={"help": "Whether to use curriculum learning."})
