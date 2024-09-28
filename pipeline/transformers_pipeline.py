import os
from dataclasses import dataclass, field
from typing import Optional, Dict
import torch
torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, AutoConfig,
)
from transformers import Trainer

from modeling_utils import (
    safe_save_model_for_hf_trainer,
    smart_tokenizer_and_embedding_resize,
)
from arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from llama_factory.llmtuner.extras.logging import get_logger
from llama_factory.llmtuner.extras.packages import is_flash_attn2_available



logger = get_logger(__name__)


def is_main_process():
    try:
        return int(os.environ["LOCAL_RANK"]) == 0
    except:
        return True


class Pipeline:
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        self.prepare_model()
        self.prepare_tokenizer()

    def prepare_model(self):
        ##### Prepare model & tokenizer #####
        model_config = AutoConfig.from_pretrained(self.model_args.model_name_or_path)
        model_config.use_cache = False

        if self.model_args.model_max_length is not None and self.model_args.model_max_length > model_config.max_position_embeddings:
            # model_config.max_length = self.model_args.model_max_length
            # model_config.max_position_embeddings = self.model_args.model_max_length
            rope_scaling = {
                "factor": self.model_args.model_max_length / model_config.max_position_embeddings,
                "type": "linear",
            }
            model_config.rope_scaling = rope_scaling

            if is_main_process():
                logger.info(f"Interpolated RoPE with {rope_scaling['type']} scaling factor={rope_scaling['factor']}")

        if getattr(model_config, "model_type", None) == "llama":
            if is_flash_attn2_available():
                model_config._flash_attn_2_enabled = True
                if is_main_process():
                    logger.info("Using FlashAttention-2 for faster training and inference.")
            else:
                if is_main_process():
                    logger.warning("FlashAttention-2 is not installed.")

        else:
            if is_main_process():
                logger.warning("Current model does not support FlashAttention.")

        if self.training_args.train_from_scratch:
            self.model = AutoModelForCausalLM.from_config(model_config)

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                config=model_config
            )
            if is_main_process():
                logger.info("Loaded pretrained model with dtype: %s", getattr(model_config, "torch_dtype", None))


    def prepare_tokenizer(self):
        if self.model_args.model_name_or_path == "openlm-research/open_llama_3b_v2":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path, use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)

        if self.model_args.model_max_length is None:
            pass
        else:
            if self.tokenizer.model_max_length != self.model_args.model_max_length:
                if is_main_process():
                    logger.warning(
                        f"Warning: This model has default model_max_length={self.tokenizer.model_max_length}, but it is set to {self.model_args.model_max_length} now."
                    )
                self.tokenizer.model_max_length = self.model_args.model_max_length
            self.tokenizer.model_max_length = self.model_args.model_max_length

        # if self.tokenizer.model_max_length > self.model_args.model_max_length:
        #     print(f'Warning: This model has longer model_max_length={self.tokenizer.model_max_length}, but it is set to {self.model_args.model_max_length} now.')
        #
        # self.tokenizer.model_max_length = min(self.model_args.model_max_length, self.tokenizer.model_max_length)

        ## If pad_token is not set, set it as eos_token. It is acceptable because pad token will not be used in training.
        ## No! It is no acceptable since it will be used for constructing the attention mask.
        # if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        #     self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        #     assert self.tokenizer.pad_token is not None and self.tokenizer.pad_token_id is not None

        if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id is None:
            self.tokenizer, self.model = smart_tokenizer_and_embedding_resize(
                special_tokens_dict={'pad_token': '<|pad|>'},
                tokenizer=self.tokenizer,
                model=self.model,
            )

    def prepare_data(self, *args, **kwargs) -> Dict:
        raise NotImplementedError

    def prepare_test_data(self, *args, **kwargs) -> Dict:
        return self.prepare_data(*args, **kwargs)   # Default to be the same as training data

    def train(self):
        data_module = self.prepare_data()

        ##### Prepare trainer #####
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            train_dataset=data_module['train_dataset'],
            eval_dataset=data_module['eval_dataset'],
            data_collator=data_module['data_collator'],
        )

        ##### Train #####
        trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)

        ##### Save #####
        safe_save_model_for_hf_trainer(
            trainer=trainer,
            output_dir=os.path.join(self.training_args.output_dir, 'hf_model')
        )

    def evaluate(self, dataloader, note, dump_file=None, dataset_name=None, *args, **kwargs):
        raise NotImplementedError

    def test(self):
        self.accelerator = Accelerator(device_placement=True)
        self.accelerator.free_memory()  # Free all lingering references

        data_module = self.prepare_test_data()

        test_dataset = data_module["test_dataset"]
        data_collator_for_inference = data_module["data_collator_for_inference"]
        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.training_args.per_device_eval_batch_size,
            num_workers=self.training_args.dataloader_num_workers,
            collate_fn=data_collator_for_inference,
            drop_last=False
        )

        self.model, self.test_dataloader = self.accelerator.prepare(
            self.model, self.test_dataloader
        )

        self.evaluate(
            dataloader=self.test_dataloader,
            note='=== Final Test ===',
            dump_file=self.data_args.cache_file,
            dataset_name=self.data_args.dataset_name
        )
