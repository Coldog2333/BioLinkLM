import copy
import os
from typing import Dict, Union, Optional
import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import ConcatDataset
torch.backends.cuda.matmul.allow_tf32 = True

from tqdm import tqdm
from transformers import Trainer
import transformers
from transformers.trainer_utils import SchedulerType
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION

from utils.data import (
    LMDataset,
    LMIndexedDataset,
    LMDataCollator
)
from pipeline.transformers_pipeline import (
    Pipeline,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)


def make_language_modeling_data_module(
    tokenizer,
    train_data_path=None, valid_data_path=None, test_data_path=None,
    prepend_bos=True, append_eos=True,
    train_dataset_type="LMIndexedDataset",
):
    if train_data_path is None:
        train_dataset = None
    else:
        if train_dataset_type == "LMIndexedDataset":
            ## Dataset from processed data
            if isinstance(train_data_path, list) and len(train_data_path) > 1:
                train_dataset = ConcatDataset(
                    [LMIndexedDataset(path, tokenizer) for path in train_data_path]
                )

            elif isinstance(train_data_path, list) and len(train_data_path) == 1:
                train_dataset = LMIndexedDataset(train_data_path[0], tokenizer)

            elif isinstance(train_data_path, str):
                train_dataset = LMIndexedDataset(train_data_path, tokenizer)

            else:
                raise ValueError(f"train_data_path should be str or list, but got {type(train_data_path)}")
        else:
            ## Dataset from files
            train_dataset = LMDataset(
                train_data_path, tokenizer, prepend_bos=prepend_bos, append_eos=append_eos,
            )

    valid_dataset = LMDataset(
        valid_data_path, tokenizer, prepend_bos=prepend_bos, append_eos=append_eos,
    ) if valid_data_path is not None else None

    test_dataset = LMDataset(
        test_data_path, tokenizer, prepend_bos=prepend_bos, append_eos=append_eos,
    ) if test_data_path is not None else None

    data_collator = LMDataCollator(tokenizer=tokenizer, is_training=True)
    data_collator_for_inference = LMDataCollator(tokenizer=tokenizer, is_training=False)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        test_dataset=test_dataset,
        data_collator=data_collator,
        data_collator_for_inference=data_collator_for_inference
    )


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    num_cycles: float = 0.5,
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of cycles for the scheduler. Only used by the `SchedulerType.COSINE` scheduler.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT or name == SchedulerType.REDUCE_ON_PLATEAU:
        return schedule_func(optimizer)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    if name == SchedulerType.INVERSE_SQRT:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    if name == SchedulerType.COSINE:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, num_cycles=num_cycles)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


class BetterTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                num_cycles=self.args.num_cycles,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler


class PretrainPipeline(Pipeline):
    def __init__(self, model_args, data_args, training_args):
        super().__init__(model_args, data_args, training_args)

    def prepare_data(self):
        ##### Prepare dataset #####
        data_module = make_language_modeling_data_module(
            tokenizer=self.tokenizer,
            train_data_path=self.data_args.train_data_path,
            valid_data_path=self.data_args.validation_data_path,
            test_data_path=self.data_args.test_data_path,
            prepend_bos=self.data_args.prepend_bos,
            append_eos=self.data_args.append_eos,
        )
        return data_module

    def prepare_test_data(self) -> Dict:
        ##### Prepare dataset #####
        data_module = make_language_modeling_data_module(
            tokenizer=self.tokenizer,
            test_data_path=self.data_args.test_data_path,
            prepend_bos=self.data_args.prepend_bos,
            append_eos=self.data_args.append_eos,
        )
        return data_module

    def train(self):
        data_module = self.prepare_data()

        ##### Prepare trainer #####
        trainer = BetterTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            train_dataset=data_module['train_dataset'],
            eval_dataset=data_module['eval_dataset'],
            data_collator=data_module['data_collator'],
        )

        ##### Train #####
        if self.training_args.resume_from_checkpoint:
            print('Resuming from checkpoint...')
        trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)

        ##### Save #####
        trainer.save_model(os.path.join(self.training_args.output_dir, 'hf_model'))
        # safe_save_model_for_hf_trainer(
        #     trainer=trainer,
        #     output_dir=os.path.join(training_args.output_dir, 'hf_model')
        # )

        ##### Test #####
        test_results = trainer.evaluate(eval_dataset=data_module['test_dataset'])
        if trainer.args.should_save:
            print(test_results)

    def evaluate(self, dataloader, note, dump_file=None, dataset_name=None, *args, **kwargs):
        return self.evaluate_ppl(dataloader, note)

    def evaluate_ppl(self, dataloader, note=''):
        self.model.eval()
        if self.accelerator.is_local_main_process:
            print(note)

        loss = 0.
        with torch.no_grad():
            for i, inputs in enumerate(tqdm(dataloader, desc='Evaluating', total=len(dataloader))):
                outputs = self.model(**inputs)
                loss_tmp = self.accelerator.gather(outputs.loss).mean()
                loss += loss_tmp.item()

        if self.accelerator.is_local_main_process:
            print(f'Loss: {loss / len(dataloader):.4f}')
            print(f'PPL: {np.exp(loss / len(dataloader)):.4f}')


if __name__ == '__main__':
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    try:
        if os.environ['LOCAL_RANK'] == '0':
            print(model_args, data_args, training_args)
    except:
        print(model_args, data_args, training_args)

    data_args.train_data_path = data_args.train_data_path.split(',')

    pipeline = PretrainPipeline(model_args, data_args, training_args)

    if training_args.stage == 'train':
        pipeline.train()

    elif training_args.stage == 'test':
        pipeline.test()

    else:
        raise NotImplementedError(f'Unexpected stage: {training_args.stage}')
