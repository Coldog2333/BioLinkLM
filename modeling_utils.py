from typing import Dict
import torch
import transformers


WEIGHTS_NAME = "pytorch_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"


def get_n_model_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    return tokenizer, model


def smart_fsdp_transformer_layer_cls_to_wrap(model, fsdp=None):
    if fsdp is None:
        return None

    model_class_name = model.__class__.__name__
    mapper = {
        'LlamaForCausalLM': 'LlamaDecoderLayer',
        'OPTModel': 'OPTDecoder',
        'GPT2Model': 'GPT2Block',
    }

    fsdp_transformer_layer_cls_to_wrap = mapper.get(model_class_name, None)

    return fsdp_transformer_layer_cls_to_wrap
