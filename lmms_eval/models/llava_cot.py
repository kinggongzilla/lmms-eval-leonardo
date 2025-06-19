import copy
import json
import logging
import math
import re
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import transformers
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from tqdm import tqdm
from transformers import AutoProcessor, StoppingCriteria, StoppingCriteriaList, MllamaForConditionalGeneration

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav

warnings.filterwarnings("ignore")
eval_logger = logging.getLogger("lmms-eval")
torch.backends.cuda.matmul.allow_tf32 = True

@register_model("llava_cot")
class Llava_COT(lmms):
    def __init__(
        self,
        pretrained: str = "Xkev/Llama-3.2V-11B-cot",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        beam_size: Optional[int] = 2,
        generation_type: Optional[str] = "single", # "stage",
        **kwargs,
    ):
        super().__init__()
        print("lmms-eval kwargs")
        print(kwargs)

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self._device = torch.device(device)
        self._rank = accelerator.local_process_index
        self._world_size = accelerator.num_processes
        self.device_map = "auto"

        self.processor = AutoProcessor.from_pretrained(pretrained)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        self.model.keep_ratio = kwargs["keep_ratio"]

        self.model.to(self._device)

        self.beam_size = beam_size
        self.batch_size_per_gpu = int(batch_size)
        self.generation_type = generation_type
        self._tokenizer = self.processor.tokenizer
        self.task_dict = {}  # Required for lmms-eval

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def device(self):
        return self._device

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def tok_encode(self, string, left_truncate_len=None, add_special_tokens=None):
        enc = self.tokenizer.encode(string, add_special_tokens=add_special_tokens or False)
        return enc[-left_truncate_len:] if left_truncate_len else enc

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def pad_sequence(self, input_ids, batch_first, padding_value):
        return torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []
        pbar = tqdm(total=len(requests), desc="Llava-COT Generating")

        for instance in requests:
            prompt, gen_kwargs, doc_to_visual, doc_id, task, split = instance.args
            print("prompt: ")
            print(prompt)
            visual = doc_to_visual(self.task_dict[task][split][doc_id])

            image = visual[0] if isinstance(visual, list) else visual
            try:
                if isinstance(image, str):
                    image = PIL.Image.open(image)
            except Exception as e:
                eval_logger.warning(f"Failed to open image {image}: {e}")
                results.append("<invalid image>")
                continue

            text = self._generate_cot(prompt, image)
            results.append(text)
            print("text from generate_until: ")
            print(text)
            pbar.update(1)

        pbar.close()
        return results

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        results = []
        for instance in requests:
            prompt, gen_kwargs, doc_to_visual, doc_to_text, doc_id, task, split = instance.args
            visual = doc_to_visual(self.task_dict[task][split][doc_id])
            image = visual[0] if isinstance(visual, list) else visual
            try:
                if isinstance(image, str):
                    image = PIL.Image.open(image)
            except Exception as e:
                eval_logger.warning(f"Failed to open image {image}: {e}")
                results.append("<invalid image>")
                continue

            conversation = []
            round_idx = 0
            while True:
                question = doc_to_text(self.task_dict[task][split][doc_id], previous_output=conversation, round_idx=round_idx)
                if question is None:
                    break
                response = self._generate_cot(question, image)
                conversation.append(response)
                round_idx += 1
            results.append("\n".join(conversation))
        return results

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Llava-COT does not currently support loglikelihood evaluation.")

    def _generate_cot(self, prompt, image):
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': prompt}
            ]
        }]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        print("Input Text:", input_text)  # Debugging print
        print("Input Text Type:", type(input_text))  # Debugging print
        print("Image Type:", type(image))  # Debugging print
        inputs = self.processor(image, input_text, return_tensors='pt').to(self.device)
        print("Inputs Type:", type(inputs))  # Debugging print
        print("Generation type: ", self.generation_type)
        if self.generation_type == "stage":
            return self._stage_beam(prompt, image, inputs)
        else:
            return self._single_pass(prompt, image, inputs)

    def _single_pass(self, prompt, image, inputs):
        output = self.model.generate(**inputs, max_new_tokens=1024)
        final_output = self.processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print("Final output single pass: ")
        
        match = re.search(r'<CONCLUSION>(.*?)</CONCLUSION>', final_output)
        if match:
            # Extract the content and remove spaces
            conclusion_content = match.group(1).replace(" ", "")
            final_output = conclusion_content
        else:
            raise ValueError('NO <CONCLUSION> tags found!') 
        print(final_output)
        return final_output

    def _stage_beam(self, prompt, image, inputs):
        stages = ['<SUMMARY>', '<CAPTION>', '<REASONING>', '<CONCLUSION>']
        ends = ['</SUMMARY>', '</CAPTION>', '</REASONING>', '</CONCLUSION>']

        # Keep the original input text (with <image> token)
        original_input_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
        input_ids = inputs['input_ids']
        initial_len = len(input_ids[0])

        for stage, end in zip(stages, ends):
            candidates = []
            for _ in range(self.beam_size):
                # Rebuild the full prompt including previously generated tokens
                current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
                input_data = self.processor(image, current_text, return_tensors='pt').to(self.device)

                kwargs = {
                    'stopping_criteria': StoppingCriteriaList([StopOnStrings([end], self.processor.tokenizer)]),
                    'do_sample': True,
                    'temperature': 0.6,
                    'top_p': 0.9,
                    'max_new_tokens': 1024
                }

                output = self.model.generate(**input_data, **kwargs)
                gen_text = self.processor.tokenizer.decode(output[0][initial_len:], skip_special_tokens=True)
                candidates.append((output[0], gen_text))

            # Pick the top candidate (you might change this to apply reranking logic later)
            selected = candidates[0][0]
            input_ids = selected.unsqueeze(0)
            initial_len = len(input_ids[0])  # Update for next stage

        final_output = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print("Final output stage beam: ")
        print(final_output)
        return final_output


class StopOnStrings(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return any(s in text for s in self.stop_strings)
