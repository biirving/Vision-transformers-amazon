from transformers import AutoProcessor, AutoModel
from datasets import load_dataset
from datasets import load_dataset
import neuronperf.torch
import torch_neuron
import os
import torch
from torch import nn, tensor
from einops import repeat, rearrange

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")

class NeuronCompatibilityWrapper(nn.Module):
    def __init__(self, model):
        super(NeuronCompatibilityWrapper, self).__init__()
        self.model = model

    def forward(self, encoding):
        out = self.model(input_ids = encoding['input_ids'], attention_mask = encoding['attention_mask'], bbox= encoding['bbox'], pixel_values = encoding['pixel_values'])
        return out


layout_for_trace = NeuronCompatibilityWrapper(model)

dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
example = dataset[0]
image = example["image"]
words = example["tokens"]
boxes = example["bboxes"]

encoding = processor(image, words, boxes=boxes, return_tensors="pt")

model = model.eval()

pipeline_sizes = [1]
batch_sizes = [3, 4]

for ncp in pipeline_sizes:
    for bs in batch_sizes:
        model_file = f"layout_neuron_ncp{ncp}_bs{bs}.pt"
        encoding['input_ids'] = repeat(encoding['input_ids'], 'b h -> (x b) h', x = bs)
        encoding['bbox'] = repeat(encoding['bbox'], 'b h w -> (x b) h w', x = bs)
        encoding['pixel_values'] = repeat(encoding['pixel_values'], 'b c h w -> (x b) c h w', x = bs)
        encoding['attention_mask'] = repeat(encoding['attention_mask'], 'b w -> (x b) w', x = bs)
        for_trace_dict = {'input_ids':encoding['input_ids'], 'attention_mask':encoding['attention_mask'], 'bbox':encoding['bbox'],'pixel_values':encoding['pixel_values']}
        
        print(f"ncp: {ncp}  bs: {bs}")

        if not os.path.exists(model_file):
            print("Attempting model compilation")
            nmod = torch.neuron.trace(layout_for_trace, example_inputs=for_trace_dict, compiler_args=['--neuroncore-pipeline-cores', f"{ncp}"], strict=False)
            nmod.save(model_file)
            del(nmod) # we need to release the model from memory so it doesn't affect benchmarking later on
        else:
            print(f"Found previously compiled model. Skipping compilation\n")
    
