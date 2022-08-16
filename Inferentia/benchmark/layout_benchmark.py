# BEiT benchmarks

from transformers import AutoProcessor, AutoModel
import torch
from datasets import load_dataset
import neuronperf.torch
import torch_neuron
import os
from einops import repeat, rearrange


pipeline_sizes = [1]
batch_sizes = [4]


processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
example = dataset[0]
image = example["image"]
words = example["tokens"]
boxes = example["bboxes"]
encoding = processor(image, words, boxes=boxes, return_tensors="pt")


for ncp in pipeline_sizes:
    for bs in batch_sizes:
        model_file = f"layout_neuron_ncp{ncp}_bs{bs}.pt"
        report_file = f"layout_neuron_ncp{ncp}_bs{bs}_benchmark.csv"

        model_file = f"layout_neuron_ncp{ncp}_bs{bs}.pt"
        encoding['input_ids'] = repeat(encoding['input_ids'], 'b h -> (x b) h', x = bs)
        encoding['bbox'] = repeat(encoding['bbox'], 'b h w -> (x b) h w', x = bs)
        encoding['pixel_values'] = repeat(encoding['pixel_values'], 'b c h w -> (x b) c h w', x = bs)
        encoding['attention_mask'] = repeat(encoding['attention_mask'], 'b w -> (x b) w', x = bs)
        for_trace_dict = {'input_ids':encoding['input_ids'], 'attention_mask':encoding['attention_mask'], 'bbox':encoding['bbox'],'pixel_values':encoding['pixel_values']}

        print(f"ncp: {ncp}  bs: {bs}")

        if not os.path.exists(report_file):
            reports = neuronperf.torch.benchmark(model_filename=model_file, inputs=for_trace_dict, batch_sizes=[bs], pipeline_sizes=[ncp])
            neuronperf.print_reports(reports)
            neuronperf.write_csv(reports, report_file)
        else:
            print(f"Report file {report_file} already exists. Skipping this benchmark run.\n")
