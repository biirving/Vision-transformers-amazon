from transformers import AutoProcessor, AutoModel
from datasets import load_dataset
import numpy as np
import time
import torch
from einops import repeat, rearrange
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__)

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")

dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
example = dataset[0]
image = example["image"]
words = example["tokens"]
boxes = example["bboxes"]


encoding = processor(image, words, boxes=boxes, return_tensors="pt")

# put both the input and the model on GPU
encoding.to(device)
model.to(device)


batch_size = [1, 2, 3, 4]


# pricing information: update this from ec2 - on demand pricing https://aws.amazon.com/ec2/pricing/on-demand/
# price as of August 8, 2022
g5_hourly_rate = 1.006



columns = ['model name', 'batch size','latency_p0', 'latency_p50', 'latency_p90', 'latency_p95', 'latency_p99', 'latency_p100', 'throughput average', 'throughput peak', 'cost per 1 m instances']
benchmarkDataFrame = pd.DataFrame(columns = columns)


index = 0
for batch in batch_size:
    encoding['input_ids'] = repeat(encoding['input_ids'], 'b h -> (x b) h', x = batch)
    encoding['bbox'] = repeat(encoding['bbox'], 'b h w -> (x b) h w', x = batch)
    encoding['pixel_values'] = repeat(encoding['pixel_values'], 'b c h w -> (x b) c h w', x = batch)
    encoding['attention_mask'] = repeat(encoding['attention_mask'], 'b w -> (x b) w', x = batch)

    latencies = []
    throughputs = []


    for u in range(1000):
        start = time.perf_counter()
        model(**encoding)
        torch.cuda.synchronize(device)
        end = time.perf_counter() - start
        a = end * 1000
        throughput = batch / end 
        latencies.append(a)
        throughputs.append(throughput)
        print(u)

    latencies = np.array(latencies)
    throughputs = np.array(throughputs)

    latencies_ordered = np.sort(latencies, axis = -1)

    latency_p100 = np.max(np.percentile(latencies_ordered, 100, -1))
    latency_p99 = np.max(np.percentile(latencies_ordered, 99, -1))
    latency_p95 = np.max(np.percentile(latencies_ordered, 95, -1))
    latency_p90 = np.max(np.percentile(latencies_ordered, 90, -1))
    latency_p50 = np.max(np.percentile(latencies_ordered, 50, -1))
    latency_p0 = np.max(np.percentile(latencies_ordered, 0, -1))

    throughput_average = np.average(throughputs, axis = -1)
    throughputs_peak = np.max(throughputs)

    # cost per 1m instances:
    time_for_1m_inferences_in_seconds = 1000000 / throughput_average
    in_hours = time_for_1m_inferences_in_seconds / 60 / 60
    cost = in_hours * inf_x_large_hourly_rate

    print('latency_p0: {:.6f}ms'.format(latency_p0))
    print('latency_p50: {:.6f}ms'.format(latency_p50))
    print('latency_p90: {:.6f}ms'.format(latency_p90))
    print('latency_p95: {:.6f}ms'.format(latency_p95))
    print('latency_p99: {:.6f}ms'.format(latency_p99))
    print('latency_100: {:.6f}ms'.format(latency_p100))
    print('throughput_average: {:.6f}'.format(throughput_average))
    print('throughput_peak: {:.6f}'.format(throughputs_peak))
    print('Cost per 1m instances: {:.6f}'.format(cost))


    benchmarkDataFrame.loc[index] = ['layoutLMv3', batch, latency_p0, latency_p50, latency_p90, latency_p95, latency_p99, latency_p100, throughput_average, throughputs_peak, cost]
    index += 1



benchmarkDataFrame.to_csv('layoutLMv3Benchmark.csv', index = False)


