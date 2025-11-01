from time import perf_counter
import numpy as np
import torch
from torch.profiler import profile, record_function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def latency_cpu(model, test_input=None, warmup_n=10, benchmark_n=100):

    times = []
    model.to("cpu")
    test_input = test_input.to("cpu") if test_input is not None else None

    if test_input is not None:
        assert test_input.device == torch.device("cpu")
        with torch.no_grad():
            for i in range(warmup_n):
                model(test_input)
    else:
        with torch.no_grad():
            for i in range(warmup_n):
                model()

    if test_input is not None:
        print(
            "Start CPU benchmark with input shape:", test_input.shape, test_input.device
        )
        for i in range(benchmark_n):
            t_0 = perf_counter()
            model(test_input)
            t_1 = perf_counter()
            times.append(t_1 - t_0)
        times = np.asarray(times) * 1000
        mean_ms = times.mean()
        std_ms = times.std()
    else:
        for i in range(benchmark_n):
            t_0 = perf_counter()
            _ = model()
            t_1 = perf_counter()
            times.append(t_1 - t_0)

        times = np.asarray(times) * 1000
        mean_ms = times.mean()
        std_ms = times.std()

    print(f"{mean_ms:.3f}ms +- {std_ms:.3f}ms")
    return mean_ms, std_ms


def latency_cpu_profiler(model, test_input, warmup_n=10, benchmark_n=100):
    # ProfilerActivity.CPU, ProfilerActivity.CUDA
    model.to("cpu")
    test_input = test_input.to("cpu") if test_input is not None else None

    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            # torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        use_cuda=False,
        # schedule=schedule(wait=1, warmup=warmup_n, active=30, repeat=10),
    ) as prof:
        with record_function("model_inference"): 
            model(test_input)
        # for i in range(benchmark_n):
        #     with torch.no_grad():
        #         model(test_input)
        #     prof.step()

    return prof


def latency_gpu_event(model, example_inputs, warmup=50, repeat=300):
    model.to(device)
    example_inputs = example_inputs.to(device)

    model.eval()
    latency = []
    for _ in range(warmup):
        model(example_inputs)

    for i in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(example_inputs)
        end.record()
        torch.cuda.synchronize()
        latency.append(start.elapsed_time(end))

    latency = torch.tensor(latency)
    return latency.mean().item(), latency.std().item()


def latency_gpu(model, test_input=None, warmup_n=10, benchmark_n=100):
    times = []
    model.to(device)
    test_input = test_input.to(device) if test_input is not None else None

    if test_input is not None:
        assert test_input.device != "cpu"
        with torch.no_grad():
            for i in range(warmup_n):
                model(test_input)
    else:
        with torch.no_grad():
            for i in range(warmup_n):
                model()

    if test_input is not None:
        print(
            "Start GPU benchmark with input shape:", test_input.shape, test_input.device
        )
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        for i in range(benchmark_n):
            t_0 = perf_counter()
            model(test_input)
            torch.cuda.synchronize()
            t_1 = perf_counter()
            times.append(t_1 - t_0)

        times = np.asarray(times) * 1000
        mean_ms = times.mean()
        std_ms = times.std()
    else:
        torch.cuda.synchronize()

        for i in range(benchmark_n):
            t_0 = perf_counter()
            _ = model()
            torch.cuda.synchronize()
            t_1 = perf_counter()
            times.append(t_1 - t_0)

        times = np.asarray(times) * 1000
        mean_ms = times.mean()
        std_ms = times.std()

    print(f"{mean_ms:.3f}ms +- {std_ms:.3f}ms")
    torch.cuda.empty_cache()
    return mean_ms, std_ms

