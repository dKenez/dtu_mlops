import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")

model = models.resnet34().to(DEVICE)
inputs = torch.randn(5, 3, 224, 224).to(DEVICE)


with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    profile_memory=True,
    on_trace_ready=tensorboard_trace_handler("./log/resnet34"),
) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

prof.export_chrome_trace("trace.json")

# prof = profile(
#     activities=[
#         torch.profiler.ProfilerActivity.CPU,
#         torch.profiler.ProfilerActivity.CUDA,
#     ]
# )

# model = model.to("cpu")
# inputs = inputs.to("cpu")

# prof.start()
# model(inputs)
# prof.stop()

# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
