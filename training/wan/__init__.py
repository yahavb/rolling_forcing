from . import configs, distributed, modules

# Neuron/training: WanI2V and WanT2V are the full generation pipelines and pull in
# torchvision (via image2video/text2video), which is NOT installed in the training
# image and is NOT used by distillation (the trainer only needs wan.modules.* —
# t5/tokenizers/model/vae/causal_model, all torchvision-free). Import them lazily so
# `from wan.modules... import ...` and `import wan` don't drag torchvision in.
# Access still works: `wan.WanT2V` / `wan.WanI2V` trigger the import on first use.
def __getattr__(name):
    if name == "WanI2V":
        from .image2video import WanI2V
        return WanI2V
    if name == "WanT2V":
        from .text2video import WanT2V
        return WanT2V
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
