> 历史资料：正文保留归档前内容，路径、命令、指标和完成状态不作为当前保证。当前说明见[文档索引](../README.md)。

# Classifer Guidance Tutorial

## Create your own guidance function

1. Create ``models/guidance/<my_guidance>.py``

```python
def my_guidance_fn(x, t, cond, inputs) -> torch.Tensor:
    ...

    return reward
```

2. Add ``<my_guidance_fn>`` in ``models/guidance/guidance_wrapper.py``

```python
# models/guidance/guidance_wrapper.py

...

class GuidanceWrapper:
    def __init__(self):
        self._guidance_fns = [
            <my_guidance_1>,
            <my_guidance_2>,
            ...
            <my_guidance_N>
        ]

    def __call__(...):
        ...

...
```

3. Run ``scripts/sim_guidance_demo.sh``
4. Enjoy.