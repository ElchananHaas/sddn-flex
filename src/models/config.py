import argparse

class Config(argparse.Namespace):
    dataset: str
    k: int
    inner_dim: int
    num_blocks: int
    noise: float
    lr: float
    inout_dim: int
    split_threshold: float
    pass
