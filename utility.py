import torch 
import click

@click.command()
def cuda_mem():
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

