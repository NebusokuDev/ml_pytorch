import torch
import torch_directml
dml = torch_directml.device()

tensor1 = torch.tensor([1]).to(dml) # Note that dml is a variable, not a string!
tensor2 = torch.tensor([2]).to(dml)

dml_algebra = tensor1 + tensor2

print(dml_algebra.item())