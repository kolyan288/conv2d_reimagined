import torch
import torch.nn as nn
from typing import Optional, Tuple
from common.conv2d_img2col import Conv2dImg2Col


class MagnitudePruner:
    def __init__(self, sparsity: float = 0.5, structured: bool = False):
        self.sparsity = sparsity
        self.structured = structured
        self.masks = {}

    def compute_mask_unstructured(self, weight: torch.Tensor, sparsity: float) -> torch.Tensor:
        abs_weight = torch.abs(weight)

        num_elements = weight.numel()
        num_prune = int(num_elements * sparsity)

        threshold = torch.kthvalue(abs_weight.view(-1), num_prune).values

        mask = (abs_weight > threshold).float()

        return mask

    def compute_mask_structured(self, weight: torch.Tensor, sparsity: float, dim: int = 0) -> torch.Tensor:
        if dim == 0:
            norm = torch.norm(weight.view(weight.size(0), -1), p=2, dim=1)
        elif dim == 1:
            norm = torch.norm(weight.view(weight.size(0), weight.size(1), -1), p=2, dim=(0, 2))
        else:
            raise ValueError(f"dim only 0 or 1, received {dim}")

        num_channels = norm.size(0)
        num_prune = int(num_channels * sparsity)

        if num_prune == 0:
            return torch.ones_like(weight)

        threshold = torch.kthvalue(norm, num_prune).values

        if dim == 0:
            mask = (norm > threshold).float().view(-1, 1, 1, 1)
            mask = mask.expand_as(weight)
        else:
            mask = (norm > threshold).float().view(1, -1, 1, 1)
            mask = mask.expand_as(weight)

        return mask

    def apply_pruning(self, model: nn.Module, layer_names: Optional[list] = None) -> nn.Module:
        self.masks = {}

        for name, module in model.named_modules():
            if layer_names is not None and name not in layer_names:
                continue

            if isinstance(module, (nn.Conv2d, Conv2dImg2Col, nn.Linear)) and hasattr(module, 'weight'):
                weight = module.weight.data

                if self.structured and isinstance(module, nn.Conv2d):
                    mask = self.compute_mask_structured(weight, self.sparsity, dim=0)
                else:
                    mask = self.compute_mask_unstructured(weight, self.sparsity)

                self.masks[name] = mask

                module.register_buffer('weight_mask', mask)
                module.weight.data = weight * mask

        return model

    def apply_mask(self, model: nn.Module) -> nn.Module:
        for name, module in model.named_modules():
            if hasattr(module, 'weight_mask'):
                module.weight.data = module.weight.data * module.weight_mask

        return model

    def get_sparsity(self, model: nn.Module, layer_names: Optional[list] = None) -> dict:
        sparsity_dict = {}
        total_zeros = 0
        total_params = 0

        for name, module in model.named_modules():
            if layer_names is not None and name not in layer_names:
                continue

            if isinstance(module, (nn.Conv2d, Conv2dImg2Col, nn.Linear)) and hasattr(module, 'weight'):
                weight = module.weight.data
                num_zeros = (weight == 0).sum().item()
                num_params = weight.numel()

                sparsity = num_zeros / num_params
                sparsity_dict[name] = {
                    'sparsity': sparsity,
                    'zeros': num_zeros,
                    'total': num_params
                }

                total_zeros += num_zeros
                total_params += num_params
        if total_params > 0:
            sparsity_dict['global'] = {
                'sparsity': total_zeros / total_params,
                'zeros': total_zeros,
                'total': total_params
            }

        return sparsity_dict

    def remove_pruning(self, model: nn.Module) -> nn.Module:

        for name, module in model.named_modules():
            if hasattr(module, 'weight_mask'):
                module.weight.data = module.weight.data * module.weight_mask
                delattr(module, 'weight_mask')

        return model


class IterativePruner:
    def __init__(self, 
                 initial_sparsity: float = 0.0,
                 final_sparsity: float = 0.9,
                 num_iterations: int = 14,
                 epochs_per_iteration: int = 3,
                 structured: bool = False):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.num_iterations = num_iterations
        self.epochs_per_iteration = epochs_per_iteration
        self.structured = structured
        self.current_iteration = 0
        self.current_sparsity = initial_sparsity
        self.epochs_count = 0

    def get_current_sparsity(self) -> float:
        if self.current_iteration >= self.num_iterations:
            return self.final_sparsity

        sparsity_range = self.final_sparsity - self.initial_sparsity
        step = max(sparsity_range / (((self.num_iterations + 1) / self.epochs_per_iteration)- 1), 0.0)

        current_sparsity = self.initial_sparsity + step * self.current_iteration
        current_sparsity = min(max(current_sparsity, 0.0), self.final_sparsity)
        return current_sparsity

    def prune_step(self, model: nn.Module, layer_names: Optional[list] = None) -> float:
        if self.epochs_count % self.epochs_per_iteration == 0:
            self.current_sparsity = self.get_current_sparsity()
            pruner = MagnitudePruner(sparsity=self.current_sparsity, structured=self.structured)
            pruner.apply_pruning(model, layer_names)
            self.current_iteration += 1
        
        self.epochs_count += 1

        return self.current_sparsity
