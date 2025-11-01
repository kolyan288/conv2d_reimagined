import torch
import torch.nn as nn
from typing import Optional, Tuple


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

            # TODO
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight'):
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

            # TODO
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight'):
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
                 num_iterations: int = 5,
                 structured: bool = False):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.num_iterations = num_iterations
        self.structured = structured
        self.current_iteration = 0

    def get_current_sparsity(self) -> float:
        if self.current_iteration >= self.num_iterations:
            return self.final_sparsity

        # TODO
        # Полиномиальный график (кубический)
        t = self.current_iteration / self.num_iterations
        sparsity = self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * (
            3 * t**2 - 2 * t**3
        )

        return sparsity

    def prune_step(self, model: nn.Module, layer_names: Optional[list] = None) -> Tuple[nn.Module, float]:
        current_sparsity = self.get_current_sparsity()

        pruner = MagnitudePruner(sparsity=current_sparsity, structured=self.structured)
        model = pruner.apply_pruning(model, layer_names)

        self.current_iteration += 1

        return model, current_sparsity


def fine_tune_pruned_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 10,
    device: str = 'cuda',
    pruner: Optional[MagnitudePruner] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> nn.Module:
    model = model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            if pruner is not None:
                pruner.apply_mask(model)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/(batch_idx+1):.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')

        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}] Summary: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')

    return model


def iterative_prune_and_finetune(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer_fn,
    final_sparsity: float = 0.9,
    num_iterations: int = 5,
    epochs_per_iteration: int = 5,
    device: str = 'cuda',
    layer_names: Optional[list] = None,
    structured: bool = False
) -> nn.Module:
    iterative_pruner = IterativePruner(
        initial_sparsity=0.0,
        final_sparsity=final_sparsity,
        num_iterations=num_iterations,
        structured=structured
    )

    model = model.to(device)

    for iteration in range(num_iterations):
        print(f'\n{"="*80}')
        print(f'Pruning Iteration {iteration + 1}/{num_iterations}')
        print(f'{"="*80}')

        model, current_sparsity = iterative_pruner.prune_step(model, layer_names)
        print(f'Target sparsity: {current_sparsity:.4f}')

        pruner = MagnitudePruner(sparsity=current_sparsity, structured=structured)
        pruner.masks = {name: module.weight_mask for name, module in model.named_modules() 
                        if hasattr(module, 'weight_mask')}
        sparsity_info = pruner.get_sparsity(model, layer_names)
        print(f"Actual global sparsity: {sparsity_info.get('global', {}).get('sparsity', 0):.4f}")

        print(f'\nFine-tuning for {epochs_per_iteration} epochs...')
        optimizer = optimizer_fn(model.parameters())
        model = fine_tune_pruned_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=epochs_per_iteration,
            device=device,
            pruner=pruner
        )

    return model


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = 'cuda'
) -> Tuple[float, float]:
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

    return test_loss, test_acc