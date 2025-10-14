"""Random state management for reproducible training."""

import random
from typing import Dict, Any
import numpy as np
import torch


def save_rng_state() -> Dict[str, Any]:
    """
    Capture current RNG states for Python, NumPy, PyTorch CPU and CUDA.
    
    Returns:
        Dictionary containing all RNG states
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state().cpu().to(dtype=torch.uint8),
    }
    
    # Save CUDA RNG states if available
    if torch.cuda.is_available():
        cuda_states = []
        for device_idx in range(torch.cuda.device_count()):
            device_state = torch.cuda.get_rng_state(device_idx)
            # Move to CPU and ensure uint8 dtype
            device_state = device_state.cpu().to(dtype=torch.uint8)
            cuda_states.append(device_state)
        state['cuda'] = cuda_states
    
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """
    Restore RNG states for Python, NumPy, PyTorch CPU and CUDA.
    
    Args:
        state: Dictionary containing RNG states (as returned by save_rng_state)
    """
    # Restore Python state
    if 'python' in state:
        random.setstate(state['python'])
    
    # Restore NumPy state
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    
    # Restore PyTorch CPU state
    if 'torch' in state:
        torch_state = state['torch']
        if isinstance(torch_state, torch.Tensor):
            if torch_state.device.type != 'cpu':
                torch_state = torch_state.cpu()
            if torch_state.dtype != torch.uint8:
                torch_state = torch_state.to(dtype=torch.uint8)
        else:
            torch_state = torch.as_tensor(torch_state, dtype=torch.uint8)
        torch.set_rng_state(torch_state)
    
    # Restore CUDA states if available
    if 'cuda' in state and torch.cuda.is_available():
        cuda_states = state['cuda']
        for device_idx, device_state in enumerate(cuda_states):
            if device_idx < torch.cuda.device_count():
                # Ensure tensor is on CPU and has uint8 dtype before setting
                if device_state.is_cuda:
                    device_state = device_state.cpu()
                device_state = device_state.to(dtype=torch.uint8)
                torch.cuda.set_rng_state(device_state, device_idx)
