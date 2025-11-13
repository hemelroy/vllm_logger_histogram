#Logger for MoE expert routing decisions

import json
import os
from typing import Optional

import torch

import vllm
from vllm.logger import init_logger

logger = init_logger(__name__)


class MoELogger:
    # Enable by setting VLLM_LOG_MOE environment variable to a file path.
    # Logs will be written in JSONL format with a meta header followed by
    # per-token routing records.
    
    _instance: Optional['MoELogger'] = None
    _initialized: bool = False
    
    def __init__(self):
        """Initialize the MoE logger based on environment variable."""
        self.enabled = False
        self.log_file = None
        self.file_handle = None
        self.logged_layer = 0  # Default to logging layer 0
        self.request_counter = 0
        
        log_path = os.environ.get("VLLM_LOG_MOE", "")
        if log_path:
            try:
                self.log_file = log_path
                self.file_handle = open(log_path, "w", encoding="utf-8")
                self.enabled = True
                
                # Allow configuration of which layer to log
                layer_str = os.environ.get("VLLM_LOG_MOE_LAYER", "0")
                try:
                    self.logged_layer = int(layer_str)
                except ValueError:
                    logger.warning(
                        f"Invalid VLLM_LOG_MOE_LAYER value: {layer_str}. "
                        f"Using default layer 0."
                    )
                    self.logged_layer = 0
                
                logger.info(
                    f"MoE logging enabled. Writing to {log_path}, "
                    f"logging layer {self.logged_layer}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize MoE logger: {e}")
                self.enabled = False
    
    @classmethod
    def get_instance(cls) -> 'MoELogger':
        """Get or create the singleton MoELogger instance."""
        if cls._instance is None or not cls._initialized:
            cls._instance = MoELogger()
            cls._initialized = True
        return cls._instance
    
    def write_meta_header(
        self,
        model_id: str,
        top_k: int,
        num_experts: int,
        device: str,
        seed: Optional[int] = None,
    ):
        """
        Write the JSONL meta header with model and configuration info.
        
        Args:
            model_id: Model identifier
            top_k: Number of experts selected per token
            num_experts: Total number of experts in the model
            device: Device type (GPU, CPU, etc.)
            seed: Random seed if set
        """
        if not self.enabled or self.file_handle is None:
            return
        
        try:
            import torch
            torch_version = torch.__version__
        except:
            torch_version = "unknown"
        
        try:
            vllm_version = vllm.__version__
        except:
            vllm_version = "unknown"
        
        meta = {
            "type": "meta",
            "model_id": model_id,
            "vllm_version": vllm_version,
            "torch_version": torch_version,
            "device": device,
            "seed": seed,
            "layers_logged": [self.logged_layer],
            "top_k": top_k,
            "num_experts": num_experts,
        }
        
        try:
            self.file_handle.write(json.dumps(meta) + "\n")
            self.file_handle.flush()
        except Exception as e:
            logger.error(f"Failed to write MoE meta header: {e}")
    
    def log_routing(
        self,
        layer_idx: int,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        req_id: Optional[str] = None,
    ):
        """
        Log the routing decision for a batch of tokens.
        
        Args:
            layer_idx: Index of the MoE layer
            topk_ids: Tensor of shape [num_tokens, top_k] with expert IDs
            topk_weights: Tensor of shape [num_tokens, top_k] with expert weights
            req_id: Request identifier (optional)
        """
        if not self.enabled or self.file_handle is None:
            return
        
        # Only log the configured layer
        if layer_idx != self.logged_layer:
            return
        
        try:
            # Move to CPU and convert to numpy for serialization
            topk_ids_cpu = topk_ids.detach().cpu().tolist()
            topk_weights_cpu = topk_weights.detach().cpu().tolist()
            
            # Generate request ID if not provided
            if req_id is None:
                req_id = f"r{self.request_counter}"
                self.request_counter += 1
            
            # Write one record per token
            num_tokens = len(topk_ids_cpu)
            for token_idx in range(num_tokens):
                record = {
                    "type": "route",
                    "req_id": req_id,
                    "token_idx": token_idx,
                    "layer": layer_idx,
                    "topk_ids": topk_ids_cpu[token_idx],
                    "topk_weights": [
                        round(w, 6) for w in topk_weights_cpu[token_idx]
                    ],
                }
                self.file_handle.write(json.dumps(record) + "\n")
            
            # Flush after each batch to ensure data is written
            self.file_handle.flush()
            
        except Exception as e:
            logger.error(f"Failed to log MoE routing: {e}")
    
    def close(self):
        """Close the log file handle."""
        if self.file_handle is not None:
            try:
                self.file_handle.close()
            except Exception as e:
                logger.error(f"Failed to close MoE log file: {e}")
            finally:
                self.file_handle = None
                self.enabled = False
    
    def __del__(self):
        """Ensure file is closed on deletion."""
        self.close()
