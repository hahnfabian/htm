import torch 


class TokenMergeBuffer:
    def __init__(self, original_tokens: torch.Tensor):
        """
        original_tokens: (B, N, D) tensor
        """
        assert original_tokens.ndim == 3, f"Expected (B, N, D), got {original_tokens.shape}"
        B, N, D = original_tokens.shape
        device = original_tokens.device
        dtype = original_tokens.dtype

        buffer_len = 2 * N - 1

        self.B, self.N, self.D = B, N, D
        self.buffer_len = buffer_len

        # Initialize buffer on same device
        self.buffer = torch.zeros((B, buffer_len, D), dtype=dtype, device=device)
        self.buffer[:, :N] = original_tokens

        self.active_mask = torch.zeros((B, buffer_len), dtype=torch.bool, device=device)
        self.active_mask[:, :N] = True

        self.merges = torch.zeros((B, buffer_len - N, 3), dtype=torch.long, device=device)
        self.merge_ptr = 0

        self.n_active_tokens = N
        self.n_total_tokens = N

        self.n_merge = 0

    def get_active_indices(self) -> torch.Tensor:
        """(B, n_active) tensor of active token indices"""
        active_idx = self.active_mask.nonzero(as_tuple=False)
        _, counts = torch.unique(active_idx[:, 0], return_counts=True)
        n_active = counts[0].item()
        return active_idx[:, 1].reshape(self.B, n_active)


    def get_active_tokens(self) -> torch.Tensor:
        """(B, n_active, D) tensor of active tokens"""
        idx = self.get_active_indices()  
        return self.buffer[torch.arange(self.B)[:, None], idx, :]

    def merge_batch(self, local_t1_idx: torch.Tensor, local_t2_idx: torch.Tensor, merged_tokens: torch.Tensor):
        """
        Perform merges for each batch element in one go.
        local_t1_idx, local_t2_idx: (B,) tensor of indices in active token list
        merged_tokens: (B, D)
        """
        assert merged_tokens.shape == (self.B, self.D)
        assert local_t1_idx.shape == local_t2_idx.shape == (self.B,)

        act_idx = self.get_active_indices()  # (B, n_active)
        t1_idx = act_idx[torch.arange(self.B), local_t1_idx]
        t2_idx = act_idx[torch.arange(self.B), local_t2_idx]
        merge_idx = torch.full((self.B,), self.n_total_tokens, device=merged_tokens.device, dtype=torch.long)

        self.buffer[torch.arange(self.B), merge_idx, :] = merged_tokens

        self.active_mask[torch.arange(self.B), merge_idx] = True
        self.active_mask[torch.arange(self.B), t1_idx] = False
        self.active_mask[torch.arange(self.B), t2_idx] = False

        self.merges[:, self.merge_ptr, :] = torch.stack([t1_idx, t2_idx, merge_idx], dim=1)
        self.merge_ptr = int(self.merge_ptr) + 1

        self.n_total_tokens += 1
        self.n_active_tokens -= 1

        self.n_merge += 1

    def get_merge_history(self) -> torch.Tensor:
        # return self.merges.clone()
        return self.merges[:, : self.merge_ptr, :].clone() # TODO: does this fix??

    def _get_active_count(self):
        return self.n_active_tokens
    
    def get_buffer(self):
        # Get a buffer copy 
        # ig bc we have the active mask we can just copy blindly 
        return self.buffer
    
    def get_active_mask(self):
        return self.active_mask

    def can_merge(self, max_depth: int) -> bool:
        """
        Returns True if merging should continue based on:
        - more than one active token remains
        - current merge count is below max_depth
        """
        return (self.n_active_tokens > 1) and (self.n_merge < max_depth)

  


class TokenUnmergeBuffer:

    def __init__(self, buffer: torch.Tensor, active_mask: torch.Tensor, n_original: int, merges: torch.Tensor):
        """
        buffer: (B, buffer_len, D) tensor — the full token buffer
        active_mask: (B, buffer_len) bool tensor — mask of active tokens
        n_original: int — number of original tokens (N)
        merges: (B, N-1, 3) tensor — same format as produced by TokenMergeBuffer.get_merge_history()
        """
        
        assert buffer.ndim == 3, f"Expected buffer of shape (B, buffer_len, D), got {buffer.shape}"
        assert active_mask.ndim == 2 and active_mask.dtype == torch.bool, f"Expected active_mask of shape (B, buffer_len) with dtype bool"
        assert merges.ndim == 3 and merges.shape[2] == 3, f"Expected merges of shape (B, *, 3), got {merges.shape}"

        B, buffer_len, D = buffer.shape
        device, dtype = buffer.device, buffer.dtype

        self.buffer = buffer.clone()
        self.active_mask = active_mask.clone()
        self.merges = merges.flip(dims=[1])  # Reverse merge order for unmerging

        self.B = B
        self.n_original = n_original
        self.buffer_len = buffer_len
        self.ptr = 0  

    def step_unmerge(self, t1_tokens: torch.Tensor, t2_tokens: torch.Tensor):
        """
        Perform one unmerge step for all batches.
        t1_tokens, t2_tokens: (B, D)
        These are the two tokens produced by unmerging the current active merged token.
        """
        assert t1_tokens.shape == t2_tokens.shape == (self.B, self.buffer.shape[-1])

        t1_idx = self.merges[:, self.ptr, 0]
        t2_idx = self.merges[:, self.ptr, 1]
        merged_idx = self.merges[:, self.ptr, 2]

        self.buffer[torch.arange(self.B), t1_idx, :] = t1_tokens
        self.buffer[torch.arange(self.B), t2_idx, :] = t2_tokens

        self.active_mask[torch.arange(self.B), merged_idx] = False
        self.active_mask[torch.arange(self.B), t1_idx] = True
        self.active_mask[torch.arange(self.B), t2_idx] = True

        self.ptr += 1    

    def get_next_to_unmerge(self):
        merged_idx = self.merges[:, self.ptr, 2]
        return self.buffer[torch.arange(self.B), merged_idx, :]

    def get_original_tokens(self) -> torch.Tensor:
        """Return the first N tokens (after full unmerge)."""
        return self.buffer[:, :self.n_original, :]

    def is_done(self) -> bool:
        """Check if all unmerges are complete."""
        return self.ptr >= self.merges.shape[1]
    


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        """
        patience: how many epochs val loss can worsen/not improve
        min_delta: required improvement to reset counter
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def should_stop(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience