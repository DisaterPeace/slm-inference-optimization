"""Phase 3 - PagedAttention: managing the KV cache like OS virtual memory.

We implement the core data structure from scratch -- a block allocator + per
sequence block tables -- and measure how much KV-cache memory paging saves
versus the traditional "reserve max length per request" approach.

We use the REAL dimensions of Qwen2.5-0.5B so the memory numbers are authentic.

Two things this file demonstrates:
  1) MEMORY: a head-to-head of contiguous pre-allocation vs paged allocation
     for a realistic batch of variable-length requests.
  2) CORRECTNESS: storing K/V into scattered physical blocks via a block table
     and gathering them back in logical order -- proving the indexing works.
"""

from __future__ import annotations

import torch
from transformers import AutoConfig

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BLOCK_SIZE = 16   # tokens of KV stored per physical block (vLLM default is 16)


class KVConfig:
    """Per-token KV-cache size, derived from the real model config."""

    def __init__(self, model_id: str = MODEL_ID, trust_remote_code: bool = False):
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.num_layers = cfg.num_hidden_layers
        # Qwen uses Grouped-Query Attention: fewer KV heads than query heads.
        # The KV cache size depends on the KV heads, NOT the query heads.
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.dtype_bytes = 2  # fp16

    @property
    def bytes_per_token(self) -> int:
        # 2 = one K tensor + one V tensor.
        return (
            2
            * self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * self.dtype_bytes
        )

    @property
    def bytes_per_block(self) -> int:
        return self.bytes_per_token * BLOCK_SIZE


class BlockAllocator:
    """A fixed pool of physical KV blocks. Hands them out and takes them back.

    This is the heart of PagedAttention: a free list of interchangeable blocks,
    exactly like an OS managing physical memory frames.
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free: list[int] = list(range(num_blocks))

    def allocate(self) -> int:
        if not self.free:
            raise MemoryError("out of KV blocks")
        return self.free.pop()

    def free_block(self, block_id: int) -> None:
        self.free.append(block_id)

    @property
    def used(self) -> int:
        return self.num_blocks - len(self.free)


class Sequence:
    """One request. Owns a BLOCK TABLE: logical position -> physical block id.

    Blocks are allocated on demand as the sequence grows -- never up front.
    """

    def __init__(self, seq_id: int, allocator: BlockAllocator):
        self.seq_id = seq_id
        self.allocator = allocator
        self.block_table: list[int] = []
        self.length = 0

    def append_tokens(self, n: int) -> None:
        """Grow the sequence by n tokens, allocating new blocks only as needed."""
        for _ in range(n):
            # Need a new block only when the current last block is full.
            if self.length % BLOCK_SIZE == 0:
                self.block_table.append(self.allocator.allocate())
            self.length += 1

    def free(self) -> None:
        for block_id in self.block_table:
            self.allocator.free_block(block_id)
        self.block_table = []
        self.length = 0


def compare_memory(kv: KVConfig) -> None:
    """Contiguous pre-allocation vs paged allocation for a realistic batch."""
    # A realistic serving batch: many short replies, a few long ones.
    actual_lengths = [40, 55, 60, 80, 120, 150, 200, 300, 512, 900]
    max_model_len = 2048  # what a naive engine reserves per request

    n = len(actual_lengths)
    total_real_tokens = sum(actual_lengths)

    # --- Naive: reserve max_model_len for EVERY request ---
    naive_tokens = n * max_model_len
    naive_bytes = naive_tokens * kv.bytes_per_token

    # --- Paged: round each sequence UP to a whole number of blocks ---
    paged_blocks = sum(
        (length + BLOCK_SIZE - 1) // BLOCK_SIZE for length in actual_lengths
    )
    paged_bytes = paged_blocks * kv.bytes_per_block

    useful_bytes = total_real_tokens * kv.bytes_per_token

    print(f"Model KV size: {kv.bytes_per_token/1024:.1f} KB/token "
          f"({kv.num_layers} layers x {kv.num_kv_heads} KV heads x {kv.head_dim} dim, fp16)")
    print(f"Block size: {BLOCK_SIZE} tokens = {kv.bytes_per_block/1024:.1f} KB/block\n")

    print(f"Batch: {n} requests, actual lengths {actual_lengths}")
    print(f"Useful KV (what's truly needed): {useful_bytes/1024**2:7.1f} MB\n")

    print(f"{'scheme':<22} | {'KV memory':>11} | {'wasted':>16}")
    print("-" * 56)
    print(f"{'naive (reserve 2048)':<22} | {naive_bytes/1024**2:8.1f} MB | "
          f"{(naive_bytes-useful_bytes)/naive_bytes*100:5.1f}% "
          f"({(naive_bytes-useful_bytes)/1024**2:.0f} MB)")
    print(f"{'paged (block=16)':<22} | {paged_bytes/1024**2:8.1f} MB | "
          f"{(paged_bytes-useful_bytes)/paged_bytes*100:5.1f}% "
          f"({(paged_bytes-useful_bytes)/1024**2:.0f} MB)")
    print(f"\nPaging uses {naive_bytes/paged_bytes:.1f}x less KV memory for the same batch.")

    # How many such requests fit in a fixed 4 GB KV budget?
    budget = 4 * 1024**3
    avg_len = total_real_tokens / n
    naive_fit = int(budget // (max_model_len * kv.bytes_per_token))
    paged_fit = int(budget // (((avg_len + BLOCK_SIZE - 1) // BLOCK_SIZE) * kv.bytes_per_block))
    print(f"In a 4 GB KV budget: naive fits ~{naive_fit} requests, "
          f"paging fits ~{paged_fit} (avg len {avg_len:.0f}).")


def verify_correctness(kv: KVConfig) -> None:
    """Prove the block-table indexing actually works: scatter K/V into physical
    blocks, then gather them back in logical order and check we get the original.
    """
    allocator = BlockAllocator(num_blocks=64)
    seq = Sequence(seq_id=0, allocator=allocator)

    seq_len = 40  # spans 3 blocks (16+16+8)
    feat = kv.num_kv_heads * kv.head_dim

    # A physical KV pool: [num_blocks, block_size, feature_dim]. In a real engine
    # this is one big GPU tensor; sequences index slices of it via their tables.
    pool = torch.zeros(allocator.num_blocks, BLOCK_SIZE, feat)
    original = torch.randn(seq_len, feat)  # fake K (or V) for one layer

    seq.append_tokens(seq_len)
    # Scatter each logical token into its physical block slot.
    for logical_pos in range(seq_len):
        block_id = seq.block_table[logical_pos // BLOCK_SIZE]
        slot = logical_pos % BLOCK_SIZE
        pool[block_id, slot] = original[logical_pos]

    # Gather back in logical order using the block table.
    gathered = torch.stack([
        pool[seq.block_table[pos // BLOCK_SIZE], pos % BLOCK_SIZE]
        for pos in range(seq_len)
    ])

    ok = torch.allclose(gathered, original)
    print(f"\nCorrectness: scattered {seq_len} tokens across blocks "
          f"{seq.block_table}, gathered back -> {'MATCH' if ok else 'MISMATCH'}")


def main() -> None:
    kv = KVConfig()
    compare_memory(kv)
    verify_correctness(kv)


if __name__ == "__main__":
    main()
