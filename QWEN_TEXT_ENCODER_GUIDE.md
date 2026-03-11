# Using Qwen3-VL-2B as Text Encoder for DPO Training

This guide explains how to use Qwen3-VL-2B as a custom text encoder in the DPO training pipeline while keeping Wan2.2-T2V-A14B-Diffusers for video generation.

## Architecture Overview

The training setup consists of:
- **Video Generation**: Wan2.2-T2V-A14B-Diffusers (VAE + Transformer)
- **Text Encoding**: Qwen3-VL-2B-Instruct
- **Embedding Projection**: Learned projection layer to map Qwen embeddings to Wan2.2's expected dimension

## Why Use Qwen3-VL-2B?

Qwen3-VL-2B offers several advantages:
1. **Advanced Vision-Language Understanding**: Better text comprehension for video prompts
2. **Multimodal Capabilities**: Can leverage both text and visual context
3. **Efficient**: Only 2B parameters, making it memory-efficient
4. **State-of-the-art**: Latest model from the Qwen family (released 2025)

## Usage

### Basic Command

```bash
python azure/train_dpo_azure.py \
  --data /path/to/pairwise_preferences.json \
  --videos-dir /path/to/videos \
  --output-dir ./outputs \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --text-encoder Qwen/Qwen3-VL-2B-Instruct \
  --batch-size 1 \
  --grad-accum 4 \
  --epochs 10 \
  --lr 1e-6 \
  --beta 0.1
```

### Key Arguments

- `--model`: The video generation model (default: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`)
- `--text-encoder`: Custom text encoder model (e.g., `Qwen/Qwen3-VL-2B-Instruct`)
  - If not specified, uses the default text encoder from the Wan2.2 pipeline

### What Gets Trained?

When using Qwen3-VL as text encoder:
1. **Wan2.2 Transformer**: ✅ Trained (policy network for DPO)
2. **Text Embedding Projector**: ✅ Trained (maps Qwen embeddings to Wan2.2 dimension)
3. **Qwen3-VL**: ❌ Frozen (used only for text encoding)
4. **VAE**: ❌ Frozen (used for video encoding)

## Technical Details

### Embedding Projection

Qwen3-VL-2B has a hidden dimension of 896, while Wan2.2 expects embeddings of dimension ~4096. A learned linear projection layer handles this transformation:

```python
class TextEmbeddingProjector(nn.Module):
    def __init__(self, qwen_dim=896, wan_dim=4096):
        super().__init__()
        self.projection = nn.Linear(qwen_dim, wan_dim)
```

### Model Checkpoints

When saving checkpoints, the following files are saved:
- `checkpoint-N/`: Policy transformer checkpoint
- `checkpoint-N/text_projector.pt`: Text projection layer weights

To load a checkpoint for inference, you'll need both the transformer and the projector.

## Available Qwen3-VL Models

You can use any of these Qwen3-VL models as text encoders:

| Model | Size | Use Case |
|-------|------|----------|
| `Qwen/Qwen3-VL-2B-Instruct` | 2B | General purpose (Recommended) |
| `Qwen/Qwen3-VL-2B-Thinking` | 2B | Enhanced reasoning |
| `Qwen/Qwen3-VL-Embedding-2B` | 2B | Optimized for embeddings |

## Memory Requirements

Using Qwen3-VL-2B adds approximately:
- **Model**: ~4GB GPU memory (in bfloat16)
- **Projector**: <10MB (896 × 4096 parameters)

Total memory requirements:
- Wan2.2-T2V-A14B: ~28GB
- Qwen3-VL-2B: ~4GB
- Training overhead: ~8-16GB
- **Total**: ~40-48GB VRAM (requires A100 40GB or similar)

## Troubleshooting

### Issue: "transformers version too old"
Qwen3-VL requires `transformers>=4.57.0`. Update with:
```bash
pip install --upgrade transformers
```

### Issue: "Out of memory"
- Reduce `--batch-size` to 1
- Increase `--grad-accum` for effective batch size
- Use `--n-frames 8` instead of 16 to reduce video memory

### Issue: "Embedding dimension mismatch"
The projection layer should handle this automatically. If issues persist, check that the correct hidden dimensions are being detected from the model configs.

## Future Enhancements

Possible improvements:
1. **Multi-layer projection**: Use MLP instead of linear layer
2. **Fine-tune Qwen**: Allow optional fine-tuning of Qwen's last layers
3. **Vision integration**: Use Qwen's vision encoder to score video frames
4. **Attention alignment**: Add cross-attention between Qwen and Wan2.2

## References

- [Qwen3-VL Model Card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [Wan2.2-T2V-A14B-Diffusers Model Card](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
