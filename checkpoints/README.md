# Checkpoints and probability caches

GitHub limits files to 100 MB, so these are hosted on Hugging Face: https://huggingface.co/dwivediishivam/indoml26_track1

```bash
python checkpoints/download_checkpoints.py --caches    # 12 members' val/test probability caches (regenerate 995227)
python checkpoints/download_checkpoints.py --weights   # trained weights for inference on new audio
python checkpoints/download_checkpoints.py --all
```

| File (`checkpoints/`) | Member | Weights | SHA-256 |
|---|---|---|---|
| `s9000_weights_9000.pt` | WavLM-base+ SED, step 9,000 | final | `a3acdf3711e314eeb203c346c184255b6ec462ef91b5b3e8115714b8531300d4` |
| `s1600_weights_1600.pt` | WavLM-base+ SED, step 1,600 | final | `4ad8a99a114034e420fb5667b00c7a9f005607cc5865a2ddc6fcba669aeb3ff8` |
| `m1_atst_student_step0018000.pt` | ATST-F + mel-CNN | student | `22593e67928cd982e39e7449c9902253af850132800e4daa1bcedc5fe8e3b0fe` |
| `m4_breath_student_step0012000.pt` | WavLM-base+ breath specialist | student | `87a98690a0afc47e6314ca6990ef76e7399fc1e782e17f67c954f003ca443028` |
| `m1g_atst_gold_ema_step0009000.pt` | ATST-F, Gold continuation | EMA | `20b800abd61ba3af6c7cd08e8c0f47017b68b9aa9be04224908673f4b9f886cc` |
| `m1v_atst_val_ema_step0009000.pt` | ATST-F + validation natural | EMA | `dcc28aa5d5b18f26a3864a84975d791ad72a5ec11d58fe1b0bd117c14ec14f82` |
| `m2_beats_ema_step0007776.pt` | BEATs + mel-CNN | EMA | `bd8b426f68a468186966c916f7f3fd01742305d6ce5051d6262f8fdea25d23de` |
| `m3_wavlm_large_ema_step0009000.pt` | WavLM-large + mel-CNN | EMA | `01ea580186e65c705f5c0140482305b33fb745ece0bf657662d572abaab676bd` |
| `m5_w2vbert_student_step0010000.pt` | w2v-BERT 2.0 + mel-CNN | student | `75e770db43df5ca1c6535190bcfbd1665c6a3c0f8307a9f792b6671d601f7733` |

`caches/<member>/{val,blind}.npz` hold float16 [channels × frames] sigmoid probabilities at 20 ms for all 12 members,
with a `.json` sidecar recording the checkpoint, step, window/hop/floor and TTA shifts. `s9000` and `s1600` use the
74-channel legacy layout, which `t1/decode/fusion.py` converts.
