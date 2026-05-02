Quick notes: running CPT training and post-run helpers

Run the training script (inside your virtualenv):

```bash
source /home/ala/myenv/bin/activate
cd /home/ala/TunisianDialogSystem
python llm/cpt/training/train.py
```

Run in tmux (recommended):

```bash
# open or attach existing session
tmux attach -t 0 || tmux new -s cpt
# create a new tmux window inside session
tmux new-window
# in the new window:
source /home/ala/myenv/bin/activate
python llm/cpt/training/train.py
# detach safely:
# Ctrl-b d
```

Run with nohup:

```bash
source /home/ala/myenv/bin/activate
nohup python llm/cpt/training/train.py > train.out 2>&1 &
```

Quick test after training:

```bash
python llm/cpt/training/quick_test.py --output_dir outputs/checkpoints/aya-expanse-8b-cpt-tunisian
```

Push adapter/checkpoints to Hugging Face (example):

```bash
export HF_TOKEN=hf_...your token...
python llm/cpt/training/push_adapter.py --local_dir outputs/checkpoints/aya-expanse-8b-cpt-tunisian --repo_id your-username/aya-expanse-8b-cpt-tunisian-adapter --push_full
```

Notes:
- `push_adapter.py` uses `huggingface_hub.upload_folder`. For very large uploads consider using `git lfs` + `huggingface-cli`.
- The training script already auto-detects and resumes from the latest checkpoint in the output folder.
