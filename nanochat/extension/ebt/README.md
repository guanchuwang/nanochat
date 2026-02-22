# NanoEBT

This is the first version of the EBT extended to NanoChat dataset.

### Setup environment

```bash
cd ../../../
bash runs/install.sh
cd nanochat/extension/ebt
```

### Dataset preparation

```bash
cd ../../../
bash runs/dataset_prepare.sh
cd nanochat/extension/ebt
```

### Setting wandb logger

Sign up a free account on [wandb.ai](https://wandb.ai/site), generate an API key, and set it in the '../../../runs/ebt_s1.sh' as follows:

```bash
export WANDB_API_KEY=<Your API Key>
```

### Training

```bash
cd ../../../
bash runs/ebt_s1.sh
cd nanochat/extension/ebt
```

You can check the training logs from the wandb dashboard at `https://wandb.ai/<Your account>/nlp_pretrain`.







