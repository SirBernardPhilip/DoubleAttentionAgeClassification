#!/bin/bash
#SBATCH -p veu             # Partition to submit to
#SBATCH --mem=30G      # Max CPU Memory
#SBATCH --gres=gpu:1


python train_softmax.py --window_size 1 --train_labels_path "/home/usuaris/veu/david.linde/features/ca_es_en/train.lst" --validation_file "/home/usuaris/veu/david.linde/features/ca_es_en/dev.lst" --front_end "resnet" --kernel_size 512 --criterion "CrossEntropyWeighted"  --model_name "resnet" --out_dir "./models/ca_es_en_losses_combv3_w"