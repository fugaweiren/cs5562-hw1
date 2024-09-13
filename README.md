# CS5562-HW1

This is the first homework assignment for CS5562 Trustworthy ML. Please clone this repo or download all the Python files and proceed with the instructions on the PDF. 

## Q3 Generate Adversarial Images
To run 1 epsilon value:
```
python launch_resnet_attack.py --batch_num 20 --batch_size 100 --eps 0.1 --results gen_data 2>&1 | tee logs/eps_01.txt
```

To run multiple epsilon values:
```
python launch_resnet_attack.py --batch_num 20 --batch_size 100 --eps 0 --results eps_02 2>&1 | tee logs/eps_00.txt &&
python launch_resnet_attack.py --batch_num 20 --batch_size 100 --eps 0.1 --results eps_01 2>&1 | tee logs/eps_01.txt &&
python launch_resnet_attack.py --batch_num 20 --batch_size 100 --eps 0.2 --results eps_02 2>&1 | tee logs/eps_02.txt &&
...
```

To generate adversarial dataset for finetuning (trainset):
```
python launch_resnet_attack.py --batch_num 100 --batch_size 100 --eps 0.1 --results gen_data 2>&1 | tee logs/gen_data.txt
```
To generate adversarial dataset for finetuning (testset):
```
python launch_resnet_attack.py --batch_num 100 --batch_size 100 --eps 0.03 --results gen_data_003 2>&1 | tee logs/gen_data_003.txt
```

To evaluate & plot attacker accuracy vs multiple eps values:  
```
python eval_plots.py
```

## Q4 Adversarial Training

Setup additional environment:
```
pip install scikit-learn
``` 


To finetune on ./results/gen_data (train set) & test on ./results/gen_data_003 (test set)
```
python finetune_defense.py --results train_01_test003 --resultsdir finetune_results --batch_size 100 --batch_num 20 2>&1 | tee logs/train.txt 
```

To evaluate & plot attacker accuracy vs epochs & losses vs epoch:  
```
python eval_plot_finetune.py
```

To load finetuned model weights:  
```
import torch

//Instantiate resnet50 model 

finetuned_model_state_dict= torch.load(f"finetune_results/train_01_test003",map_location=torch.device('cpu'))["model_weights"]
model.load_state_dict(finetuned_model_state_dict)

...
```

To load finetuned model weights that was trained by me:
```
Download train_01_test003 using the <google drive link> in cs5562-hw1-submission.pdf

Copy train_01_test003 to finetune_results/

Load model_state_dict as shown above 
```

