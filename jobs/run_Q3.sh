python launch_resnet_attack.py --batch_num 20 --batch_size 100 --eps 0.03 --results eps_003 2>&1 | tee logs/eps_003.txt &&
python launch_resnet_attack.py --batch_num 20 --batch_size 100 --eps 0.1 --results eps_01 2>&1 | tee logs/eps_01.txt &&
python launch_resnet_attack.py --batch_num 20 --batch_size 100 --eps 0.2 --results eps_02 2>&1 | tee logs/eps_02.txt &&
python launch_resnet_attack.py --batch_num 20 --batch_size 100 --eps 0.3 --results eps_03 2>&1 | tee logs/eps_03.txt &&
python launch_resnet_attack.py --batch_num 20 --batch_size 100 --eps 0.4 --results eps_04 2>&1 | tee logs/eps_04.txt &&
python launch_resnet_attack.py --batch_num 20 --batch_size 100 --eps 0.5 --results eps_05 2>&1 | tee logs/eps_05.txt &&
python launch_resnet_attack.py --batch_num 20 --batch_size 100 --eps 0.6 --results eps_06 2>&1 | tee logs/eps_06.txt &&
python launch_resnet_attack.py --batch_num 20 --batch_size 100 --eps 0.7 --results eps_07 2>&1 | tee logs/eps_07.txt &&
python launch_resnet_attack.py --batch_num 20 --batch_size 100 --eps 0.8 --results eps_08 2>&1 | tee logs/eps_08.txt &&
python launch_resnet_attack.py --batch_num 20 --batch_size 100 --eps 0.9 --results eps_09 2>&1 | tee logs/eps_09.txt
