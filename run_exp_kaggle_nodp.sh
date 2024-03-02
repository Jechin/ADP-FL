seed_total=1
current_time=$(date +"%Y%m%d-%H%M%S")
mode=$1
# Run the experiment
for ((i=1; i<=$seed_total; i++))
do
    python fed_main.py -mode fedsgd -N 20 ---seed 1 --no_dp --save_path /kaggle/working/results/$current_time --data_path /kaggle/input/rsna-ich/research/dept8/qdou/data/RSNA-ICH/organized/stage_2_train
    python fed_main.py -mode fedadam -N 20 ---seed 1 --no_dp --save_path /kaggle/working/results/$current_time --data_path /kaggle/input/rsna-ich/research/dept8/qdou/data/RSNA-ICH/organized/stage_2_train
    python fed_main.py -mode fedsgd -N 6 ---seed 1 --data prostate --no_dp --save_path /kaggle/working/results/$current_time --data_path /kaggle/input/prostate/dataset2D
    python fed_main.py -mode fedadam -N 6 ---seed 1 --data prostate --no_dp --save_path /kaggle/working/results/$current_time --data_path /kaggle/input/prostate/dataset2D
done
