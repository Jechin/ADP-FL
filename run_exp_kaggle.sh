seed_total=1
current_time=$(date +"%Y%m%d-%H%M%S")
mode=$1
# Run the experiment
for ((i=1; i<=$seed_total; i++))
do
    python fed_main.py --adaclip --epsilon 10 --mode $mode --adp_round --adp_noise --seed $i -N 20 --save_path /kaggle/working/$current_time --round_factor 0.99 --round_threshold 1e-4 --data_path /kaggle/input/rsna-ich/research/dept8/qdou/data/RSNA-ICH/organized/stage_2_train
    python fed_main.py --adaclip --epsilon 10 --mode $mode --adp_round --adp_noise --seed $i -N 20 --save_path /kaggle/working/$current_time --round_factor 0.99 --round_threshold 5e-4 --data_path /kaggle/input/rsna-ich/research/dept8/qdou/data/RSNA-ICH/organized/stage_2_train
    python fed_main.py --adaclip --epsilon 10 --mode $mode --adp_round --adp_noise --seed $i -N 20 --save_path /kaggle/working/$current_time --round_factor 0.99 --round_threshold 1e-3 --data_path /kaggle/input/rsna-ich/research/dept8/qdou/data/RSNA-ICH/organized/stage_2_train
    python fed_main.py --adaclip --epsilon 10 --mode $mode --adp_round --adp_noise --seed $i -N 6 --data prostate --save_path /kaggle/working/$current_time --round_factor 0.99 --round_threshold 1e-4 --eps 200 --data_path /kaggle/input/prostate/dataset2D
    python fed_main.py --adaclip --epsilon 10 --mode $mode --adp_round --adp_noise --seed $i -N 6 --data prostate --save_path /kaggle/working/$current_time --round_factor 0.99 --round_threshold 5e-4 --eps 200 --data_path /kaggle/input/prostate/dataset2D
    python fed_main.py --adaclip --epsilon 10 --mode $mode --adp_round --adp_noise --seed $i -N 6 --data prostate --save_path /kaggle/working/$current_time --round_factor 0.99 --round_threshold 1e-3 --eps 200 --data_path /kaggle/input/prostate/dataset2D
    # # no_dp
    # python fed_main.py --adaclip --epsilon 10 --mode $mode --no_dp --seed $i -N 20 --save_path /kaggle/working/$current_time --data_path /kaggle/input/rsna-ich/research/dept8/qdou/data/RSNA-ICH/organized/stage_2_train
    # # adp_noise and adp_round
    # python fed_main.py --adaclip --epsilon 10 --mode $mode --adp_round --adp_noise --seed $i -N 20 --save_path /kaggle/working/$current_time --round_factor 0.99 --data_path /kaggle/input/rsna-ich/research/dept8/qdou/data/RSNA-ICH/organized/stage_2_train
    # # normal noise
    # python fed_main.py --adaclip --epsilon 10 --mode $mode --seed $i -N 20 --save_path /kaggle/working/$current_time --data_path /kaggle/input/rsna-ich/research/dept8/qdou/data/RSNA-ICH/organized/stage_2_train
    # # adp_noise
    # python fed_main.py --adaclip --epsilon 10 --mode $mode --adp_noise --seed $i -N 20 --save_path /kaggle/working/$current_time --data_path /kaggle/input/rsna-ich/research/dept8/qdou/data/RSNA-ICH/organized/stage_2_train
    
    # # no_dp
    # python fed_main.py --adaclip --epsilon 10 --mode $mode --no_dp --seed $i -N 6 --data prostate --save_path /kaggle/working/$current_time --eps 200 --round_threshold 5e-4 --data_path /kaggle/input/prostate/dataset2D
    # # adp_noise and adp_round
    # python fed_main.py --adaclip --epsilon 10 --mode $mode --adp_round --adp_noise --seed $i -N 6 --data prostate --save_path /kaggle/working/$current_time --round_factor 0.99 --round_threshold 5e-4 --eps 200 --data_path /kaggle/input/prostate/dataset2D
    # # normal noise
    # python fed_main.py --adaclip --epsilon 10 --mode $mode --seed $i -N 6 --data prostate --save_path /kaggle/working/$current_time --eps 200 --round_threshold 5e-4 --data_path /kaggle/input/prostate/dataset2D
    # # adp_noise
    # python fed_main.py --adaclip --epsilon 10 --mode $mode --adp_noise --seed $i -N 6 --data prostate --save_path /kaggle/working/$current_time --eps 200 --round_threshold 5e-4 --data_path /kaggle/input/prostate/dataset2D
done
