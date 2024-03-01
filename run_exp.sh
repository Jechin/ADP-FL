seed_total=1
current_time=$(date +"%Y%m%d-%H%M%S")
# Run the experiment
for ((i=1; i<=$seed_total; i++))
do
    # no_dp
    python fed_main.py --adaclip --epsilon 10 --mode no_dp --seed $i -N 20 --save_path $current_time
    # adp_noise and adp_round
    python fed_main.py --adaclip --epsilon 10 --mode dpsgd --adp_round --adp_noise --seed $i -N 20 --save_path $current_time --round_factor 0.99 --round_threshold 0.0001
    # normal noise
    python fed_main.py --adaclip --epsilon 10 --mode dpsgd --seed $i -N 20 --save_path $current_time 
    # adp_noise
    python fed_main.py --adaclip --epsilon 10 --mode dpsgd --adp_noise --seed $i -N 20 --save_path $current_time 
    
    # no_dp
    python fed_main.py --adaclip --epsilon 10 --mode no_dp --seed $i -N 6 --data prostate --save_path $current_time --eps 200
    # adp_noise and adp_round
    python fed_main.py --adaclip --epsilon 10 --mode dpsgd --adp_round --adp_noise --seed $i -N 6 --data prostate --save_path $current_time --round_factor 0.99 --round_threshold 0.0005 --eps 200
    # normal noise
    python fed_main.py --adaclip --epsilon 10 --mode dpsgd --seed $i -N 6 --data prostate --save_path $current_time --eps 200
    # adp_noise
    python fed_main.py --adaclip --epsilon 10 --mode dpsgd --adp_noise --seed $i -N 6 --data prostate --save_path $current_time --eps 200
done
