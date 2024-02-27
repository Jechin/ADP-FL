conda activate dpfl

# Run the experiment
python fed_main.py --adaclip --epsilon 10 --mode dpsgd --adp_round --adp_noise --seed 1 -N 20
# python fed_main.py --adaclip --epsilon 10 --mode dpsgd --adp_round --adp_noise --seed 2 -N 20
# python fed_main.py --adaclip --epsilon 10 --mode dpsgd --adp_round --adp_noise --seed 3 -N 20

python fed_main.py --adaclip --epsilon 10 --mode no_dp --adp_round --adp_noise --seed 1 -N 20
# python fed_main.py --adaclip --epsilon 10 --mode no_dp --adp_round --adp_noise --seed 2 -N 20
# python fed_main.py --adaclip --epsilon 10 --mode no_dp --adp_round --adp_noise --seed 3 -N 20

python fed_main.py --adaclip --epsilon 10 --mode dpsgd --adp_round --seed 1 -N 20
# python fed_main.py --adaclip --epsilon 10 --mode dpsgd --adp_round --adp_noise --seed 2 -N 20
# python fed_main.py --adaclip --epsilon 10 --mode dpsgd --adp_round --adp_noise --seed 3 -N 20

python fed_main.py --adaclip --epsilon 10 --mode dpsgd --adp_round --adp_noise --seed 1 -N 6 -data prostate
# python fed_main.py --adaclip --epsilon 10 --mode dpsgd --adp_round --adp_noise --seed 2 -N 6 -data prostate
# python fed_main.py --adaclip --epsilon 10 --mode dpsgd --adp_round --adp_noise --seed 3 -N 6 -data prostate

python fed_main.py --adaclip --epsilon 10 --mode no_dp --adp_round --adp_noise --seed 1 -N 6 -data prostate
# python fed_main.py --adaclip --epsilon 10 --mode no_dp --adp_round --adp_noise --seed 2 -N 6 -data prostate
# python fed_main.py --adaclip --epsilon 10 --mode no_dp --adp_round --adp_noise --seed 3 -N 6 -data prostate

python fed_main.py --adaclip --epsilon 10 --mode dpsgd --adp_round --seed 1 -N 6 --data prostate
# python fed_main.py --adaclip --epsilon 10 --mode dpsgd --adp_round --seed 2 -N 6 --data prostate
# python fed_main.py --adaclip --epsilon 10 --mode dpsgd --adp_round --seed 3 -N 6 --data prostate