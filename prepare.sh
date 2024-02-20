# prepare conda environment
conda create -n adp-fl python=3.11
conda activate adp-fl

# Prostate
scp root@140.238.14.153:/root/miccai/prostate/prostate.zip ./dataset/Prostate/prostate.zip
cd dataset/Prostate
unzip prostate.zip
echo "Prostate dataset prepared"

#RSNA-ICH
pip install gdown
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1bhe_0KvdxEli7-6ZrQ9ahaDPpSnvF4UW -O ./dataset/RSNA-ICH/brain.zip
unzip brain.zip
echo "RSNA-ICH dataset prepared"


