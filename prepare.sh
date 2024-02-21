# prepare conda environment
###
 # @Description: 
 # @Author: Jechin jechinyu@163.com
 # @Date: 2024-02-20 22:12:39
 # @LastEditors: Jechin jechinyu@163.com
 # @LastEditTime: 2024-02-21 00:43:12
### 
conda create -n adp-fl python=3.11
conda activate adp-fl
pip install numpy==1.23.2 

# Prostate
scp root@140.238.14.153:/root/miccai/prostate/prostate.zip ./dataset/Prostate/prostate.zip
cd ./dataset/Prostate
unzip prostate.zip
cd ...
echo "Prostate dataset prepared"

#RSNA-ICH
pip install gdown
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1bhe_0KvdxEli7-6ZrQ9ahaDPpSnvF4UW -O ./dataset/RSNA-ICH/brain.zip
cd ./dataset/RSNA-ICH
unzip brain.zip
echo "RSNA-ICH dataset prepared"


