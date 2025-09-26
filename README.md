# AdaAggRL
source code for the paper 'Defending Against Sophisticated Poisoning Attacks with RL-based Aggregation in Federated Learning'
## Code Structure
```utilities.py``` contains all helper functions and defense algorithms including median, clipping median, krum and FLtrust.\
```exp_environments.py``` contains the environment used for training defense policy.\
```exp_environments_RLattack.py``` contains the environment used for training defense policy under RL-attack.\
```main.py``` contains code for training. \
```attack_utilities.py``` contains code for poisoning attacks.
## Setup Environment

Please run the following command to install required packages

```
# requirements
pip install -r requirements.txt
```
## Train
After downloading the data set and setting the parameters as required in ```main.py```, you can run the ```main.py``` file for FL training. Note: In actual run time, debug may be required to adapt to the current device and environment
## References
Our inverting gradients implementation is modified from https://github.com/JonasGeiping/invertinggradients

Our RL-attack implementation is modified from https://github.com/SliencerX/Learning-to-Attack-Federated-Learning

## Cite Format

    @misc{wang2024defendingsophisticatedpoisoningattacks,
          title={Defending Against Sophisticated Poisoning Attacks with RL-based Aggregation in Federated Learning}, 
          author={Yujing Wang and Hainan Zhang and Sijia Wen and Wangjie Qiu and Binghui Guo},
          year={2024},
          eprint={2406.14217},
          archivePrefix={arXiv},
          primaryClass={cs.LG},
          url={https://arxiv.org/abs/2406.14217}, 
        }
