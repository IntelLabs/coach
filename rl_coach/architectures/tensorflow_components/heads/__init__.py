from .head import Head#, HeadLoss
from .q_head import QHead
#from .ppo_head import PPOHead
from .v_head import VHead


__all__ = [
    'Head',
    #'PPOHead',
    'QHead',
    'VHead'
]

