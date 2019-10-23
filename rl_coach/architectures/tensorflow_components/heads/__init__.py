from .head import Head#, HeadLoss
from .q_head import QHead
from .policy_head import PolicyHead
#from .ppo_head import PPOHead
from .ppo_v_head import PPOVHead
from .v_head import VHead


__all__ = [
    'Head',
    'PolicyHead',
    #'PPOHead',
    'PPOVHead',
    'QHead',
    'VHead'
]

