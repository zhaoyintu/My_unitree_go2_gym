from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.Go2_MoB.GO2_JUMP.go2_jump_env import GO2_JUMP_Robot
from legged_gym.envs.Go2_MoB.GO2_JUMP.GO2_JUMP_config import GO2_JUMP_Cfg_Yu,GO2_JUMP_PPO_Yu


from legged_gym.envs.Go2_MoB.GO2_Trot.GO2_Trot import GO2_Trot_Robot
from legged_gym.envs.Go2_MoB.GO2_Trot.GO2_Trot_config import GO2_Trot_Cfg_Yu,GO2_Trot_PPO_Yu

from legged_gym.envs.Go2_MoB.GO2_Trot.GO2_Stairs import GO2_Stairs_Robot
from legged_gym.envs.Go2_MoB.GO2_Trot.GO2_Stairs_config import GO2_Stairs_Cfg_Yu,GO2_Stairs_PPO_Yu

from legged_gym.envs.GO2_Flip.GO2_BackFlip.GO2_BackFlip_env import Go2_BackFlip
from legged_gym.envs.GO2_Flip.GO2_BackFlip.GO2_BackFlip_Config import GO2_BackFlip_Cfg_Yu, GO2_BackFlip_PPO_Yu

from legged_gym.envs.GO2_Flip.GO2_Spring_Jump.GO2_Spring_Jump_env import GO2_Spring_Jump_Robot
from legged_gym.envs.GO2_Flip.GO2_Spring_Jump.GO2_Spring_Jump_Config import GO2_Spring_Jump_Cfg_Yu, GO2_Spring_Jump_PPO_Yu

from legged_gym.envs.GO2_Stand.GO2_Handstand.Go2_handstand import Go2_stand
from legged_gym.envs.GO2_Stand.GO2_Handstand.Go2_handstand_Config import GO2Cfg_Handstand,GO2CfgPPO_Handstand

from legged_gym.envs.GO2_Stand.GO2_Leggedstand.Go2_legstand import Go2_legstand
from legged_gym.envs.GO2_Stand.GO2_Leggedstand.Go2_legstand_Config import GO2Cfg_Leggedstand,GO2CfgPPO_Leggedstand

from legged_gym.envs.Lite3_Stand.Lite3_Handstand.Lite3_handstand import Lite3_legstand
from legged_gym.envs.Lite3_Stand.Lite3_Handstand.Lite3_handstand_Config import Lite3Cfg_Leggedstand, Lite3CfgPPO_Leggedstand

from legged_gym.envs.Lite3_Stand.Lite3_Handstand_Strict.Lite3_handstand_strict import Lite3_legstand_strict
from legged_gym.envs.Lite3_Stand.Lite3_Handstand_Strict.Lite3_handstand_strict_Config import Lite3Cfg_LeggedstandStrict, Lite3CfgPPO_LeggedstandStrict

from legged_gym.utils.task_registry import task_registry


task_registry.register( "go2_trot", GO2_Trot_Robot, GO2_Trot_Cfg_Yu(), GO2_Trot_PPO_Yu())
task_registry.register( "go2_stairs", GO2_Stairs_Robot, GO2_Stairs_Cfg_Yu(), GO2_Stairs_PPO_Yu())
task_registry.register( "go2_jump", GO2_JUMP_Robot, GO2_JUMP_Cfg_Yu(), GO2_JUMP_PPO_Yu())
task_registry.register( "go2_handstand", Go2_stand, GO2Cfg_Handstand(), GO2CfgPPO_Handstand())
task_registry.register( "go2_leggedstand", Go2_legstand, GO2Cfg_Leggedstand(), GO2CfgPPO_Leggedstand())
task_registry.register( "go2_spring_jump", GO2_Spring_Jump_Robot, GO2_Spring_Jump_Cfg_Yu(), GO2_Spring_Jump_PPO_Yu())
task_registry.register( "go2_backflip", Go2_BackFlip, GO2_BackFlip_Cfg_Yu(), GO2_BackFlip_PPO_Yu())
# Lite3 front-paw handstand: name is correctly "handstand" (front-paw stand,
# head down) — same task family as `go2_leggedstand` despite the inverted
# Go2 naming convention.
task_registry.register( "lite3_handstand", Lite3_legstand, Lite3Cfg_Leggedstand(), Lite3CfgPPO_Leggedstand())
# Strict variant: same task with stronger anti-kneeling guards (per-body
# collision split + 12-joint default_pos_reward) for A/B comparison.
task_registry.register( "lite3_handstand_strict", Lite3_legstand_strict, Lite3Cfg_LeggedstandStrict(), Lite3CfgPPO_LeggedstandStrict())

print("注册的任务:  ",task_registry.task_classes)
