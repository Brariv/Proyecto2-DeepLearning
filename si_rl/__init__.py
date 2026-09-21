"""si_rl: agente de Reinforcement Learning (familia DQN / Rainbow) para ALE/SpaceInvaders-v5.

Módulos:
    env       -> creación del entorno y wrappers de preprocesamiento (crear_entorno)
    vec_env   -> entornos vectorizados (DummyVecEnv / SubprocVecEnv)
    replay    -> replay buffer eficiente en memoria con n-step y Prioritized Experience Replay
    networks  -> redes (Nature CNN, IMPALA CNN, dueling, NoisyNet, C51)
    agent     -> función de pérdida (DQN / Double DQN / C51) y selección de acciones
    presets   -> configuraciones por iteración (dqn, ddqn, dueling, rainbow, rainbow_fast ...)
    utils     -> utilidades (dispositivo, semillas, logging CSV, schedules)
"""

__version__ = "1.0.0"
