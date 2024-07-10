# config_manager.py

# Initialize the global variable as None
_global_config = None

def set_global_config(config):
    global _global_config
    _global_config = config

def get_global_config():
    return _global_config

def get_pbs_debug():
    global _global_config
    return _global_config["Global"]["pbs_debug"] == "true" if _global_config else False