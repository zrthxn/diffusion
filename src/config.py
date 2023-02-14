from yaml import safe_load

class Configuration:
    def __init__(self, config: dict) -> None:
        self.__dict__ = config
        self.keys = config

# Every key in config should be at the root depth
with open("defaults.yml", 'r') as f:
    defaults = Configuration(safe_load(f))

def makeconfig(argv: list):
    config_dict = defaults.keys
    for arg in argv:
        if not arg: continue

        value_index = arg.find("=")
        name, value = arg, True
        if value_index != -1:
            name = arg[0:value_index]
            value = arg[value_index+1:]
        
        assert name in config_dict.keys(), "Cannot add configurations"
        config_dict[name] = type(config_dict[name])(value)

    if config_dict["dryrun"]:
        config_dict = { **config_dict,
            "batch_size": 16,
            "epochs": 1,
        }

    defaults.__dict__ = config_dict
    defaults.keys = config_dict

def print_help():
    nspaces = lambda n: ''.join([' ' for _ in range(n)])
    M = max(map(len, list(defaults.keys))) + 4
    options = '\n'.join([f"\t{k}{nspaces(M-len(k))}{v}" for k, v in defaults.keys.items()][:-1])
    print(f"""
    Usage:
        ./main.py [COMMANDS] [OPTIONS]
        
    Option Format
        --<option_name>=<option_value>
        
    COMMANDS:
    
        train   Train a new diffusion model
        test    Generate images from a trained model.
        help    Print this help message
    
    OPTIONS:
    
    {options}
        
    Help String
    """)
    exit(0)