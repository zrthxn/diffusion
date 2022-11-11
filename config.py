class Configuration:
    def __init__(self, config: dict) -> None:
        self.__dict__ = config
        self.keys = config
        
# Every key in config should be at the root depth
defaults = Configuration({
    "device": "cpu",
    "batch_size": 32,

    "noise": "linear", 
    "timesteps": 300,
    "start": 0.001,
    "end": 1.000,

    "lr": 1e-3,
    "epochs": 10,
})

def makeconfig(argv: list[str]):
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

    defaults.__dict__ = config_dict
    defaults.keys = config_dict

def print_help():
    print("""
    Usage:
        ./main.py [COMMANDS] [OPTIONS]
        
    Option Format
        --<option_name>=<option_value>
        
    COMMANDS:
    
        train   Train a new ensemble model
        test    Test a trained ensemble and get benchmarks. Can be used with `load`
        load    Load a trained ensemble from disk.
        plot    Plot predictions for a trained ensemble.
        help    Print this help message
    
    OPTIONS:
    
    
        
    Help String
    """)