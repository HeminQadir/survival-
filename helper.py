import os 


def make_dir(dir_):
    # Check if the directory exists
    if not os.path.exists(dir_):
        # Create the directory
        os.makedirs(dir_)
    print(f"Directory '{dir_}' checked/created successfully.")


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("="*10)
    print("Number of parameters:", params/1000000)
    print("*"*10)
    