
def print_model_info(model):
    print(model)
    no_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("{name} : {no}".format(name=name, no=param.numel()))
            no_params += param.numel()

    print("Overall number of params: ", no_params)
