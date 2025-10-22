from data.LinearF16SS import B_f1

def apply_fault(env: object, fault_type: str):
    '''
    pass unwrapped env
    '''   
    if fault_type == "elevator_loss":
        env.B = B_f1
    
    elif fault_type == "null":
        pass
    else:
        print(f'apply_fault error: fault {fault_type} not recognised')
    
        