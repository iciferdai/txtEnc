from modelMgmt import *

def pre_init(need_data=True, sft_data=False):
    print('init model...')
    my_model = TxtEnc()
    my_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if need_data:
        print('Preparing data...')
        if not sft_data:
            train_dataloader = process_data()
        else:
            train_dataloader = process_sft_data()
    else:
        train_dataloader = None
    print('init ModelManagement...')
    my_mgmt = ModelManagement(my_model, train_dataloader, my_dev)
    return my_mgmt

def main_train(steps):
    my_mgmt = pre_init()
    print('init train...')
    my_mgmt.init_train()
    my_mgmt.init_weights()
    print('Start train...')
    my_mgmt.train_epochs(steps)
    my_mgmt.save_state()
    my_mgmt.show_dashboard()

def sft_train(cpt_name, steps):
    my_mgmt = pre_init(True, True)
    print('init train...')
    my_mgmt.init_sft_train(cpt_name)
    print('Start train...')
    my_mgmt.train_sft_epochs(steps)
    my_mgmt.save_state()
    my_mgmt.show_dashboard()

def check_status():
    my_mgmt = pre_init(False)
    print('load status of best_test...')
    #my_mgmt.init_train()
    my_mgmt.init_dashboard()
    my_mgmt.load_state('State_Ep10000_0.0002.pkl')
    #my_mgmt.load_checkpoint('CheckPoint_Ep50000_1.1481.pth')
    my_mgmt.progress_info(True)
    my_mgmt.show_dashboard()

if __name__ == '__main__':
    #main_train(1000)
    sft_train('CheckPoint_Ep1000_0.0978.pth', 100)
    #check_status()