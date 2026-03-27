from modelMgmt import *
from Main_Train import pre_init

def main_eval_auto(cpt_name):
    my_mgmt = pre_init(True)
    print('init evaluate...')
    my_mgmt.init_eval()
    print('load checkpoint...')
    my_mgmt.load_checkpoint(cpt_name, True)
    time.sleep(0.01)
    my_mgmt.predict_auto()

def qd_eval_manual(cpt_name, res_num):
    my_mgmt = pre_init(True)
    print('init evaluate...')
    my_mgmt.init_eval()
    my_mgmt.init_qd_client()
    print('load checkpoint...')
    my_mgmt.load_checkpoint(cpt_name, True)
    my_mgmt.update_qd_client()
    time.sleep(0.01)
    input_t = input("\nPress send input: ")
    my_mgmt.qdrant_query(input_t, res_num)

def chroma_eval_manual(cpt_name, res_num):
    my_mgmt = pre_init(True)
    print('init evaluate...')
    my_mgmt.init_eval()
    my_mgmt.init_ch_client()
    print('load checkpoint...')
    my_mgmt.load_checkpoint(cpt_name, True)
    my_mgmt.update_ch_client()
    time.sleep(0.01)
    input_t = input("\nPress send input: ")
    my_mgmt.chroma_query(input_t, res_num)


if __name__ == '__main__':
    #main_eval_auto('best_loss_cpt.pth')
    #qd_eval_manual('CheckPoint_Ep100_0.3814.pth', 3)
    chroma_eval_manual('CheckPoint_Ep100_0.3814.pth', 3)