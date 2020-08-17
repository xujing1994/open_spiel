from absl import app
from absl import flags
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt

def read_wr(txt_name):
    text_file = open(txt_name, "r")
    lines = text_file.read().split("\n")
    list1 = []
    list2 = []
    list3 = []
    for line in lines[:-1]:
        [str1, str2, str3] = line.split(" ")
        list1.append(float(str1))
        list2.append(float(str2))
        list3.append(float(str3))
    return list1, list2, list3

def read_exploitability(txt_name):
    txt_file = open(txt_name, 'r')
    lines = txt_file.read().split('\n')
    num_list = []
    for str in lines[:-1]:
        if str == "NaN":
            num_list.append(1)
        else:
            num_list.append(float(str))
    return num_list

def merge_expl(txt_name_1, txt_name_2):
    num_list_1 = read_exploitability(txt_name_1)
    num_list_2 = read_exploitability(txt_name_2)
    return num_list_1 + num_list_2

def read_loss(txt_name):
    txt_file = open(txt_name)
    lines = txt_file.read().split('\n')
    list1 = []
    list2 = []
    for line in lines[:-1]:
        [str1, str2] = line.split(' ')
        if str1 != 'None':
            list1.append(float(str1))
        else:
            list1.append(str1)
        if str2 != 'None':
            list2.append(float(str2))
        else:
            list2.append(str2)
    for idx, number in enumerate(list1):
        if number == 'None':
            list1[idx] = list1[idx+1]
    for number, idx in enumerate(list2):
        if number == 'None':
            list2[idx] = list2[idx+1]
    return list1, list2
def read_behavior_probs(txt_name):
    text_file = open(txt_name, "r")
    lines = text_file.read().split("\n")
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []
    list8 = []
    for line in lines[:-1]:
        [str1, str2, str3, str4, str5, str6, str7, str8] = line.split(" ")
        list1.append(float(str1))
        list2.append(float(str2))
        list3.append(float(str3))
        list4.append(float(str4))
        list5.append(float(str5))
        list6.append(float(str6))
        list7.append(float(str7))
        list8.append(float(str8))
    return list1, list2, list3, list4, list5, list6, list7, list8


def main(argv):
    kuhn_poker_nfsp_0 = "/home/jxu8/Code_update/open_spiel/evaluation_data/eval_kp_nfsp_0.1_7_27/"
    kuhn_poker_nfsp_1 = "/home/jxu8/Code_update/open_spiel/evaluation_data/eval_kp_nfsp_1_7_28/"
    ttt_nfsp_0 = "/home/jxu8/Code_update/open_spiel/evaluation_data/eval_ttt_nfsp_0.1_8_6/"
    ttt_nfsp_0_2 = "/home/jxu8/Code_update/open_spiel/evaluation_data/eval_ttt_nfsp_0.2_8_6/"
    ttt_nfsp_1 = "/home/jxu8/Code_update/open_spiel/evaluation_data/eval_ttt_nfsp_1_7_29/"

    kuhn_poker_psro = "/home/jxu8/Code/open_spiel/evaluation_data/eval_kuhn_poker_psro_7_2/"

    expl_0 = []
    expl_1 = []
    expl_0_2 = []
    expl_0_5 = []
    expl_0_0_1 = []
    loss_agent0 = []
    loss_agent1 = []

    win_rates_against_random1_eta0 = [] #load win rates against random agent1, trained with eta 0.1
    win_rates_against_random0_eta0 = [] #load win rates against random agetn0, trained with eta 0.1
    win_rates_against_random1_eta1 = []
    win_rates_against_random0_eta1 = []
    win_rates_against_eachother_eta0 = [] #load win rates against each other, trained with eta 0.1
    win_rates_against_eachother_eta1 = []

    expl_0.append(read_exploitability('/home/jxu8/Code_update/open_spiel/sessions_nfsp/kuhn_poker_0.1_7_27/' + 'exploitabilities.txt'))
    expl_0.append(merge_expl(ttt_nfsp_0 + 'exploitabilities.txt', "/home/jxu8/Code_update/open_spiel/evaluation_data/eval_ttt_nfsp_0.1_8_12/exploitabilities_from_1506e4.txt"))
    expl_1.append(read_exploitability('/home/jxu8/Code_update/open_spiel/sessions_nfsp/kuhn_poker_1_7_28/' + 'exploitability.txt'))
    expl_1.append(read_exploitability(ttt_nfsp_1 + 'exploitabilities.txt'))
    expl_0_2.append(read_exploitability('/home/jxu8/Code_update/open_spiel/sessions_nfsp/kuhn_poker_0.2_8_6/exploitability.txt'))
    expl_0_2.append(merge_expl(ttt_nfsp_0_2 + 'exploitabilities.txt', "/home/jxu8/Code_update/open_spiel/evaluation_data/eval_ttt_nfsp_0.2_8_12/exploitabilities_from_1506e4.txt"))
    expl_0_5.append(read_exploitability('/home/jxu8/Code_update/open_spiel/sessions_nfsp/kuhn_poker_0.5_8_6/exploitability.txt'))
    expl_0_0_1.append(read_exploitability('/home/jxu8/Code_update/open_spiel/sessions_nfsp/kuhn_poker_0.01_8_6/exploitability.txt'))

    loss_agent0.append(read_loss('/home/jxu8/Code_update/open_spiel/sessions_nfsp/kuhn_poker_0.1_7_27/loss_agent0.txt'))
    loss_agent1.append(read_loss('/home/jxu8/Code_update/open_spiel/sessions_nfsp/kuhn_poker_0.1_7_27/loss_agent1.txt'))
    loss_agent0.append(read_loss('/home/jxu8/Code_update/open_spiel/sessions_nfsp/kuhn_poker_1_7_28/loss_agent0.txt'))
    loss_agent1.append(read_loss('/home/jxu8/Code_update/open_spiel/sessions_nfsp/kuhn_poker_1_7_28/loss_agent1.txt'))

    loss_agent0.append(read_loss('/home/jxu8/Code_update/open_spiel/sessions_nfsp/tic_tac_toe_0.1_7_26/loss_agent0.txt'))
    loss_agent1.append(read_loss('/home/jxu8/Code_update/open_spiel/sessions_nfsp/tic_tac_toe_0.1_7_26/loss_agent1.txt'))
    loss_agent0.append(read_loss('/home/jxu8/Code_update/open_spiel/sessions_nfsp/tic_tac_toe_1_7_29/loss_agent0.txt'))
    loss_agent1.append(read_loss('/home/jxu8/Code_update/open_spiel/sessions_nfsp/tic_tac_toe_1_7_29/loss_agent1.txt'))
    # load win_rates with eta0.1 in training
    win_rates_against_random1_eta0.append(read_wr(ttt_nfsp_0 + 'win_rates/eta_0/win_rates_against_random_agent1.txt')) # load win rates of trained agent0 against random agent1, with eta 0 in evaluation process (average policy only)
    win_rates_against_random0_eta0.append(read_wr(ttt_nfsp_0 + 'win_rates/eta_0/win_rates_against_random_agent0.txt')) # load win rates of trained agent1 against random agent0, with eta 0 in evaluation process (average policy only)

    win_rates_against_random1_eta0.append(read_wr(kuhn_poker_nfsp_0 + 'win_rates/eta_0/win_rates_against_random_agent1.txt')) # load win rates of trained agent0 against random agent1, with eta 0 in evaluation process (average policy only)
    win_rates_against_random0_eta0.append(read_wr(kuhn_poker_nfsp_0 + 'win_rates/eta_0/win_rates_against_random_agent0.txt')) # load win rates of trained agent1 against random agent0, with eta 0 in evaluation process (average policy only)

    win_rates_against_random1_eta0.append(read_wr(ttt_nfsp_0_2 + 'win_rates/eta_0/win_rates_against_random_agent1.txt'))
    win_rates_against_random0_eta0.append(read_wr(ttt_nfsp_0_2 + 'win_rates/eta_0/win_rates_against_random_agent0.txt'))
    win_rates_against_eachother_eta0.append(read_wr(ttt_nfsp_0 + 'win_rates/eta_0/win_rates_against_eachother.txt')) # load win rates of trained agent0 against trained agent1, with eta 0 for both in the evaluation process (both use average policy only)

    win_rates_against_eachother_eta0.append(read_wr(kuhn_poker_nfsp_0 + 'win_rates/eta_0/win_rates_against_eachother.txt')) # load win rates of trained agent0 against trained agent1, with eta 0 for both in the evaluation process (both use average policy only)
    win_rates_against_eachother_eta0.append(read_wr(ttt_nfsp_0_2 + 'win_rates/eta_0/win_rates_against_eachother.txt'))

    #load win rates with eta1 in training
    win_rates_against_random1_eta1.append(read_wr(ttt_nfsp_1 + 'win_rates/eta_0/win_rates_against_random_agent1.txt')) # load win rates of trained agent0 against random agent1, with eta 0 in evaluation process (average policy only)
    win_rates_against_random0_eta1.append(read_wr(ttt_nfsp_1 + 'win_rates/eta_0/win_rates_against_random_agent0.txt')) # load win rates of trained agent1 against random agent0, with eta 0 in evaluation process (average policy only)

    win_rates_against_random1_eta1.append(read_wr(kuhn_poker_nfsp_1 + 'win_rates/eta_0/win_rates_against_random_agent1.txt')) # load win rates of trained agent0 against random agent1, with eta 0 in evaluation process (average policy only)
    win_rates_against_random0_eta1.append(read_wr(kuhn_poker_nfsp_1 + 'win_rates/eta_0/win_rates_against_random_agent0.txt')) # load win rates of trained agent1 against random agent0, with eta 0 in evaluation process (average policy only)

    win_rates_against_eachother_eta1.append(read_wr(ttt_nfsp_1 + 'win_rates/eta_0/win_rates_against_eachother.txt')) # load win rates of trained agent0 against trained agent1, with eta 0 for both in the evaluation process (both use average policy only)

    win_rates_against_eachother_eta1.append(read_wr(kuhn_poker_nfsp_1 + 'win_rates/eta_0/win_rates_against_eachother.txt')) # load win rates of trained agent0 against trained agent1, with eta 0 for both in the evaluation process (both use average policy only)

    # episode = range(10, 3010, 10)
    # plot exploitability, avg utility in kuhn_poker_nfsp_0.1_7_27
    plt.figure(figsize=(10, 3))
    #plt.subplot(311)
    plt.xlim(-10, 300)
    y_ticks = np.arange(0, 0.50, 0.05)
    line1, = plt.plot(expl_0[0], "b", label="kuhn_poker_nfsp_0.1")
    line2, = plt.plot(expl_0_0_1[0], 'm', label='kuhn_poker_nfsp_0.01')
    line3, = plt.plot(expl_0_2[0], 'y', label="kuhn_poker_nfsp_0.2")
    line4, = plt.plot(expl_0_5[0], 'g', label='kuhn_poker_nfsp_0.5')
    line5, = plt.plot(expl_1[0], "r", label="kuhn_poker_nfsp_1")

    plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2, line3, line4, line5], loc='upper right')
    plt.ylabel('exploitability')
    plt.xlabel('episode(*1e4)')
    plt.show()

    #plt win rates in kuhn_poker_nfsp_0.1_7_27
    plt.figure(figsize=(15, 12))
    ax2 = plt.subplot(311)
    ax2.set_title("trained agent0 against random agent1(eta 0 in evaluation)")
    plt.ylim(0, 1)
    line1, = plt.plot(win_rates_against_random1_eta0[1][0], 'b', label='kp_nfsp_0.1_trained_agent0')
    line2, = plt.plot(win_rates_against_random1_eta0[1][1], 'r', label='kp_nfsp_0.1_random_agent1')
    line3, = plt.plot(win_rates_against_random1_eta0[1][2], 'g', label='kp_nfsp_0.1_draw')
    plt.legend(handles=[line1, line2, line3], loc='lower right', prop={'size': 9})
    plt.ylabel('win_rate')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(312)
    ax2.set_title('random agent0 against trained agent1(eta 0 in evaluation)')
    plt.ylim(0, 1)
    line1, = plt.plot(win_rates_against_random0_eta0[1][0], 'r', label='kp_nfsp_0.1_random_agent0')
    line2, = plt.plot(win_rates_against_random0_eta0[1][1], 'b', label='kp_nfsp_0.1_traomed_agent1')
    line3, = plt.plot(win_rates_against_random0_eta0[1][2], 'g', label='kp_nfsp_0.1_draw')
    plt.legend(handles=[line1, line2, line3], loc='lower right', prop={'size': 9})
    plt.ylabel('win_rate')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(313)
    ax2.set_title("win rates against each other(eta 0 for both in evaluation)")
    plt.ylim(0, 1)
    line1, = plt.plot(win_rates_against_eachother_eta0[1][0], 'b', label='kp_nfsp_0.1_agent0')
    line2, = plt.plot(win_rates_against_eachother_eta0[1][1], 'r', label='kp_nfsp_0.1_agent1')
    line3, = plt.plot(win_rates_against_eachother_eta0[1][2], 'g', label='kp_nfsp_0.1_draw')
    plt.legend(handles=[line1, line2, line3], loc='lower right', prop={'size': 9})
    plt.ylabel('win_rate')
    plt.xlabel('episode(*1e4)')
    plt.show()

    # plot random agent0 against trained agent1 in ttt_nfsp_0.1_7_24
    # plot win rates of trained agents against each other in ttt_nfsp_0.1_7_24

    # plt win rates in kuhn_poker_nfsp_1_7_28
    plt.figure(figsize=(15, 12))
    ax2 = plt.subplot(311)
    ax2.set_title("trained agent0 against random agent1(eta 0 in evaluation)")
    plt.ylim(0, 1)
    line1, = plt.plot(win_rates_against_random1_eta1[1][0], 'b', label='kp_nfsp_1_trained_agent0')
    line2, = plt.plot(win_rates_against_random1_eta1[1][1], 'r', label='kp_nfsp_1_random_agent1')
    line3, = plt.plot(win_rates_against_random1_eta1[1][2], 'g', label='kp_nfsp_1_draw')
    plt.legend(handles=[line1, line2, line3], loc='lower right', prop={'size': 9})
    plt.ylabel('win_rate')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(312)
    ax2.set_title('random agent0 against trained agent1(eta 0 in evaluation)')
    plt.ylim(0, 1)
    line1, = plt.plot(win_rates_against_random0_eta1[1][0], 'r', label='kp_nfsp_1_random_agent0')
    line2, = plt.plot(win_rates_against_random0_eta1[1][1], 'b', label='kp_nfsp_1_traomed_agent1')
    line3, = plt.plot(win_rates_against_random0_eta1[1][2], 'g', label='kp_nfsp_1_draw')
    plt.legend(handles=[line1, line2, line3], loc='lower right', prop={'size': 9})
    plt.ylabel('win_rate')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(313)
    ax2.set_title("win rates against each other(eta 0 for both in evaluation)")
    plt.ylim(0, 1)
    line1, = plt.plot(win_rates_against_eachother_eta1[1][0], 'b', label='kp_nfsp_1_agent0')
    line2, = plt.plot(win_rates_against_eachother_eta1[1][1], 'r', label='kp_nfsp_1_agent1')
    line3, = plt.plot(win_rates_against_eachother_eta1[1][2], 'g', label='kp_nfsp_1_draw')
    plt.legend(handles=[line1, line2, line3], loc='lower right', prop={'size': 9})
    plt.ylabel('win_rate')
    plt.xlabel('episode(*1e4)')
    plt.show()
    
    # plt.subplot(312)
    # line1, = plt.plot(loss_agent0[0][0], "b", label="supervised learning loss")
    # line2, = plt.plot(loss_agent0[0][1], "r", label="reinforcement learning loss")
    # plt.legend(handles=[line1, line2], loc='upper right')
    # plt.ylabel('loss_agent0')
    # plt.xlabel('episode(*1e4)')
    # 
    # plt.subplot(313)
    # line1, = plt.plot(loss_agent1[0][0], "b", label="supervised learning loss")
    # line2, = plt.plot(loss_agent1[0][1], "r", label="reinforcement learning loss")
    # plt.legend(handles=[line1, line2], loc='upper right')
    # plt.ylabel('loss_agent1')
    # plt.xlabel('episode(*1e4)')
    # plt.show()

    # plot exploitability in tic_tac_toe_nfsp_0.1_7_26, tic_tac_toe_nfsp_1_7_29
    plt.figure(figsize=(10,3))
    #plt.subplot(311)
    plt.ylim(0, 1.05)
    y_ticks = np.arange(0, 1.1, 0.1)
    x_range_0 = [6*(x+1) for x in range(len(expl_0[1]))]
    line1, = plt.plot(x_range_0, expl_0[1], "b", label="ttt_nfsp_0.1")
    x_range_1 = [6*(x+1) for x in range(len(expl_0_2[1]))]
    line2, = plt.plot(x_range_1, expl_0_2[1], "g", label="ttt_nfsp_0.2")
    x_range_2 = [4*(x+1) for x in range(len(expl_1[1]))]
    line3, = plt.plot(x_range_2, expl_1[1], "r", label="ttt_nfsp_1")
    plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2, line3], loc='upper right')
    plt.ylabel('exploitability')
    plt.xlabel('episode(*1e4)')
    plt.show()

    # plt.subplot(312)
    # line1, = plt.plot(loss_agent0[0][0], "b", label="supervised learning loss")
    # line2, = plt.plot(loss_agent0[0][1], "r", label="reinforcement learning loss")
    # plt.legend(handles=[line1, line2], loc='lower right')
    # plt.ylabel('loss_agent0')
    # plt.xlabel('episode(*1e4)')
    # 
    # plt.subplot(313)
    # line1, = plt.plot(loss_agent1[0][0], "b", label="supervised learning loss")
    # line2, = plt.plot(loss_agent1[0][1], "r", label="reinforcement learning loss")
    # plt.legend(handles=[line1, line2], loc='lower right')
    # plt.ylabel('loss_agent1')
    # plt.xlabel('episode(*1e4)')
    # plt.show()
    
    # plot trained agent0 against random agent1 in tic_tac_toe_nfsp_0.1_7_26
    plt.figure(figsize=(15, 12))
    ax2 = plt.subplot(311)
    ax2.set_title("trained agent0 against random agent1(eta 0 in evaluation)")
    plt.ylim(0, 1)
    line1, = plt.plot(win_rates_against_random1_eta0[0][0], 'b', label='ttt_nfsp_0.1_trained_agent0')
    line2, = plt.plot(win_rates_against_random1_eta0[0][1], 'r', label='ttt_nfsp_0.1_random_agent1')
    line3, = plt.plot(win_rates_against_random1_eta0[0][2], 'g', label='ttt_nfsp_0.1_draw')
    plt.legend(handles=[line1, line2, line3], loc='lower right', prop={'size': 9})
    plt.ylabel('win_rate')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(312)
    ax2.set_title('random agent0 against trained agent1(eta 0 in evaluation)')
    plt.ylim(0, 1)
    line1, = plt.plot(win_rates_against_random0_eta0[0][0], 'r', label='ttt_nfsp_0.1_random_agent0')
    line2, = plt.plot(win_rates_against_random0_eta0[0][1], 'b', label='ttt_nfsp_0.1_traomed_agent1')
    line3, = plt.plot(win_rates_against_random0_eta0[0][2], 'g', label='ttt_nfsp_0.1_draw')
    plt.legend(handles=[line1, line2, line3], loc='lower right', prop={'size': 9})
    plt.ylabel('win_rate')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(313)
    ax2.set_title("win rates against each other(eta 0 for both in evaluation)")
    plt.ylim(0, 1)
    line1, = plt.plot(win_rates_against_eachother_eta0[0][0], 'b', label='ttt_nfsp_0.1_agent0')
    line2, = plt.plot(win_rates_against_eachother_eta0[0][1], 'r', label='ttt_nfsp_0.1_agent1')
    line3, = plt.plot(win_rates_against_eachother_eta0[0][2], 'g', label='ttt_nfsp_0.1_draw')
    plt.legend(handles=[line1, line2, line3], loc='lower right', prop={'size': 9})
    plt.ylabel('win_rate')
    plt.xlabel('episode(*1e4)')
    plt.show()

    plt.figure(figsize=(15, 12))
    ax2 = plt.subplot(311)
    ax2.set_title("trained agent0 against random agent1(eta 0 in evaluation)")
    plt.ylim(0, 1)
    line1, = plt.plot(win_rates_against_random1_eta0[2][0], 'b', label='ttt_nfsp_0.2_trained_agent0')
    line2, = plt.plot(win_rates_against_random1_eta0[2][1], 'r', label='ttt_nfsp_0.2_random_agent1')
    line3, = plt.plot(win_rates_against_random1_eta0[2][2], 'g', label='ttt_nfsp_0.2_draw')
    plt.legend(handles=[line1, line2, line3], loc='lower right', prop={'size': 9})
    plt.ylabel('win_rate')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(312)
    ax2.set_title('random agent0 against trained agent1(eta 0 in evaluation)')
    plt.ylim(0, 1)
    line1, = plt.plot(win_rates_against_random0_eta0[2][0], 'r', label='ttt_nfsp_0.2_random_agent0')
    line2, = plt.plot(win_rates_against_random0_eta0[2][1], 'b', label='ttt_nfsp_0.2_traomed_agent1')
    line3, = plt.plot(win_rates_against_random0_eta0[2][2], 'g', label='ttt_nfsp_0.2_draw')
    plt.legend(handles=[line1, line2, line3], loc='lower right', prop={'size': 9})
    plt.ylabel('win_rate')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(313)
    ax2.set_title("win rates against each other(eta 0 for both in evaluation)")
    plt.ylim(0, 1)
    line1, = plt.plot(win_rates_against_eachother_eta0[2][0], 'b', label='ttt_nfsp_0.2_agent0')
    line2, = plt.plot(win_rates_against_eachother_eta0[2][1], 'r', label='ttt_nfsp_0.2_agent1')
    line3, = plt.plot(win_rates_against_eachother_eta0[2][2], 'g', label='ttt_nfsp_0.2_draw')
    plt.legend(handles=[line1, line2, line3], loc='lower right', prop={'size': 9})
    plt.ylabel('win_rate')
    plt.xlabel('episode(*1e4)')
    plt.show()

    # plot random agent0 against trained agent1 in ttt_nfsp_0.1_7_26

    # plt win rates of ttt_nfsp_1_7_29, trained agent0 against random agent1
    plt.figure(figsize=(15, 12))
    ax2 = plt.subplot(311)
    ax2.set_title("trained agent0 against random agent1(eta 0 in evaluation)")
    plt.ylim(0, 1)
    line1, = plt.plot(win_rates_against_random1_eta1[0][0], 'b', label='ttt_nfsp_1_trained_agent0')
    line2, = plt.plot(win_rates_against_random1_eta1[0][1], 'r', label='ttt_nfsp_1_random_agent1')
    line3, = plt.plot(win_rates_against_random1_eta1[0][2], 'g', label='ttt_nfsp_1_draw')
    plt.legend(handles=[line1, line2, line3], loc='lower right', prop={'size': 9})
    plt.ylabel('win_rate')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(312)
    ax2.set_title('random agent0 against trained agent1(eta 0 in evaluation)')
    plt.ylim(0, 1)
    line1, = plt.plot(win_rates_against_random0_eta1[0][0], 'r', label='ttt_nfsp_1_random_agent0')
    line2, = plt.plot(win_rates_against_random0_eta1[0][1], 'b', label='ttt_nfsp_1_traomed_agent1')
    line3, = plt.plot(win_rates_against_random0_eta1[0][2], 'g', label='ttt_nfsp_1_draw')
    plt.legend(handles=[line1, line2, line3], loc='lower right', prop={'size': 9})
    plt.ylabel('win_rate')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(313)
    ax2.set_title("win rates against each other(eta 0 for both in evaluation)")
    plt.ylim(0, 1)
    line1, = plt.plot(win_rates_against_eachother_eta1[0][0], 'b', label='ttt_nfsp_1_agent0')
    line2, = plt.plot(win_rates_against_eachother_eta1[0][1], 'r', label='ttt_nfsp_1_agent1')
    line3, = plt.plot(win_rates_against_eachother_eta1[0][2], 'g', label='ttt_nfsp_1_draw')
    plt.legend(handles=[line1, line2, line3], loc='lower right', prop={'size': 9})
    plt.ylabel('win_rate')
    plt.xlabel('episode(*1e4)')
    plt.show()

if __name__ == "__main__":
    app.run(main)