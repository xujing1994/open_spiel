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

def au_mean(list, start):
    return sum(list[start:]) / len(list[start:])


def main(argv):
    kuhn_poker_nfsp_0 = "/home/jxu8/Code_update/open_spiel/evaluation_data/eval_kp_nfsp_0.1_7_27/"
    kuhn_poker_nfsp_1 = "/home/jxu8/Code_update/open_spiel/evaluation_data/eval_kp_nfsp_1_7_28/"
    ttt_nfsp_0 = "/home/jxu8/Code_update/open_spiel/evaluation_data/eval_ttt_nfsp_0.1_7_26/"
    ttt_nfsp_1 = "/home/jxu8/Code_update/open_spiel/evaluation_data/eval_ttt_nfsp_1_7_29/"

    kuhn_poker_psro = "/home/jxu8/Code/open_spiel/evaluation_data/eval_kuhn_poker_psro_7_2/"

    avg_rewards_0_against_eachother = []
    avg_rewards_1_against_eachother = []
    avg_rewards_0_against_random_agent1 = []
    avg_rewards_0_against_random_agent0 = []
    avg_rewards_1_against_random_agent1 = []
    avg_rewards_1_against_random_agent0 = []
    avg_rewards_0_between_random_agents = []

    au_0_against_eachother_mean = []
    au_1_against_eachother_mean = []
    au_0_against_random_agent1_mean = []
    au_0_against_random_agent0_mean = []
    au_1_against_random_agent1_mean = []
    au_1_against_random_agent0_mean = []
                                                                                                               # (trained agetn0 use best response policy only and trained agent 1 use average policy only)
    # load avg_utility against eachother with eta 0 in training
    avg_rewards_0_against_eachother.append(read_loss(kuhn_poker_nfsp_0 + 'avg_utility_5000/eta_0/avg_utility_against_eachother.txt'))
    avg_rewards_0_against_eachother.append(read_loss(kuhn_poker_nfsp_0 + 'avg_utility/eta_1/avg_utility_against_eachother.txt'))
    avg_rewards_0_against_eachother.append(read_loss(kuhn_poker_nfsp_0 + 'avg_utility/eta_0.1/avg_utility_against_eachother.txt'))
    avg_rewards_0_against_eachother.append(read_loss(kuhn_poker_nfsp_0 + 'avg_utility/eta_0_1/avg_utility_against_eachother.txt'))
    avg_rewards_0_against_eachother.append(read_loss(kuhn_poker_nfsp_0 + 'avg_utility/eta_1_0/avg_utility_against_eachother.txt'))

    au_0_against_eachother_mean.append([au_mean(avg_rewards_0_against_eachother[0][0], 200), au_mean(avg_rewards_0_against_eachother[0][1], 200)])

    avg_rewards_0_against_eachother.append(read_loss(ttt_nfsp_0 + 'avg_utility/eta_0/avg_utility_against_eachother.txt'))
    avg_rewards_0_against_eachother.append(read_loss(ttt_nfsp_0 + 'avg_utility/eta_1/avg_utility_against_eachother.txt'))
    avg_rewards_0_against_eachother.append(read_loss(ttt_nfsp_0 + 'avg_utility/eta_0.1/avg_utility_against_eachother.txt'))
    avg_rewards_0_against_eachother.append(read_loss(ttt_nfsp_0 + 'avg_utility/eta_0_1/avg_utility_against_eachother.txt'))
    avg_rewards_0_against_eachother.append(read_loss(ttt_nfsp_0 + 'avg_utility/eta_1_0/avg_utility_against_eachother.txt'))
    # load avg_utility against eachother with eta 1 in training
    avg_rewards_1_against_eachother.append(read_loss(kuhn_poker_nfsp_1 + 'avg_utility_5000/eta_0/avg_utility_against_eachother.txt'))
    avg_rewards_1_against_eachother.append(read_loss(kuhn_poker_nfsp_1 + 'avg_utility/eta_1/avg_utility_against_eachother.txt'))
    avg_rewards_1_against_eachother.append(read_loss(kuhn_poker_nfsp_1 + 'avg_utility/eta_0.1/avg_utility_against_eachother.txt'))
    avg_rewards_1_against_eachother.append(read_loss(kuhn_poker_nfsp_1 + 'avg_utility/eta_0_1/avg_utility_against_eachother.txt'))
    avg_rewards_1_against_eachother.append(read_loss(kuhn_poker_nfsp_1 + 'avg_utility/eta_1_0/avg_utility_against_eachother.txt'))

    au_1_against_eachother_mean.append([au_mean(avg_rewards_1_against_eachother[0][0], 200), au_mean(avg_rewards_1_against_eachother[0][1], 200)])

    avg_rewards_1_against_eachother.append(read_loss(ttt_nfsp_1 + 'avg_utility/eta_0/avg_utility_against_eachother.txt'))
    avg_rewards_1_against_eachother.append(read_loss(ttt_nfsp_1 + 'avg_utility/eta_1/avg_utility_against_eachother.txt'))
    avg_rewards_1_against_eachother.append(read_loss(ttt_nfsp_1 + 'avg_utility/eta_0.1/avg_utility_against_eachother.txt'))
    avg_rewards_1_against_eachother.append(read_loss(ttt_nfsp_1 + 'avg_utility/eta_0_1/avg_utility_against_eachother.txt'))
    avg_rewards_1_against_eachother.append(read_loss(ttt_nfsp_1 + 'avg_utility/eta_1_0/avg_utility_against_eachother.txt'))

    avg_rewards_0_against_random_agent1.append(read_loss(kuhn_poker_nfsp_0 + 'avg_utility_5000/eta_0/avg_utility_against_random_agent1.txt'))
    avg_rewards_0_against_random_agent0.append(read_loss(kuhn_poker_nfsp_0 + 'avg_utility_5000/eta_0/avg_utility_against_random_agent0.txt'))
    avg_rewards_1_against_random_agent1.append(read_loss(kuhn_poker_nfsp_1 + 'avg_utility_5000/eta_0/avg_utility_against_random_agent1.txt'))
    avg_rewards_1_against_random_agent0.append(read_loss(kuhn_poker_nfsp_1 + 'avg_utility_5000/eta_0/avg_utility_against_random_agent0.txt'))

    au_0_against_random_agent1_mean.append([au_mean(avg_rewards_0_against_random_agent1[0][0], 200), au_mean(avg_rewards_0_against_random_agent1[0][1], 200)])
    au_0_against_random_agent0_mean.append([au_mean(avg_rewards_0_against_random_agent0[0][0], 200), au_mean(avg_rewards_0_against_random_agent0[0][1], 200)])
    au_1_against_random_agent1_mean.append([au_mean(avg_rewards_1_against_random_agent1[0][0], 200), au_mean(avg_rewards_1_against_random_agent1[0][1], 200)])
    au_1_against_random_agent0_mean.append([au_mean(avg_rewards_1_against_random_agent0[0][0], 200), au_mean(avg_rewards_1_against_random_agent0[0][1], 200)])

    avg_rewards_0_between_random_agents.append(read_loss(kuhn_poker_nfsp_0 + 'avg_utility/eta_0/avg_utility_between_random_agents.txt'))


    # plt avg utility in kuhn_poker_nfsp_0.1_7_27
    plt.figure(figsize=(15, 10))
    ax2 = plt.subplot(211)
    y_ticks = np.arange(-0.20, 0.25, 0.05)
    ax2.set_title("average utility with eta 0 in evaluation (0.1 in training)")
    line1, = plt.plot(avg_rewards_0_against_eachother[0][0], "b", label="agent0, {}".format(au_0_against_eachother_mean[0][0]))
    line2, = plt.plot(avg_rewards_0_against_eachother[0][1], "r", label="agent1, {}".format(au_0_against_eachother_mean[0][1]))
    plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2], loc='upper right')
    plt.ylabel('avg utility')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(212)
    y_ticks = np.arange(-0.20, 0.25, 0.05)
    ax2.set_title("average utility with eta 0 in evaluation (1 in training)")
    line1, = plt.plot(avg_rewards_1_against_eachother[0][0], "b", label="agent0, {}".format(au_1_against_eachother_mean[0][0]))
    line2, = plt.plot(avg_rewards_1_against_eachother[0][1], "r", label="agent1, {}".format(au_1_against_eachother_mean[0][1]))
    plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2], loc='upper right')
    plt.ylabel('avg utility')
    plt.xlabel('episode(*1e4)')
    plt.show()


    # ax2 = plt.subplot(512)
    # ax2.set_title("average utility with eta 1 in evaluation (0.1 in training)")
    # line1, = plt.plot(avg_rewards_0_against_eachother[1][0], "b", label="agent0")
    # line2, = plt.plot(avg_rewards_0_against_eachother[1][1], "r", label="agent1")
    # plt.legend(handles=[line1, line2], loc='upper right')
    # plt.ylabel('avg utility')
    # plt.xlabel('episode(*1e4)')
    #
    # ax2 = plt.subplot(513)
    # ax2.set_title("average utility with eta 0.1 in evaluation (0.1 in training)")
    # line1, = plt.plot(avg_rewards_0_against_eachother[2][0], "b", label="agent0")
    # line2, = plt.plot(avg_rewards_0_against_eachother[2][1], "r", label="agent1")
    # plt.legend(handles=[line1, line2], loc='upper right')
    # plt.ylabel('avg utility')
    # plt.xlabel('episode(*1e4)')
    #
    # ax2 = plt.subplot(514)
    # ax2.set_title("average utility with eta 0_1 in evaluation (0.1 in training)")
    # line1, = plt.plot(avg_rewards_0_against_eachother[3][0], "b", label="agent0")
    # line2, = plt.plot(avg_rewards_0_against_eachother[3][1], "r", label="agent1")
    # plt.legend(handles=[line1, line2], loc='upper right')
    # plt.ylabel('avg utility')
    # plt.xlabel('episode(*1e4)')
    #
    # ax2 = plt.subplot(515)
    # ax2.set_title("average utility with eta 1_0 in evaluation (0.1 in training)")
    # line1, = plt.plot(avg_rewards_0_against_eachother[4][0], "b", label="agent0")
    # line2, = plt.plot(avg_rewards_0_against_eachother[4][1], "r", label="agent1")
    # plt.legend(handles=[line1, line2], loc='upper right')
    # plt.ylabel('avg utility')
    # plt.xlabel('episode(*1e4)')
    # plt.show()


    # plt avg utility in kuhn_poker_nfsp_1_7_28
    # plt.figure(figsize=(15, 20))
    # ax2 = plt.subplot(511)
    # y_ticks = np.arange(-0.20, 0.25, 0.05)
    # ax2.set_title("average utility with eta 0 in evaluation (1 in training)")
    # line1, = plt.plot(avg_rewards_1_against_eachother[0][0], "b", label="agent0, {}".format(au_1_against_eachother_mean[0][0]))
    # line2, = plt.plot(avg_rewards_1_against_eachother[0][1], "r", label="agent1, {}".format(au_1_against_eachother_mean[0][1]))
    # plt.yticks(y_ticks)
    # plt.legend(handles=[line1, line2], loc='upper right')
    # plt.ylabel('avg utility')
    # plt.xlabel('episode(*1e4)')
    #
    # ax2 = plt.subplot(512)
    # ax2.set_title("average utility with eta 1 in evaluation (1 in training)")
    # line1, = plt.plot(avg_rewards_1_against_eachother[1][0], "b", label="agent0")
    # line2, = plt.plot(avg_rewards_1_against_eachother[1][1], "r", label="agent1")
    # plt.legend(handles=[line1, line2], loc='upper right')
    # plt.ylabel('avg utility')
    # plt.xlabel('episode(*1e4)')
    #
    # ax2 = plt.subplot(513)
    # ax2.set_title("average utility with eta 0.1 in evaluation (1 in training)")
    # line1, = plt.plot(avg_rewards_1_against_eachother[2][0], "b", label="agent0")
    # line2, = plt.plot(avg_rewards_1_against_eachother[2][1], "r", label="agent1")
    # plt.legend(handles=[line1, line2], loc='upper right')
    # plt.ylabel('avg utility')
    # plt.xlabel('episode(*1e4)')
    #
    # ax2 = plt.subplot(514)
    # ax2.set_title("average utility with eta 0_1 in evaluation (1 in training)")
    # line1, = plt.plot(avg_rewards_1_against_eachother[3][0], "b", label="agent0")
    # line2, = plt.plot(avg_rewards_1_against_eachother[3][1], "r", label="agent1")
    # plt.legend(handles=[line1, line2], loc='upper right')
    # plt.ylabel('avg utility')
    # plt.xlabel('episode(*1e4)')
    #
    # ax2 = plt.subplot(515)
    # ax2.set_title("average utility with eta 1_0 in evaluation (1 in training)")
    # line1, = plt.plot(avg_rewards_1_against_eachother[4][0], "b", label="agent0")
    # line2, = plt.plot(avg_rewards_1_against_eachother[4][1], "r", label="agent1")
    # plt.legend(handles=[line1, line2], loc='upper right')
    # plt.ylabel('avg utility')
    # plt.xlabel('episode(*1e4)')
    # plt.show()

    plt.figure(figsize=(15, 10))
    ax2 = plt.subplot(211)
    y_ticks = np.arange(-0.20, 0.25, 0.05)
    ax2.set_title("average utility(trained agent0 vs random agent1) with eta 0 in evaluation (0.1 in training)")
    line1, = plt.plot(avg_rewards_0_against_random_agent1[0][0], "b", label="trained_agent0, {}".format(au_0_against_random_agent1_mean[0][0]))
    line2, = plt.plot(avg_rewards_0_against_random_agent1[0][1], "r", label="random_agent1, {}".format(au_0_against_random_agent1_mean[0][1]))
    #plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2], loc='upper right')
    plt.ylabel('avg utility')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(212)
    y_ticks = np.arange(-0.20, 0.25, 0.05)
    ax2.set_title("average utility(random agent0 vs trained agent1) with eta 0 in evaluation (0.1 in training)")
    line1, = plt.plot(avg_rewards_0_against_random_agent0[0][0], "r", label="random_agent0, {}".format(au_0_against_random_agent0_mean[0][0]))
    line2, = plt.plot(avg_rewards_0_against_random_agent0[0][1], "b", label="trained_agent1, {}".format(au_0_against_random_agent0_mean[0][1]))
    #plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2], loc='upper right')
    plt.ylabel('avg utility')
    plt.xlabel('episode(*1e4)')
    plt.show()

    plt.figure(figsize=(15, 10))
    ax2 = plt.subplot(211)
    y_ticks = np.arange(-0.20, 0.25, 0.05)
    ax2.set_title("average utility(trained agent0 vs random agent1) with eta 0 in evaluation (1 in training)")
    line1, = plt.plot(avg_rewards_1_against_random_agent1[0][0], "b", label="trained_agent0, {}".format(au_1_against_random_agent1_mean[0][0]))
    line2, = plt.plot(avg_rewards_1_against_random_agent1[0][1], "r", label="random_agent1, {}".format(au_1_against_random_agent1_mean[0][1]))
    #plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2], loc='upper right')
    plt.ylabel('avg utility')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(212)
    y_ticks = np.arange(-0.20, 0.25, 0.05)
    ax2.set_title("average utility(random agent0 vs trained agent1) with eta 0 in evaluation (1 in training)")
    line1, = plt.plot(avg_rewards_1_against_random_agent0[0][0], "r", label="random agent0, {}".format(au_1_against_random_agent0_mean[0][0]))
    line2, = plt.plot(avg_rewards_1_against_random_agent0[0][1], "b", label="trained_agent1, {}".format(au_1_against_random_agent0_mean[0][1]))
    #plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2], loc='upper right')
    plt.ylabel('avg utility')
    plt.xlabel('episode(*1e4)')
    plt.show()

    plt.figure(figsize=(15, 5))
    y_ticks = np.arange(-0.20, 0.25, 0.05)
    ax2.set_title("average utility(trained agent0 vs random agent1) with eta 0 in evaluation (1 in training)")
    line1, = plt.plot(avg_rewards_0_between_random_agents[0][0], "b", label="random_agent0, {}".format(au_mean(avg_rewards_0_between_random_agents[0][0], 200)))
    line2, = plt.plot(avg_rewards_0_between_random_agents[0][1], "r", label="random_agent1, {}".format(au_mean(avg_rewards_0_between_random_agents[0][1], 200)))
    #plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2], loc='upper right')
    plt.ylabel('avg utility')
    plt.xlabel('episode(*1e4)')
    plt.show()



if __name__ == "__main__":
    app.run(main)