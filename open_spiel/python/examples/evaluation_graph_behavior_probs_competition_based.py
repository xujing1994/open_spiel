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

def main(argv):
    kuhn_poker_nfsp_0 = "/home/jxu8/Code_update/open_spiel/evaluation_data/eval_kp_nfsp_0.1_7_27/"
    kuhn_poker_nfsp_1 = "/home/jxu8/Code_update/open_spiel/evaluation_data/eval_kp_nfsp_1_7_28/"
    ttt_nfsp_0 = "/home/jxu8/Code_update/open_spiel/evaluation_data/eval_ttt_nfsp_0.1_7_26/"
    ttt_nfsp_1 = "/home/jxu8/Code_update/open_spiel/evaluation_data/eval_ttt_nfsp_1_7_29/"

    kuhn_poker_psro = "/home/jxu8/Code/open_spiel/evaluation_data/eval_kuhn_poker_psro_7_2/"

    bp_jk_cb = []
    bp_jq_cb = []
    bp_kj_cb = []
    bp_kq_cb = []
    bp_qj_cb = []
    bp_qk_cb = []

    bp_jk_cb.append(read_behavior_probs(kuhn_poker_nfsp_0 + 'behavior_probs/eta_0/competition_based/JK.txt'))
    bp_jq_cb.append(read_behavior_probs(kuhn_poker_nfsp_0 + 'behavior_probs/eta_0/competition_based/JQ.txt'))
    bp_kj_cb.append(read_behavior_probs(kuhn_poker_nfsp_0 + 'behavior_probs/eta_0/competition_based/KJ.txt'))
    bp_kq_cb.append(read_behavior_probs(kuhn_poker_nfsp_0 + 'behavior_probs/eta_0/competition_based/KQ.txt'))
    bp_qj_cb.append(read_behavior_probs(kuhn_poker_nfsp_0 + 'behavior_probs/eta_0/competition_based/QJ.txt'))
    bp_qk_cb.append(read_behavior_probs(kuhn_poker_nfsp_0 + 'behavior_probs/eta_0/competition_based/QK.txt'))

    bp_jk_cb.append(read_behavior_probs(kuhn_poker_nfsp_1 + 'behavior_probs/eta_0/competition_based/JK.txt'))
    bp_jq_cb.append(read_behavior_probs(kuhn_poker_nfsp_1 + 'behavior_probs/eta_0/competition_based/JQ.txt'))
    bp_kj_cb.append(read_behavior_probs(kuhn_poker_nfsp_1 + 'behavior_probs/eta_0/competition_based/KJ.txt'))
    bp_kq_cb.append(read_behavior_probs(kuhn_poker_nfsp_1 + 'behavior_probs/eta_0/competition_based/KQ.txt'))
    bp_qj_cb.append(read_behavior_probs(kuhn_poker_nfsp_1 + 'behavior_probs/eta_0/competition_based/QJ.txt'))
    bp_qk_cb.append(read_behavior_probs(kuhn_poker_nfsp_1 + 'behavior_probs/eta_0/competition_based/QK.txt'))

    #plt alpha in kuhn_poker_nfsp_0.1(eta 0)
    tmp_list = [bp_jk_cb[0], bp_jq_cb[0], bp_kj_cb[0], bp_kq_cb[0], bp_qj_cb[0], bp_qk_cb[0]]
    alpha_1 = [1 - tmp_list[0][0][i] for i in range(len(tmp_list[0][0]))]
    alpha_2 = [1 - tmp_list[1][0][i] for i in range(len(tmp_list[1][0]))]
    alpha_3 = [(1/3) * (1 - tmp_list[2][0][i]) for i in range(len(tmp_list[2][0]))]
    alpha_4 = [(1/3) * (1 - tmp_list[3][0][i]) for i in range(len(tmp_list[2][0]))]
    alpha_5 = [tmp_list[4][7][i] - 1/3 for i in range(len(tmp_list[4][7]))]
    alpha_6 = [tmp_list[5][7][i] - 1/3 for i in range(len(tmp_list[5][7]))]

    ax2 = plt.figure(figsize=(10, 5))
    #ax2.set_title("JK (kuhn_poker_nfsp_0.1, eta0.1 in evaluation)")
    #plt.ylim(0, 0.35)
    line1, = plt.plot(alpha_1, "b-", label="JK & JQ")
    #line2, = plt.plot(alpha_2, "b*", label="2")
    line3, = plt.plot(alpha_3, "g-", label="KJ & KQ")
    #line4, = plt.plot(alpha_4, "g*", label="4")
    line5, = plt.plot(alpha_5, "y-", label="QJ & QK")
    #line6, = plt.plot(alpha_6, "y*", label="6")
    #plt.legend(handles=[line1, line2, line3, line4, line5, line6], loc='upper right')
    plt.legend(handles=[line1, line3, line5], loc='upper right')
    plt.ylabel('alpha')
    plt.xlabel('episode(*1e4)')
    plt.show()

    #plt alpha in kuhn_poker_nfsp_1(eta 0)
    tmp_list = [bp_jk_cb[1], bp_jq_cb[1], bp_kj_cb[1], bp_kq_cb[1], bp_qj_cb[1], bp_qk_cb[1]]
    alpha_1 = [1 - tmp_list[0][0][i] for i in range(len(tmp_list[0][0]))]
    alpha_2 = [1 - tmp_list[1][0][i] for i in range(len(tmp_list[1][0]))]
    alpha_3 = [(1/3) * (1 - tmp_list[2][0][i]) for i in range(len(tmp_list[2][0]))]
    alpha_4 = [(1/3) * (1 - tmp_list[3][0][i]) for i in range(len(tmp_list[2][0]))]
    alpha_5 = [tmp_list[4][7][i] - 1/3 for i in range(len(tmp_list[4][7]))]
    alpha_6 = [tmp_list[5][7][i] - 1/3 for i in range(len(tmp_list[5][7]))]

    ax2 = plt.figure(figsize=(10, 5))
    #ax2.set_title("JK (kuhn_poker_nfsp_0.1, eta0.1 in evaluation)")
    #plt.ylim(0, 0.35)
    line1, = plt.plot(alpha_1, "b-", label="JK & JQ")
    #line2, = plt.plot(alpha_2, "b*", label="2")
    line3, = plt.plot(alpha_3, "g-", label="KJ & KQ")
    #line4, = plt.plot(alpha_4, "g*", label="4")
    line5, = plt.plot(alpha_5, "y-", label="QJ & QK")
    #line6, = plt.plot(alpha_6, "y*", label="6")
    #plt.legend(handles=[line1, line2, line3, line4, line5, line6], loc='upper right')
    plt.legend(handles=[line1, line3, line5], loc='upper right')
    plt.ylabel('alpha')
    plt.xlabel('episode(*1e4)')
    plt.show()


    plt.figure(figsize=(10, 10))
    ax2 = plt.subplot(311)
    ax2.set_title("JK (kuhn_poker_nfsp_0.1, eta0 in evaluation)")
    #plt.ylim(0, 0.35)
    y_ticks = np.arange(0, 1.1, 0.1)
    line1, = plt.plot(bp_jk_cb[0][0], "b", label="1")
    line2, = plt.plot(bp_jk_cb[0][2], "r", label="2")
    line3, = plt.plot(bp_jk_cb[0][4], "g", label="3")
    line4, = plt.plot(bp_jk_cb[0][6], "y", label="4")
    plt.axhline(y=2/3,ls=":",c="blue")
    plt.axhline(y=1,ls=":",c="blue")
    plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2, line3, line4], loc='upper right')
    plt.ylabel('behavior_probs')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(312)
    ax2.set_title("JQ")
    #plt.ylim(0, 0.35)
    y_ticks = np.arange(0, 1.1, 0.1)
    line1, = plt.plot(bp_jq_cb[0][0], "b", label="1")
    line2, = plt.plot(bp_jq_cb[0][2], "r", label="2")
    line3, = plt.plot(bp_jq_cb[0][4], "g", label="3")
    line4, = plt.plot(bp_jq_cb[0][6], "y", label="4")
    plt.axhline(y=2/3,ls=":",c="blue")
    plt.axhline(y=1,ls=":",c="blue")
    plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2, line3, line4], loc='upper right')
    plt.ylabel('behavior_probs')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(313)
    ax2.set_title("KJ")
    #plt.ylim(0, 0.35)
    y_ticks = np.arange(0, 1.1, 0.1)
    line1, = plt.plot(bp_kj_cb[0][0], "b", label="1")
    line2, = plt.plot(bp_kj_cb[0][2], "r", label="2")
    line3, = plt.plot(bp_kj_cb[0][4], "g", label="3")
    line4, = plt.plot(bp_kj_cb[0][6], "y", label="4")
    plt.axhline(y=0,ls=":",c="blue")
    plt.axhline(y=1,ls=":",c="blue")
    plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2, line3, line4], loc='upper right')
    plt.ylabel('behavior_probs')
    plt.xlabel('episode(*1e4)')
    plt.show()

    plt.figure(figsize=(10, 10))
    ax2 = plt.subplot(311)
    ax2.set_title("KQ")
    #plt.ylim(0, 0.35)
    y_ticks = np.arange(0, 1.1, 0.1)
    line1, = plt.plot(bp_kq_cb[0][0], "b", label="1")
    line2, = plt.plot(bp_kq_cb[0][2], "r", label="2")
    line3, = plt.plot(bp_kq_cb[0][4], "g", label="3")
    line4, = plt.plot(bp_kq_cb[0][6], "y", label="4")
    plt.axhline(y=0,ls=":",c="blue")
    plt.axhline(y=1,ls=":",c="blue")
    plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2, line3, line4], loc='upper right')
    plt.ylabel('behavior_probs')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(312)
    ax2.set_title("QJ")
    #plt.ylim(0, 0.35)
    y_ticks = np.arange(0, 1.1, 0.1)
    line1, = plt.plot(bp_qj_cb[0][0], "b", label="1")
    line2, = plt.plot(bp_qj_cb[0][2], "r", label="2")
    line3, = plt.plot(bp_qj_cb[0][4], "g", label="3")
    line4, = plt.plot(bp_qj_cb[0][6], "y", label="4")
    plt.axhline(y=1/3,ls=":",c="yellow")
    plt.axhline(y=2/3,ls=":",c="yellow")
    plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2, line3, line4], loc='upper right')
    plt.ylabel('behavior_probs')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(313)
    ax2.set_title("QK")
    #plt.ylim(0, 0.35)
    y_ticks = np.arange(0, 1.1, 0.1)
    line1, = plt.plot(bp_qk_cb[0][0], "b", label="1")
    line2, = plt.plot(bp_qk_cb[0][2], "r", label="2")
    line3, = plt.plot(bp_qk_cb[0][4], "g", label="3")
    line4, = plt.plot(bp_qk_cb[0][6], "y", label="4")
    plt.axhline(y=1/3,ls=":",c="yellow")
    plt.axhline(y=2/3,ls=":",c="yellow")
    plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2, line3, line4], loc='upper right')
    plt.ylabel('behavior_probs')
    plt.xlabel('episode(*1e4)')
    plt.show()

    # plt bp for kuhn_poker_nfsp_0.1, eta1 in evaluation
    plt.figure(figsize=(10, 10))
    ax2 = plt.subplot(311)
    ax2.set_title("JK (kuhn_poker_nfsp_1, eta0 in evaluation)")
    #plt.ylim(0, 0.35)
    y_ticks = np.arange(0, 1.1, 0.1)
    line1, = plt.plot(bp_jk_cb[1][0], "b", label="1")
    line2, = plt.plot(bp_jk_cb[1][2], "r", label="2")
    line3, = plt.plot(bp_jk_cb[1][4], "g", label="3")
    line4, = plt.plot(bp_jk_cb[1][6], "y", label="4")
    plt.axhline(y=2/3,ls=":",c="blue")
    plt.axhline(y=1,ls=":",c="blue")
    plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2, line3, line4], loc='upper right')
    plt.ylabel('behavior_probs')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(312)
    ax2.set_title("JQ")
    #plt.ylim(0, 0.35)
    y_ticks = np.arange(0, 1.1, 0.1)
    line1, = plt.plot(bp_jq_cb[1][0], "b", label="1")
    line2, = plt.plot(bp_jq_cb[1][2], "r", label="2")
    line3, = plt.plot(bp_jq_cb[1][4], "g", label="3")
    line4, = plt.plot(bp_jq_cb[1][6], "y", label="4")
    plt.axhline(y=2/3,ls=":",c="blue")
    plt.axhline(y=1,ls=":",c="blue")
    plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2, line3, line4], loc='upper right')
    plt.ylabel('behavior_probs')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(313)
    ax2.set_title("KJ")
    #plt.ylim(0, 0.35)
    y_ticks = np.arange(0, 1.1, 0.1)
    line1, = plt.plot(bp_kj_cb[1][0], "b", label="1")
    line2, = plt.plot(bp_kj_cb[1][2], "r", label="2")
    line3, = plt.plot(bp_kj_cb[1][4], "g", label="3")
    line4, = plt.plot(bp_kj_cb[1][6], "y", label="4")
    plt.axhline(y=0,ls=":",c="blue")
    plt.axhline(y=1,ls=":",c="blue")
    plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2, line3, line4], loc='upper right')
    plt.ylabel('behavior_probs')
    plt.xlabel('episode(*1e4)')
    plt.show()

    plt.figure(figsize=(10, 10))
    ax2 = plt.subplot(311)
    ax2.set_title("KQ")
    #plt.ylim(0, 0.35)
    y_ticks = np.arange(0, 1.1, 0.1)
    line1, = plt.plot(bp_kq_cb[1][0], "b", label="1")
    line2, = plt.plot(bp_kq_cb[1][2], "r", label="2")
    line3, = plt.plot(bp_kq_cb[1][4], "g", label="3")
    line4, = plt.plot(bp_kq_cb[1][6], "y", label="4")
    plt.axhline(y=0,ls=":",c="blue")
    plt.axhline(y=1,ls=":",c="blue")
    plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2, line3, line4], loc='upper right')
    plt.ylabel('behavior_probs')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(312)
    ax2.set_title("QJ")
    #plt.ylim(0, 0.35)
    y_ticks = np.arange(0, 1.1, 0.1)
    line1, = plt.plot(bp_qj_cb[1][0], "b", label="1")
    line2, = plt.plot(bp_qj_cb[1][2], "r", label="2")
    line3, = plt.plot(bp_qj_cb[1][4], "g", label="3")
    line4, = plt.plot(bp_qj_cb[1][6], "y", label="4")
    plt.axhline(y=1/3,ls=":",c="yellow")
    plt.axhline(y=2/3,ls=":",c="yellow")
    plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2, line3, line4], loc='upper right')
    plt.ylabel('behavior_probs')
    plt.xlabel('episode(*1e4)')

    ax2 = plt.subplot(313)
    ax2.set_title("QK")
    #plt.ylim(0, 0.35)
    y_ticks = np.arange(0, 1.1, 0.1)
    line1, = plt.plot(bp_qk_cb[1][0], "b", label="1")
    line2, = plt.plot(bp_qk_cb[1][2], "r", label="2")
    line3, = plt.plot(bp_qk_cb[1][4], "g", label="3")
    line4, = plt.plot(bp_qk_cb[1][6], "y", label="4")
    plt.axhline(y=1/3,ls=":",c="yellow")
    plt.axhline(y=2/3,ls=":",c="yellow")
    plt.yticks(y_ticks)
    plt.legend(handles=[line1, line2, line3, line4], loc='upper right')
    plt.ylabel('behavior_probs')
    plt.xlabel('episode(*1e4)')
    plt.show()

if __name__ == "__main__":
    app.run(main)