import threading
import lstm_sz
import lstm_bw
import lstm_dur
import queue
import sys


def main(out_f=None):
    thr = []
    bw_q = queue.Queue()
    sz_q = queue.Queue()
    comb_q = queue.Queue()
    #BW PREDICTOR
    t_thread = threading.Thread(target=lstm_bw.main, args=(bw_q,out_f))
    thr.append(t_thread)
    #SZ PREDICTOR
    t_thread = threading.Thread(target=lstm_sz.main, args=(sz_q,out_f))
    thr.append(t_thread)
    #BW DUR PREDICTOR
    t_thread = threading.Thread(target=lstm_dur.main, args=(bw_q,comb_q,True,True,out_f))
    thr.append(t_thread)
    #SZ DUR PREDICTOR
    t_thread = threading.Thread(target=lstm_dur.main, args=(sz_q,comb_q,False,True,out_f))
    thr.append(t_thread)

    #START THREADS
    for thread in thr: 
        thread.start()

    #JOIN THREADS
    for thread in thr:
        thread.join()

if __name__ == "__main__":
    out_f = None
    if len(sys.argv) >= 2:
        outfile = sys.argv[1]
        out_f = open(outfile, "w+")
    main(out_f)
