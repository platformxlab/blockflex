import datetime as dt

BLOCKFLEX_DIR="/home/osdi22ae/BlockFlex/blockflex/" # CHANGE THIS PATH
DEBUG = True
def log_msg(*msg, out_f=None):
    '''
    Log a message with the current time stamp.
    '''
    msg = [str(_) for _ in msg]
    if DEBUG:
        if out_f is not None:
            print("[%s] %s" % ((dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), " ".join(msg)), file=out_f)
            out_f.flush()
        else:
            print("[%s] %s" % ((dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), " ".join(msg)))