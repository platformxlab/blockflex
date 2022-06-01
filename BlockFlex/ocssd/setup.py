#!/usr/bin/python3

from macros import *

def set_target(ids):
    for id in ids:
        log_msg(f" ==== Creating iSCSI target {id} ====", emphasize=True)
        # log_msg(f"###### Setup iSCSI target {id}")
        # log_msg(ISCSI_TARGET_IMGS)
        TGT_BS=ISCSI_TARGET_IMGS[id]
        ISCSI_TARGET_NAME=ISCSI_TARGET_NAMES[id]

        run_command(f"mkdir -p {IMG_DIR}")
        run_command(f"fallocate -l {DUMMY_DISK_SIZE} {TGT_BS}")

        run_command(f"sudo {PFE_USR_DIR}/tgtadm  --lld iscsi --op new --mode target --tid {id} -T {ISCSI_TARGET_NAME}")
        run_command(f"sudo {PFE_USR_DIR}/tgtadm  --lld iscsi --op new --mode logicalunit --tid {id} --lun 1 -b {TGT_BS}")
        run_command(f"sudo {PFE_USR_DIR}/tgtadm   --lld iscsi --op bind --mode target --tid {id} -I ALL")
    # run_command(f"sudo {PFE_USR_DIR}/tgtadm   --lld iscsi --op show --mode target")

def set_initiator(ids):
    log_msg(f" ==== Starting iSCSI initiator ====", emphasize=True)
    run_command(f"sudo service open-iscsi stop && sudo service open-iscsi start", suppress_output=True)
    run_command(f"sudo iscsiadm --mode discovery --type sendtargets --portal {ISCSI_TARGET_IP}")
    for id in ids:
        ISCSI_TARGET_NAME=ISCSI_TARGET_NAMES[id]
        run_command(f"sudo iscsiadm --mode node --targetname {ISCSI_TARGET_NAME} --portal {ISCSI_TARGET_IP}:3260 --login")

def run_tgtd():
    log_msg(f" ==== Starting iSCSI daemon ====", emphasize=True)
    run_command(f"mkdir -p {PFE_LOG_DIR}")
    run_command(f"sudo rm {IO_HEAD_LOG} {IO_DATA_LOG} {IO_HEAD_SPLIT_LOG} {ERR_BLOCK_LOG} {TGTD_LOG}")
    run_command(f"touch {IO_HEAD_LOG} {IO_DATA_LOG} {IO_HEAD_SPLIT_LOG} {ERR_BLOCK_LOG} {TGTD_LOG}")
    run_command(f"sudo chmod 666 {IO_HEAD_LOG} {IO_DATA_LOG} {IO_HEAD_SPLIT_LOG} {ERR_BLOCK_LOG} {TGTD_LOG}")

    # Run tgtd
    run_command(f"sudo {PFE_USR_DIR}/tgtd -f --pfe-io-header-log {IO_HEAD_LOG} --pfe-fail-type-tgtd 0 --pfe-err-blk {ERR_BLOCK_LOG} --pfe-io-data-log {IO_DATA_LOG} --pfe-enable-record 0 >  {TGTD_LOG} 2>&1 &")
    # run_command(f"sudo {PFE_USR_DIR}/tgtd --pfe-io-header-log {IO_HEAD_LOG} --pfe-fail-type-tgtd 0 --pfe-err-blk {ERR_BLOCK_LOG} --pfe-io-data-log {IO_DATA_LOG} --pfe-enable-record 0 >  {TGTD_LOG} 2>&1 ")

def kill_tgtd():
    log_msg(f" ==== Killing iSCSI daemon ====", warning=True)
    run_command("sudo killall -9 tgtd")

def clean_logs():
    log_msg(f" ==== Cleaning log files ====", warning=True)
    #LOGS = [IO_HEAD_LOG, IO_DATA_LOG, IO_HEAD_SPLIT_LOG, ERR_BLOCK_LOG, TGTD_LOG]
    LOGS = [IO_HEAD_LOG, IO_DATA_LOG, IO_HEAD_SPLIT_LOG, ERR_BLOCK_LOG]
    for LOG in LOGS:
        if os.path.exists(LOG):
            run_command(f"cat /dev/null > {LOG}")

def start_log():
    run_command(f"touch {PFE_USR_DIR}/start_log")

def stop_log():
    run_command(f"rm -f {PFE_USR_DIR}/start_log")

def start_replay():
    run_command(f"touch {PFE_USR_DIR}/start_replay")

def stop_replay():
    run_command(f"rm -f {PFE_USR_DIR}/start_replay")

def reset_queues():
    log_msg(f" ==== Resetting message queues ====", warning=True)
    for queue in QUEUES:
        assert(queue != "" and queue != "/")
        run_command(f"rm -f {queue}")
        run_command(f"touch {queue}")

def reset_flags():
    log_msg(f" ==== Resetting flags ====", warning=True)
    run_command(f"rm -f {HARVEST_FLAG}")
    run_command(f"rm -f {END_FLAG}")

def clear_queues():
    log_msg(f" ==== Clearing message queues ====", warning=True)
    run_command(f"sudo ./mqueue_cleaner")
    run_command(f"sudo ./mqueue_cleaner")


def obtain_disk_id(ids):
    disk_ids = []
    raw_output = run_command("sudo lsblk -d", output=True)
    raw_output = raw_output['stdout']
    for line in raw_output.strip().split("\n"):
        line = list(filter(lambda x: len(x) > 0, line.strip().split(" ")))
        if line[3] == DUMMY_DISK_SIZE_STRING:
            disk_ids.append(line[0])

    # FIXME: find a way to restart the script instead of exit
    assert(len(disk_ids) == len(ids))
        

    log_msg("Virtual disk ids", disk_ids)

    return disk_ids

def kill_vms(vms):
    log_msg(f" ==== Killing VMs ====", warning=True)
    for vm in vms:
        run_command(f"sudo virsh shutdown {vm}")

    # check if all vms are shutdown  
    while True:
        raw_output = run_command("sudo virsh list --all --state-shutoff", output=True)
        raw_output = raw_output['stdout']
        all_shutdown = True
        for vm in vms:
            if vm not in raw_output:
                all_shutdown = False
                break
        if not all_shutdown:
            time.sleep(3)
        else:
            break

def start_vms(vms):
    log_msg(f" ==== Starting VMs ====", emphasize=True)
    for vm in vms:
        run_command(f"sudo virsh start {vm}")

    # check if all vms are running
    while True:
        raw_output = run_command("sudo virsh list --all --state-running", output=True)
        raw_output = raw_output['stdout']
        all_running = True
        for vm in vms:
            if vm not in raw_output:
                all_running = False
                break
        if not all_running:
            time.sleep(3)
        else:
            break
    
    # test ssh connection
    log_msg("Waiting for ssh connections (may take a while)")
    for vm in vms:
        while True:
            output = run_command(f"ssh {vm} 'date'", error=True, verbose=False)
            output = output['stderr']
            if "No route to host" in output:
                time.sleep(3)
            else:
                break

def attach_device(disk_ids, vms):
    log_msg(f" ==== Attaching VM devices ====", emphasize=True)
    for i, vm in enumerate(vms):
        run_command(f"sudo virsh attach-disk {vm} /dev/{disk_ids[i]} --target vdb")

def detach_device(vms):
    log_msg(f" ==== Detaching VM disks ====", warning=True)
    raw_output = run_command("sudo virsh list --all --state-running", output=True)
    raw_output = raw_output['stdout']
    for vm in vms:
        if vm not in raw_output:
            continue
        raw_output = run_command(f"ssh {vm} 'ls /dev/vd*'", output=True)
        raw_output = raw_output['stdout']
        ## FIXME: by default we assume the last disk in VM is the virtual disk 
        disk_id = raw_output.strip().split("\n")[-1].split("/")[-1]
        ## we only find the default disk -- no additional virtual disks are attached yet
        if "vda" in disk_id:
            continue

        run_command(f"sudo virsh detach-disk {vm} --target {disk_id}")

def run_workloads(with_vm, vms, workload="terasort", mode="harvest"):
    log_msg(f" ==== Running workload {workload} with {mode} mode ====", emphasize=True)
    if with_vm:
        run_command(f"ssh {vms[0]} 'sudo python3 workload_parser.py {workload} {workload}.out' &")
        if mode == "harvest":
            while not os.path.exists(HARVEST_FLAG):
                time.sleep(5)
            run_command(f"ssh {vms[1]} 'sudo python3 workload_parser.py ycsb ycsb.out ' &")
    else:
        run_command(f"sudo python3 workload.py {workload} {USR_DIR}/logs/{workload}.out &")
        if mode == "harvest":
            while not os.path.exists(HARVEST_FLAG):
                time.sleep(5)
            log_msg(f" ==== Running regular workload ycsb ====", emphasize=True)
            run_command(f"sudo python3 workload.py ycsb {USR_DIR}/logs/ycsb.out &")

def stop_workloads(with_vm, vms):
    log_msg(f" ==== Stopping workloads ====", warning=True)
    if with_vm:
        raw_output = run_command("sudo virsh list --all --state-running", output=True)
        raw_output = raw_output['stdout']
        for vm in vms:
            if vm not in raw_output:
                continue
            run_command("ssh "+vm+" \"ps ax | grep 'python3 workload_parser.py' | awk '{ print \$1 }' | xargs sudo kill -9\" ")
            run_command("ssh "+vm+" \"ps ax | grep 'python3 joblib' | awk '{ print \$1 }' | xargs sudo kill -9\" ")
    else:
        run_command("ps ax | grep 'python3 workload.py' | awk '{ print $1 }' | xargs sudo kill -9")
        run_command("ps ax | grep 'python3 joblib' | awk '{ print $1 }' | xargs sudo kill -9")

def stop_ocssd():
    log_msg(f" ==== Stopping blockflex ====", warning=True)
    run_command("sudo killall harvest")

def run_ocssd(workload="terasort", mode="harvest"):
    log_msg(f" ==== Running blockflex ====", emphasize=True)
    OUT_DIR = f"{USR_DIR}/results/{workload}"
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
        run_command(f"sudo chown $USER {OUT_DIR}")
    if mode == "static":
        run_command(f"sudo ../blockflex/harvest 0 0 > {OUT_DIR}/no_harvest_bw.out &")
    elif mode == "harvest":
        run_command(f"sudo ../blockflex/harvest 4 4 > {OUT_DIR}/harvest_bw.out &")
    elif mode == "unsold":
        run_command(f"sudo ../blockflex/harvest 0 4 > {OUT_DIR}/unsold_harvest_bw.out &")

def run_predictor():
    log_msg(f" ==== Running predictor ====", emphasize=True)
    run_command(f"echo '0' > {BLOCKFLEX_DIR}/bw_inputs.txt")
    run_command(f"echo '0' > {BLOCKFLEX_DIR}/sz_inputs.txt")
    run_command(f"sudo python3 ../blockflex/pred.py {USR_DIR}/logs/predictor.out &")

def stop_predictor():
    log_msg(f" ==== Stopping predictor ====", warning=True)
    run_command("ps ax | grep 'pred.py' | awk '{ print $1 }' | xargs sudo kill -9")

if __name__ == "__main__":
    ids = [1,2]
    vms = ["vm0", "vm1"]
    VERBOSE = True
    WITH_VM = False
    running_mode = 'regular'
    if len(sys.argv) >= 2:
        running_mode = sys.argv[1]

    stop_predictor()
    stop_workloads(WITH_VM, vms)
    stop_ocssd()
    clear_queues()

    if WITH_VM:
        detach_device(vms)
        kill_vms(vms)

    kill_tgtd()
    clean_logs()
    stop_log()
    stop_replay()
    reset_queues()
    reset_flags()

    if running_mode == "clean":
        exit()

    run_tgtd()
    start_log()
    time.sleep(1)
    set_target(ids)
    set_initiator(ids)

    disk_ids = obtain_disk_id(ids)

    if WITH_VM:
        start_vms(vms)
        attach_device(disk_ids, vms)

    start_replay()

    workloads = ['ml_prep', 'terasort', 'pagerank']
    modes = ['static','harvest','unsold']

    for workload in workloads:
        for mode in modes:
            run_ocssd(workload=workload, mode=mode)
            run_predictor()
            run_workloads(WITH_VM, vms, workload=workload, mode=mode)
            while not os.path.exists(END_FLAG):
                time.sleep(10)
            stop_workloads(WITH_VM, vms)
            stop_predictor()

            stop_ocssd()
            clear_queues()
            reset_flags()
