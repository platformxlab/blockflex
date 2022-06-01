/********************************************************************
* FILE NAME: cmd.c
*
*
* PURPOSE: cmd and routing function define here.
*
* 
* NOTES:
*
* 
* DEVELOPMENT HISTORY: 
* 
* Date Author Release  Description Of Change 
* ---- ----- ---------------------------- 
*2014.6.27, dxu, initial coding.
*
****************************************************************/ 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include "bflex.h"
#include "usage.h"

struct cmd_func_stru g_cmd_func_list[] = 
{
    /*debug*/
    {"-help", usage},    
    {"-list", dump_command},
    {"-rdreg", parse_rdreg_cmdline},
    {"-wrreg", parse_wrreg_cmdline},
    {"-rdreg64", parse_rdreg64_cmdline},
    {"-wrreg64", parse_wrreg64_cmdline},
    {"-rddword", parse_rddword_cmdline},
    {"-wrdword", parse_wrdword_cmdline},
    {"-checkcq", parse_checkcq_cmdline},
    {"-checksq", parse_checksq_cmdline},
    {"-dumpbflex", parse_dumpbflex_cmdline},
    {"-dumppci", parse_dumppci_cmdline},

    /*admin*/
    {"-delsq", parse_delsq_cmdline},
    {"-crtsq", parse_crtsq_cmdline},
    {"-delcq", parse_delcq_cmdline},
    {"-crtcq", parse_crtcq_cmdline},
    {"-getlp", parse_getlp_cmdline},
    {"-idn",   parse_idn_cmdline},
    {"-abort", parse_abort_cmdline},
    {"-setft", parse_setft_cmdline},
    {"-getft", parse_getft_cmdline},
    {"-asyner", parse_asyner_cmdline},
    {"-fmtnvm", parse_fmtnvm_cmdline},
    {"-fwdown", parse_fwdown_cmdline},
    {"-fwactv", parse_fwactv_cmdline},     
    {"-rstns", parse_rstns_cmdline},

    /*PPA sync*/
    {"-wrppasync", parse_wrppa_sync_cmdline},    
    {"-rdppasync", parse_rdppa_sync_cmdline}, 
    {"-wrpparawsync", parse_wrpparaw_sync_cmdline},     
    {"-rdpparawsync", parse_rdpparaw_sync_cmdline},      
    {"-ersppasync", parse_ersppa_sync_cmdline},
    {"-badblock", parse_badblock_cmdline},
    {"-ersblock", parse_erase_raidblock_cmdline}, 
    {"-ersall", parse_erase_disk_cmdline},    
    
    {"-getmac", parse_get_mac_addr_cmdline},
    {"-setmac", parse_set_mac_addr_cmdline},    
    {"-rdregspa", parse_rdregspa_cmdline},
    {"-wrregspa", parse_wrregspa_cmdline},
    {"-rdddr32", parse_rddword_cmdline},
    {"-wrddr32", parse_wrdword_cmdline},
    {"-datacheck", parse_datacheck_cmdline},
    {"-addrcheck", parse_addrcheck_cmdline},    
    {"-memcheck", parse_memcheck_cmdline},
    {"-autotest", parse_autotest_cmdline},        

    {NULL, NULL},
};

const int cmd_func_list_size = sizeof(g_cmd_func_list)/sizeof(struct cmd_func_stru);

