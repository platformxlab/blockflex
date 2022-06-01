/***********************************************************************
* FILE NAME: usage.c
*
*
* PURPOSE: display user guide message of utest for user
*
*
* ---- ----- ---------------------------- ---------
*2015.2.4, dxu, initial coding.
*
*************************************************************************/
#include <stdio.h>
#include "bflex.h"
#include "usage.h"

struct cmd_func_stru cmd_usage_guide[] = 
{
	/* debug cmd */
    {"rdreg", rdreg_usage},		
    {"wrreg", wrreg_usage},    
	{"checkcq", checkcq_usage},
    {"checksq", checksq_usage},

    /* admin cmd */
    {"delsq", delsq_usage},
    {"crtsq", crtsq_usage},
    {"delcq", delcq_usage},
    {"crtcq", crtcq_usage},
    {"getlp", getlp_usage},
    {"idn",   idn_usage},
    {"abort", abort_usage},
    {"setft", setft_usage},
    {"getft", getft_usage},
    {"asyner", asyner_usage},
    {"fmtnvm", fmtnvm_usage},
    {"fwdown", fwdown_usage},
    {"fwactv", fwactv_usage},  

	/* IO cmd */
	{"rdregspa", rdregspa_usage},
    {"wrregspa", wrregspa_usage},
    {"wrppasync", wrppasync_usage},		
    {"rdppasync", rdppasync_usage},    
    {"ersppasync", ersppasync_usage},    
    {"wrpparawsync", wrpparawsync_usage},     
    {"rdpparawsync", rdpparawsync_usage},

    {"ersall", ersall_usage},    
    {"badblock", badblock_usage},    
    {NULL, NULL},
};

int cmd_count = sizeof(cmd_usage_guide)/sizeof(struct cmd_func_stru);

/* cmd detail usage introduce should add later */

int rdreg_usage(int argc, char* argv[])
{
	//parse_rdreg_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-r    register bar offset");
	DISPLAY("--------------------------");

	return 0;
}

int wrreg_usage(int argc, char* argv[])
{
	//parse_wrreg_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-r    register bar offset");	
	DISPLAY("-v    WriteVal");
	DISPLAY("--------------------------");

	return 0;
}

int checksq_usage(int argc, char* argv[])
{
	//parse_checksq_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-q    qid");
	DISPLAY("--------------------------");

	return 0;
}

int checkcq_usage(int argc, char* argv[])
{
	//parse_checkcq_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-q    qid");
	DISPLAY("--------------------------");

	return 0;
}

int crtcq_usage(int argc, char* argv[])
{
	//parse_crtcq_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-q    qid");	
	DISPLAY("-v    int vector   default equal to cqid");
	DISPLAY("-s    qsize");    
	DISPLAY("-w    sq_where  0:Host    1:DDR");
	DISPLAY("-c    cq_where  0:Host    1:DDR");
	DISPLAY("--------------------------");

	return 0;
}

int crtsq_usage(int argc, char* argv[])
{
	//parse_crtsq_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-q    qid");	
	DISPLAY("-p    queue priority");	
	DISPLAY("PS:   sqsize=cqsize");
	DISPLAY("--------------------------");

	return 0;
}

int delsq_usage(int argc, char* argv[])
{
	//parse_delsq_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-q    qid");
	DISPLAY("--------------------------");

	return 0;
}

int delcq_usage(int argc, char* argv[])
{
	//parse_delcq_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-q    qid");
	DISPLAY("--------------------------");

	return 0;
}

int getlp_usage(int argc, char* argv[])
{
	//parse_getlp_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-l    log id");	
	DISPLAY("-n    number dword");	
	DISPLAY("-i:   nsid");
	DISPLAY("-o:   prp1 offset");
	DISPLAY("-e:   prp2 offset");	
	DISPLAY("-f:   file for returned log ");
	DISPLAY("--------------------------");

	return 0;
}

int idn_usage(int argc, char* argv[])
{
	//parse_idn_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-c    cns  control or namespace");	
	DISPLAY("-i:   nsid");
	DISPLAY("-o:   prp1 offset");
	DISPLAY("-e:   prp2 offset");	
	DISPLAY("-f:   file for returned information");
	DISPLAY("--------------------------");

	return 0;
}

int abort_usage(int argc, char* argv[])
{
	//parse_abort_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-q    qid");	
	DISPLAY("-c:   cmdid");
	DISPLAY("--------------------------");

	return 0;
}

int getft_usage(int argc, char* argv[])
{
	//parse_getft_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-u    feature id");	
	DISPLAY("-s:   sel");	
	DISPLAY("-i:   nsid");	
	DISPLAY("-o:   prp1 offset");
	DISPLAY("-e:   prp2 offset");	
	DISPLAY("-f:   file for returned imformation");
	DISPLAY("--------------------------");

	return 0;
}

/* fid=0x03  fid=0x0c  fid=0x81 (need file)*/

int setft_usage(int argc, char* argv[])
{
	//parse_setft_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-u    feature id");	
	DISPLAY("-s:   saved  if");	
	DISPLAY("-i:   nsid");	
	DISPLAY("-d:   dword11  	fid=0x03 fid=0x0c fid=0x81 (need file)");
	DISPLAY("-o:   prp1 offset");
	DISPLAY("-e:   prp2 offset");	
	DISPLAY("-f:   file to setft");
	DISPLAY("--------------------------");

	return 0;
}

int asyner_usage(int argc, char* argv[])
{
	//parse_asyner_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("--------------------------");

	return 0;
}

int fwdown_usage(int argc, char* argv[])
{
	//parse_fwdown_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-r    Firmware Image Dword offset start to download 0-base");	
	DISPLAY("-n:   number dword to download 0-base");	
	DISPLAY("-o:   prp1 offset");	
	DISPLAY("-f:   Firmware Image");
	DISPLAY("--------------------------");

	return 0;
}

int fwactv_usage(int argc, char* argv[])
{
	//parse_fwactv_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-s    Firmware Slot");	
	DISPLAY("-a:   Active Action");
	DISPLAY("--------------------------");

	return 0;
}

int fmtnvm_usage(int argc, char* argv[])
{
	//parse_fmtnvm_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-i    nsid");	
	DISPLAY("-v:   dword10  format value");
	DISPLAY("--------------------------");

	return 0;
}

int wrregspa_usage(int argc, char* argv[])
{
 	//parse_wrregspa_cmdline   a:n:q:i:f:k:
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-a    bar offset address  DDR start_addr:0x400000");
	DISPLAY("-n    number dword  1-base");
	DISPLAY("-q    qid");
	DISPLAY("-i    nsid");
	DISPLAY("-f    datafile");	
	DISPLAY("--------------------------");

	return 0;
}

int rdregspa_usage(int argc, char* argv[])
{
 	//parse_rdregspa_cmdline   a:n:q:i:f:k:
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-a    bar offset address  DDR start_addr:0x500000");
	DISPLAY("-n    number dword  1-base");
	DISPLAY("-q    qid");
	DISPLAY("-i    nsid");
	DISPLAY("-f    datafile");	
	DISPLAY("--------------------------");

	return 0;
}

int wrppasync_usage(int argc, char* argv[])
{
 	//parse_wrppa_sync_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-a    ppa_addr");
	DISPLAY("-n    nlb  1-base");
	DISPLAY("-q    qid");
	DISPLAY("-i    nsid");
	DISPLAY("-t    ctrl plane mode and Aes Key");
	DISPLAY("-f    datafile");	
	DISPLAY("-m    metadatafile");
	DISPLAY("--------------------------");

	return 0;
}

int rdppasync_usage(int argc, char* argv[])
{
	//parse_rdppa_sync_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-a    ppa_addr");
	DISPLAY("-n    nlb  1-base");
	DISPLAY("-q    qid");
	DISPLAY("-i    nsid");
	DISPLAY("-t    ctrl plane mode and Aes Key");
	DISPLAY("-f    datafile");	
	DISPLAY("-m    metadatafile");
	DISPLAY("--------------------------");

	return 0;
}


int wrpparawsync_usage(int argc, char* argv[])
{
 	//parse_wrpparaw_sync_cmdline a:n:f:m:i:q:c:k:t:
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-a    ppa_addr");
	DISPLAY("-n    nlb  1-base");
	DISPLAY("-q    qid");
	DISPLAY("-i    nsid");
	DISPLAY("-t    ctrl plane mode and Aes Key");
	DISPLAY("-f    datafile");	
	DISPLAY("-m    metadatafile");
	DISPLAY("--------------------------");

	return 0;
}

int rdpparawsync_usage(int argc, char* argv[])
{
	//parse_rdpparaw_sync_cmdline
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-a    ppa_addr");
	DISPLAY("-n    nlb  1-base");
	DISPLAY("-q    qid");
	DISPLAY("-i    nsid");
	DISPLAY("-t    ctrl plane mode and Aes Key");
	DISPLAY("-f    datafile");	
	DISPLAY("-m    metadatafile");
	DISPLAY("--------------------------");

	return 0;
}


int ersppasync_usage(int argc, char* argv[])
{
	//parse_ersppa_sync_cmdline 
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-a    ppa_addr");
	DISPLAY("-n    nlb  1-base");
	DISPLAY("-q    qid");
	DISPLAY("-i    nsid");
	DISPLAY("-t    ctrl plane mode and Aes Key");	
	DISPLAY("--------------------------");

	return 0;
}

int ersall_usage(int argc, char* argv[])
{
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");	
	DISPLAY("-i    nsid");
	DISPLAY("-q    qid");
	DISPLAY("-f    badbin file");
	DISPLAY("-m    chmask_sw");	
	DISPLAY("--------------------------");
	return 0;
}

int badblock_usage(int argc, char* argv[])
{
	DISPLAY("--------------------------");	
	DISPLAY("-k    devid");
	DISPLAY("-t    flash type 0:Toshiba   1:Micron   3:Skhynix");
	DISPLAY("-f    badbin comment file");
	DISPLAY("-m    badbin file");	
	DISPLAY("--------------------------");
	return 0;
}


int dump_command(int argc, char* argv[])
{
  //printf("Utest version: %s\nSupport flash type: %s\n", UTEST_VER, FLASH_TYPE);
  
  /* debug */
  DISPLAY("-rdreg");
  DISPLAY("-wrreg");
  DISPLAY("-rdreg64");
  DISPLAY("-wrreg64");
  DISPLAY("-checkcq");
  DISPLAY("-checksq");
  DISPLAY("-dumpbflex");
  DISPLAY("-dumppci");
  DISPLAY("-setdbsw");
  DISPLAY("-getdbsw");

  /* admin */
  DISPLAY("-delsq");
  DISPLAY("-crtsq");
  DISPLAY("-delcq");
  DISPLAY("-crtcq");
  DISPLAY("-getlp");
  DISPLAY("-idn");
  DISPLAY("-abort");
  DISPLAY("-setft");
  DISPLAY("-getft");
  DISPLAY("-asyner");
  DISPLAY("-fmtnvm");
  DISPLAY("-fwdown");
  DISPLAY("-fwactv");     

  /* PPA sync */
  DISPLAY("-wrppasync");    
  DISPLAY("-rdppasync"); 
  DISPLAY("-wrpparawsync");     
  DISPLAY("-rdpparawsync");      
  DISPLAY("-ersppasync");

  /* DDR related */
  DISPLAY("-rdregspa");
  DISPLAY("-wrregspa");
  DISPLAY("-rdddr32");
  DISPLAY("-wrddr32");
  DISPLAY("-datacheck");
  DISPLAY("-addrcheck");    
  DISPLAY("-memcheck");
  DISPLAY("-autotest");        
  DISPLAY("-badblock");

  /* Ktest use */
  DISPLAY("-ppapf");

  return 0;
}
