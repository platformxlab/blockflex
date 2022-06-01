/***********************************************************************
* FILE NAME: usage.h
*
*
* PURPOSE: headerfile of usage.c. 
*
*
* ---- ----- ---------------------------- 
*2015.2.4, dxu, initial coding.
*
*************************************************************************/
#ifndef _UTEST_USAGE_H
#define _UTEST_USAGE_H

int rdreg_usage(int argc, char* argv[]);
int wrreg_usage(int argc, char* argv[]);
int checkcq_usage(int argc, char* argv[]);
int checksq_usage(int argc, char* argv[]);

int delsq_usage(int argc, char* argv[]);
int crtsq_usage(int argc, char* argv[]);
int delcq_usage(int argc, char* argv[]);
int crtcq_usage(int argc, char* argv[]);
int getlp_usage(int argc, char* argv[]);
int idn_usage(int argc, char* argv[]);
int abort_usage(int argc, char* argv[]);
int setft_usage(int argc, char* argv[]);
int getft_usage(int argc, char* argv[]);
int asyner_usage(int argc, char* argv[]);
int fmtnvm_usage(int argc, char* argv[]);
int fwdown_usage(int argc, char* argv[]);
int fwactv_usage(int argc, char* argv[]);

int wrregspa_usage(int argc, char* argv[]);
int rdregspa_usage(int argc, char* argv[]);
int wrppasync_usage(int argc, char* argv[]);
int rdppasync_usage(int argc, char* argv[]);
int wrpparawsync_usage(int argc, char* argv[]);
int rdpparawsync_usage(int argc, char* argv[]);
int ersppasync_usage(int argc, char* argv[]);
int ersall_usage(int argc, char* argv[]);
int badblock_usage(int argc, char* argv[]);

#endif
