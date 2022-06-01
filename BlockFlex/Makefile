SUBDIRS = blockflex ocssd
SUBDIRSCLEAN=$(addsuffix \ clean,$(SUBDIRS))

all: subdirs

subdirs: $(SUBDIRS)
$(SUBDIRS):
	make -C $@

clean: $(SUBDIRSCLEAN)
$(SUBDIRSCLEAN):
	make -C $@ 

.PHONY: subdirs $(SUBDIRS)
