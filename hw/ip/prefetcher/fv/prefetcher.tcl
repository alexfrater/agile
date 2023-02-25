clear -all
analyze -clear

analyze -sv ../rtl/prefetcher.sv
analyze -sv prefetcher_fv_intf.sv
analyze -sv ../rtl/prefetcher_fetch_tag.sv
analyze -sv prefetcher_fetch_tagfv_intf.sv

elaborate -bbox_mul 64 -top axi_read_master

# Setup global clocks and resets
clock core_clk
reset -expression !(resetn)

# Setup task
task -set <embedded>
set_proofgrid_max_jobs 4
set_proofgrid_max_local_jobs 4

# Assumptions
# =====================================

# Launch proof
# prove -all