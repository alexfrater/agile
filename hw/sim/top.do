onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -group top /top_wrapper/top_i/sys_clk
add wave -noupdate -group top /top_wrapper/top_i/sys_rst
add wave -noupdate -group top /top_wrapper/top_i/regbank_clk
add wave -noupdate -group top /top_wrapper/top_i/regbank_resetn
add wave -noupdate -group top /top_wrapper/top_i/host_axil_awvalid
add wave -noupdate -group top /top_wrapper/top_i/host_axil_awready
add wave -noupdate -group top /top_wrapper/top_i/host_axil_awaddr
add wave -noupdate -group top /top_wrapper/top_i/host_axil_awprot
add wave -noupdate -group top /top_wrapper/top_i/host_axil_wvalid
add wave -noupdate -group top /top_wrapper/top_i/host_axil_wready
add wave -noupdate -group top /top_wrapper/top_i/host_axil_wdata
add wave -noupdate -group top /top_wrapper/top_i/host_axil_wstrb
add wave -noupdate -group top /top_wrapper/top_i/host_axil_bvalid
add wave -noupdate -group top /top_wrapper/top_i/host_axil_bready
add wave -noupdate -group top /top_wrapper/top_i/host_axil_bresp
add wave -noupdate -group top /top_wrapper/top_i/host_axil_arvalid
add wave -noupdate -group top /top_wrapper/top_i/host_axil_arready
add wave -noupdate -group top /top_wrapper/top_i/host_axil_araddr
add wave -noupdate -group top /top_wrapper/top_i/host_axil_arprot
add wave -noupdate -group top /top_wrapper/top_i/host_axil_rvalid
add wave -noupdate -group top /top_wrapper/top_i/host_axil_rready
add wave -noupdate -group top /top_wrapper/top_i/host_axil_rdata
add wave -noupdate -group top /top_wrapper/top_i/host_axil_rresp
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_awid
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_awaddr
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_awlen
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_awsize
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_awburst
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_awlock
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_awcache
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_awprot
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_awqos
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_awvalid
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_awready
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_wdata
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_wstrb
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_wlast
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_wvalid
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_wready
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_bid
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_bresp
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_bvalid
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_bready
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_arid
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_araddr
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_arlen
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_arsize
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_arburst
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_arlock
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_arcache
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_arprot
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_arqos
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_arvalid
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_arready
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_rid
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_rdata
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_rresp
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_rlast
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_rvalid
add wave -noupdate -group top /top_wrapper/top_i/ample_axi_rready
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_awid
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_awaddr
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_awlen
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_awsize
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_awburst
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_awlock
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_awcache
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_awprot
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_awqos
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_awvalid
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_awready
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_wdata
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_wstrb
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_wlast
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_wvalid
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_wready
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_bid
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_bresp
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_bvalid
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_bready
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_arid
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_araddr
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_arlen
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_arsize
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_arburst
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_arlock
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_arcache
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_arprot
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_arqos
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_arvalid
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_arready
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_rid
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_rdata
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_rresp
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_rlast
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_rvalid
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_fetch_axi_rready
add wave -noupdate -group top /top_wrapper/top_i/axil_interconnect_m_axi_awaddr
add wave -noupdate -group top /top_wrapper/top_i/axil_interconnect_m_axi_awprot
add wave -noupdate -group top /top_wrapper/top_i/axil_interconnect_m_axi_awvalid
add wave -noupdate -group top /top_wrapper/top_i/axil_interconnect_m_axi_awready
add wave -noupdate -group top /top_wrapper/top_i/axil_interconnect_m_axi_wdata
add wave -noupdate -group top /top_wrapper/top_i/axil_interconnect_m_axi_wstrb
add wave -noupdate -group top /top_wrapper/top_i/axil_interconnect_m_axi_wvalid
add wave -noupdate -group top /top_wrapper/top_i/axil_interconnect_m_axi_wready
add wave -noupdate -group top /top_wrapper/top_i/axil_interconnect_m_axi_bresp
add wave -noupdate -group top /top_wrapper/top_i/axil_interconnect_m_axi_bvalid
add wave -noupdate -group top /top_wrapper/top_i/axil_interconnect_m_axi_bready
add wave -noupdate -group top /top_wrapper/top_i/axil_interconnect_m_axi_araddr
add wave -noupdate -group top /top_wrapper/top_i/axil_interconnect_m_axi_arprot
add wave -noupdate -group top /top_wrapper/top_i/axil_interconnect_m_axi_arvalid
add wave -noupdate -group top /top_wrapper/top_i/axil_interconnect_m_axi_arready
add wave -noupdate -group top /top_wrapper/top_i/axil_interconnect_m_axi_rdata
add wave -noupdate -group top /top_wrapper/top_i/axil_interconnect_m_axi_rresp
add wave -noupdate -group top /top_wrapper/top_i/axil_interconnect_m_axi_rvalid
add wave -noupdate -group top /top_wrapper/top_i/axil_interconnect_m_axi_rready
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_araddr
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_arburst
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_arcache
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_arid
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_arlen
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_arlock
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_arprot
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_arqos
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_arsize
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_arvalid
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_arready
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_awaddr
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_awburst
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_awcache
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_awid
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_awlen
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_awlock
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_awprot
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_awqos
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_awready
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_awsize
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_awvalid
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_bid
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_bready
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_bresp
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_bvalid
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_rdata
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_rid
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_rlast
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_rready
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_rresp
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_rvalid
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_wdata
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_wlast
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_wready
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_wstrb
add wave -noupdate -group top /top_wrapper/top_i/weight_bank_axi_wvalid
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_araddr
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_arburst
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_arcache
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_arid
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_arlen
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_arlock
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_arprot
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_arqos
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_arsize
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_arvalid
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_arready
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_awaddr
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_awburst
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_awcache
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_awid
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_awlen
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_awlock
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_awprot
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_awqos
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_awready
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_awsize
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_awvalid
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_bid
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_bready
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_bresp
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_bvalid
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_rdata
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_rid
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_rlast
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_rready
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_rresp
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_rvalid
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_wdata
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_wlast
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_wready
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_wstrb
add wave -noupdate -group top /top_wrapper/top_i/read_master_axi_wvalid
add wave -noupdate -group top /top_wrapper/top_i/nsb_age_req_valid
add wave -noupdate -group top /top_wrapper/top_i/nsb_age_req_ready
add wave -noupdate -group top /top_wrapper/top_i/nsb_age_req
add wave -noupdate -group top /top_wrapper/top_i/nsb_age_resp_valid
add wave -noupdate -group top /top_wrapper/top_i/nsb_age_resp
add wave -noupdate -group top /top_wrapper/top_i/nsb_fte_req_valid
add wave -noupdate -group top /top_wrapper/top_i/nsb_fte_req_ready
add wave -noupdate -group top /top_wrapper/top_i/nsb_fte_req
add wave -noupdate -group top /top_wrapper/top_i/nsb_fte_resp_valid
add wave -noupdate -group top /top_wrapper/top_i/nsb_fte_resp
add wave -noupdate -group top /top_wrapper/top_i/nsb_prefetcher_req_valid
add wave -noupdate -group top /top_wrapper/top_i/nsb_prefetcher_req_ready
add wave -noupdate -group top /top_wrapper/top_i/nsb_prefetcher_req
add wave -noupdate -group top /top_wrapper/top_i/nsb_prefetcher_resp_valid
add wave -noupdate -group top /top_wrapper/top_i/nsb_prefetcher_resp
add wave -noupdate -group top /top_wrapper/top_i/message_channel_req_valid
add wave -noupdate -group top /top_wrapper/top_i/message_channel_req_ready
add wave -noupdate -group top /top_wrapper/top_i/message_channel_req
add wave -noupdate -group top /top_wrapper/top_i/message_channel_resp_valid
add wave -noupdate -group top /top_wrapper/top_i/message_channel_resp_ready
add wave -noupdate -group top /top_wrapper/top_i/weight_channel_req_valid
add wave -noupdate -group top /top_wrapper/top_i/weight_channel_req_ready
add wave -noupdate -group top /top_wrapper/top_i/weight_channel_req
add wave -noupdate -group top /top_wrapper/top_i/weight_channel_resp_valid
add wave -noupdate -group top /top_wrapper/top_i/weight_channel_resp_ready
add wave -noupdate -group top /top_wrapper/top_i/aggregation_buffer_set_node_id_valid
add wave -noupdate -group top /top_wrapper/top_i/aggregation_buffer_set_node_id
add wave -noupdate -group top /top_wrapper/top_i/aggregation_buffer_write_enable
add wave -noupdate -group top /top_wrapper/top_i/aggregation_buffer_write_address
add wave -noupdate -group top /top_wrapper/top_i/aggregation_buffer_write_count
add wave -noupdate -group top /top_wrapper/top_i/aggregation_buffer_feature_count
add wave -noupdate -group top /top_wrapper/top_i/aggregation_buffer_node_id
add wave -noupdate -group top /top_wrapper/top_i/aggregation_buffer_pop
add wave -noupdate -group top /top_wrapper/top_i/aggregation_buffer_out_feature
add wave -noupdate -group top /top_wrapper/top_i/aggregation_buffer_out_feature_valid
add wave -noupdate -group top /top_wrapper/top_i/aggregation_buffer_slot_free
add wave -noupdate -group top /top_wrapper/top_i/transformation_buffer_write_enable
add wave -noupdate -group top /top_wrapper/top_i/transformation_buffer_write_address
add wave -noupdate -group top /top_wrapper/top_i/transformation_buffer_feature_count
add wave -noupdate -group top /top_wrapper/top_i/transformation_buffer_pop
add wave -noupdate -group top /top_wrapper/top_i/transformation_buffer_out_feature
add wave -noupdate -group top /top_wrapper/top_i/transformation_buffer_slot_free
add wave -noupdate -group top /top_wrapper/top_i/scale_factor_queue_pop
add wave -noupdate -group top /top_wrapper/top_i/scale_factor_queue_out_data
add wave -noupdate -group top /top_wrapper/top_i/scale_factor_queue_out_valid
add wave -noupdate -group top /top_wrapper/top_i/scale_factor_queue_count
add wave -noupdate -group top /top_wrapper/top_i/scale_factor_queue_empty
add wave -noupdate -group top /top_wrapper/top_i/scale_factor_queue_full
add wave -noupdate -group top /top_wrapper/top_i/nsb_nodeslot_neighbour_count_count_hw
add wave -noupdate -group top /top_wrapper/top_i/nsb_nodeslot_node_id_id_hw
add wave -noupdate -group top /top_wrapper/top_i/nsb_nodeslot_precision_precision_hw
add wave -noupdate -group top /top_wrapper/top_i/nsb_nodeslot_config_make_valid_value_hw
add wave -noupdate -group top /top_wrapper/top_i/nsb_nodeslot_neighbour_count_strobe_hw
add wave -noupdate -group top /top_wrapper/top_i/nsb_nodeslot_node_id_strobe_hw
add wave -noupdate -group top /top_wrapper/top_i/nsb_nodeslot_precision_strobe_hw
add wave -noupdate -group top /top_wrapper/top_i/nsb_nodeslot_config_make_valid_strobe_hw
add wave -noupdate -group top /top_wrapper/top_i/graph_config_node_count_value
add wave -noupdate -group top /top_wrapper/top_i/ctrl_start_nodeslot_fetch_value
add wave -noupdate -group top /top_wrapper/top_i/ctrl_start_nodeslot_fetch_start_addr_value
add wave -noupdate -group top /top_wrapper/top_i/ctrl_start_nodeslot_fetch_done_value
add wave -noupdate -group top /top_wrapper/top_i/ctrl_start_nodeslot_fetch_done_ack_value
add wave -noupdate -group top /top_wrapper/top_i/nodeslot_finished
add wave -noupdate -group top /top_wrapper/top_i/status_nodeslots_empty_mask_0_value
add wave -noupdate -group top /top_wrapper/top_i/status_nodeslots_empty_mask_1_value
add wave -noupdate -group top /top_wrapper/top_i/status_nodeslots_empty_mask_2_value
add wave -noupdate -group top /top_wrapper/top_i/status_nodeslots_empty_mask_3_value
add wave -noupdate -group top /top_wrapper/top_i/status_nodeslots_empty_mask_4_value
add wave -noupdate -group top /top_wrapper/top_i/status_nodeslots_empty_mask_5_value
add wave -noupdate -group top /top_wrapper/top_i/status_nodeslots_empty_mask_6_value
add wave -noupdate -group top /top_wrapper/top_i/status_nodeslots_empty_mask_7_value
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_araddr
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_arburst
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_arcache
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_arid
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_arlen
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_arlock
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_arprot
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_arqos
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_arsize
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_arvalid
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_arready
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_awaddr
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_awburst
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_awcache
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_awid
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_awlen
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_awlock
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_awprot
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_awqos
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_awready
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_awsize
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_awvalid
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_bid
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_bready
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_bresp
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_bvalid
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_rdata
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_rid
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_rlast
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_rready
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_rresp
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_rvalid
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_wdata
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_wlast
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_wready
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_wstrb
add wave -noupdate -group top /top_wrapper/top_i/transformation_engine_axi_wvalid
add wave -noupdate -group top /top_wrapper/top_i/clk
add wave -noupdate -group top /top_wrapper/top_i/rst
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/clk
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/rst
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_awid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_awaddr
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_awlen
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_awsize
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_awburst
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_awlock
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_awcache
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_awprot
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_awqos
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_awuser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_awvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_awready
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_wdata
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_wstrb
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_wlast
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_wuser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_wvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_wready
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_bid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_bresp
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_buser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_bvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_bready
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_arid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_araddr
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_arlen
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_arsize
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_arburst
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_arlock
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_arcache
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_arprot
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_arqos
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_aruser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_arvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_arready
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_rid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_rdata
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_rresp
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_rlast
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_ruser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_rvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s00_axi_rready
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_awid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_awaddr
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_awlen
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_awsize
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_awburst
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_awlock
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_awcache
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_awprot
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_awqos
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_awuser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_awvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_awready
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_wdata
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_wstrb
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_wlast
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_wuser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_wvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_wready
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_bid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_bresp
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_buser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_bvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_bready
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_arid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_araddr
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_arlen
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_arsize
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_arburst
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_arlock
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_arcache
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_arprot
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_arqos
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_aruser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_arvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_arready
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_rid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_rdata
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_rresp
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_rlast
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_ruser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_rvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s01_axi_rready
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_awid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_awaddr
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_awlen
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_awsize
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_awburst
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_awlock
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_awcache
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_awprot
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_awqos
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_awuser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_awvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_awready
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_wdata
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_wstrb
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_wlast
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_wuser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_wvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_wready
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_bid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_bresp
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_buser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_bvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_bready
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_arid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_araddr
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_arlen
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_arsize
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_arburst
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_arlock
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_arcache
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_arprot
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_arqos
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_aruser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_arvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_arready
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_rid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_rdata
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_rresp
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_rlast
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_ruser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_rvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/s02_axi_rready
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_awid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_awaddr
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_awlen
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_awsize
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_awburst
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_awlock
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_awcache
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_awprot
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_awqos
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_awregion
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_awuser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_awvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_awready
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_wdata
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_wstrb
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_wlast
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_wuser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_wvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_wready
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_bid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_bresp
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_buser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_bvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_bready
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_arid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_araddr
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_arlen
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_arsize
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_arburst
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_arlock
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_arcache
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_arprot
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_arqos
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_arregion
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_aruser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_arvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_arready
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_rid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_rdata
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_rresp
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_rlast
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_ruser
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_rvalid
add wave -noupdate -group {memory interconnect} /top_wrapper/top_i/memory_interconnect/m00_axi_rready
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {0 ps} 0}
quietly wave cursor active 0
configure wave -namecolwidth 150
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 1
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -gridoffset 0
configure wave -gridperiod 1
configure wave -griddelta 40
configure wave -timeline 0
configure wave -timelineunits ps
update
WaveRestoreZoom {0 ps} {1 ns}
bookmark add wave bookmark8 {{17959609 ps} {22111537 ps}} 0
bookmark add wave bookmark9 {{18551117 ps} {18990769 ps}} 0
