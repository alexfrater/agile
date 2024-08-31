onerror {resume}
quietly virtual function -install /top_wrapper/top_i/transformation_engine_i -env /top_wrapper { &{/top_wrapper/top_i/transformation_engine_i/axi_write_master_data[31], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[30], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[29], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[28], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[27], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[26], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[25], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[24], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[23], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[22], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[21], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[20], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[19], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[18], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[17], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[16], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[15], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[14], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[13], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[12], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[11], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[10], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[9], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[8], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[7], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[6], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[5], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[4], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[3], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[2], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[1], /top_wrapper/top_i/transformation_engine_i/axi_write_master_data[0] }} sys1
quietly WaveActivateNextPane {} 0
add wave -noupdate -expand -group FTE -expand -group Sys1 -radix float32 /top_wrapper/top_i/transformation_engine_i/sys1
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/core_clk
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/resetn
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/regbank_clk
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/regbank_resetn
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_awaddr
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_awprot
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_awvalid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_awready
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_wdata
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_wstrb
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_wvalid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_wready
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_araddr
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_arprot
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_arvalid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_arready
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_rdata
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_rresp
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_rvalid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_rready
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_bresp
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_bvalid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_bready
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/nsb_fte_req_valid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/nsb_fte_req_ready
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/nsb_fte_req
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/nsb_fte_resp_valid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/nsb_fte_resp
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/aggregation_buffer_node_id
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/aggregation_buffer_pop
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/aggregation_buffer_out_feature_valid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/aggregation_buffer_out_feature
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/aggregation_buffer_slot_free
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/weight_channel_req_valid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/weight_channel_req_ready
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/weight_channel_req
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/weight_channel_resp_valid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/weight_channel_resp_ready
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_buffer_write_enable
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_buffer_write_address
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_buffer_slot_free
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_araddr
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arburst
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arcache
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arlen
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arlock
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arprot
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arqos
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arsize
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arvalid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arready
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_rdata
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_rid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_rlast
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_rready
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_rresp
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_rvalid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awaddr
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awburst
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awcache
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awlen
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awlock
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awprot
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awqos
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awready
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awsize
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awvalid
add wave -noupdate -expand -group FTE -radix float32 /top_wrapper/top_i/transformation_engine_i/sys1
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_wdata
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_wlast
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_wready
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_wstrb
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_wvalid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_bid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_bready
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_bresp
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_bvalid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_in_features_strobe
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_in_features_count
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_out_features_strobe
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_out_features_count
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_activation_function_strobe
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_activation_function_value
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_bias_strobe
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_bias_value
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_leaky_relu_alpha_strobe
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_leaky_relu_alpha_value
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_out_features_address_msb_strobe
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_out_features_address_lsb_strobe
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_out_features_address_msb_value
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_out_features_address_lsb_value
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/ctrl_buffering_enable_strobe
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/ctrl_buffering_enable_value
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/ctrl_writeback_enable_strobe
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/ctrl_writeback_enable_value
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/axi_write_master_req_valid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/axi_write_master_req_ready
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/axi_write_master_req_start_address
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/axi_write_master_req_len
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/axi_write_master_pop
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/axi_write_master_data_valid
add wave -noupdate -expand -group FTE -expand /top_wrapper/top_i/transformation_engine_i/axi_write_master_data
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/axi_write_master_resp_valid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/axi_write_master_resp_ready
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_req_ready
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_resp_valid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_resp_ready
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_resp
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_req_valid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_req_ready
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_req_start_address
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_req_len
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_pop
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_data_valid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_data
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_data_unreversed
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_resp_valid
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_resp_ready
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_resp_valid_bin
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_write_master_alloc_bin
add wave -noupdate -expand -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_write_master_alloc_bin_q
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/core_clk}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/resetn}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/nsb_fte_req_valid}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/nsb_fte_req_ready}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/nsb_fte_req}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/nsb_fte_resp_valid}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/nsb_fte_resp_ready}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/nsb_fte_resp}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/transformation_core_aggregation_buffer_node_id}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/transformation_core_aggregation_buffer_pop}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/transformation_core_aggregation_buffer_out_feature_valid}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/transformation_core_aggregation_buffer_out_feature}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/transformation_core_aggregation_buffer_slot_free}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/weight_channel_req_valid}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/weight_channel_req_ready}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/weight_channel_req}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/weight_channel_resp_valid}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/weight_channel_resp_ready}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/transformation_buffer_write_enable}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/transformation_buffer_write_address}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/transformation_buffer_write_data}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/transformation_buffer_slot_free}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/axi_write_master_req_valid}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/axi_write_master_req_ready}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/axi_write_master_req_start_address}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/axi_write_master_req_len}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/axi_write_master_pop}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/axi_write_master_data_valid}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/axi_write_master_data}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/axi_write_master_data_unreversed}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/axi_write_master_resp_valid}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/axi_write_master_resp_ready}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/layer_config_in_features_count}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/layer_config_out_features_count}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/layer_config_out_features_address_msb_value}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/layer_config_out_features_address_lsb_value}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/layer_config_bias_value}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/layer_config_activation_function_value}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/layer_config_leaky_relu_alpha_value}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/ctrl_buffering_enable_value}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/ctrl_writeback_enable_value}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/fte_state}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/fte_state_n}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/last_weight_resp_received}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/valid_row}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/nsb_req_nodeslots_q}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/nodeslot_count}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/nodeslots_to_buffer}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/nodeslots_to_writeback}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/sys_module_forward_valid}
add wave -noupdate -subitemconfig {{/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/sys_module_forward[0]} {-childformat {{{[2]} -radix float32} {{[1]} -radix float32} {{[0]} -radix float32}} -expand} {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/sys_module_forward[0][2]} {-radix float32} {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/sys_module_forward[0][1]} {-radix float32} {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/sys_module_forward[0][0]} {-radix float32}} {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/sys_module_forward}
add wave -noupdate -expand {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/sys_module_down_in}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/sys_module_down_in_valid}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/sys_module_down_out_valid}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/sys_module_flush_done}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/sys_module_pe_acc_row0}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/shift_sys_module}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/bias_valid}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/activation_valid}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/begin_feature_dump}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/pulse_systolic_module}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/slot_pop_shift}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/busy_aggregation_slots_snapshot}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/pe_delay_counter}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/transformation_buffer_slot_arb_oh}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/sent_writeback_beats}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/writeback_required_beats}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/sys_module_node_id_snapshot}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/out_features_required_bytes}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/fast_pulse_counter}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/bias_applied}
add wave -noupdate {/top_wrapper/top_i/transformation_engine_i/genblk1[0]/feature_transformation_core_i/activation_applied}
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {26310000 ps} 0} {Trace {453050162 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 400
configure wave -valuecolwidth 145
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
WaveRestoreZoom {12145008 ps} {40474992 ps}
bookmark add wave bookmark6 {{18551117 ps} {18990769 ps}} 0
bookmark add wave bookmark7 {{17959609 ps} {22111537 ps}} 0
