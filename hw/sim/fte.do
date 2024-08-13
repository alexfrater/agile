onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/core_clk
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/resetn
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/regbank_clk
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/regbank_resetn
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_awaddr
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_awprot
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_awvalid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_awready
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_wdata
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_wstrb
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_wvalid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_wready
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_araddr
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_arprot
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_arvalid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_arready
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_rdata
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_rresp
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_rvalid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_rready
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_bresp
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_bvalid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/s_axi_bready
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/nsb_fte_req_valid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/nsb_fte_req_ready
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/nsb_fte_req
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/nsb_fte_resp_valid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/nsb_fte_resp
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/aggregation_buffer_node_id
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/aggregation_buffer_pop
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/aggregation_buffer_out_feature_valid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/aggregation_buffer_out_feature
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/aggregation_buffer_slot_free
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/weight_channel_req_valid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/weight_channel_req_ready
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/weight_channel_req
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/weight_channel_resp_valid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/weight_channel_resp_ready
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_buffer_write_enable
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_buffer_write_address
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_buffer_slot_free
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_araddr
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arburst
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arcache
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arlen
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arlock
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arprot
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arqos
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arsize
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arvalid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_arready
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_rdata
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_rid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_rlast
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_rready
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_rresp
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_rvalid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awaddr
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awburst
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awcache
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awlen
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awlock
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awprot
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awqos
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awready
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awsize
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_awvalid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_wdata
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_wlast
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_wready
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_wstrb
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_wvalid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_bid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_bready
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_bresp
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_engine_axi_interconnect_axi_bvalid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_in_features_strobe
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_in_features_count
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_out_features_strobe
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_out_features_count
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_activation_function_strobe
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_activation_function_value
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_bias_strobe
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_bias_value
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_leaky_relu_alpha_strobe
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_leaky_relu_alpha_value
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_out_features_address_msb_strobe
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_out_features_address_lsb_strobe
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_out_features_address_msb_value
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/layer_config_out_features_address_lsb_value
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/ctrl_buffering_enable_strobe
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/ctrl_buffering_enable_value
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/ctrl_writeback_enable_strobe
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/ctrl_writeback_enable_value
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/axi_write_master_req_valid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/axi_write_master_req_ready
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/axi_write_master_req_start_address
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/axi_write_master_req_len
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/axi_write_master_pop
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/axi_write_master_data_valid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/axi_write_master_data
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/axi_write_master_resp_valid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/axi_write_master_resp_ready
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_req_ready
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_resp_valid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_resp_ready
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_resp
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_req_valid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_req_ready
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_req_start_address
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_req_len
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_pop
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_data_valid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_data
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_data_unreversed
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_resp_valid
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_axi_write_master_resp_ready
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_resp_valid_bin
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_write_master_alloc_bin
add wave -noupdate -group FTE /top_wrapper/top_i/transformation_engine_i/transformation_core_write_master_alloc_bin_q
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {68986696 ps} 0} {Trace {453050162 ps} 0}
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
WaveRestoreZoom {0 ps} {544083750 ps}
bookmark add wave bookmark6 {{18551117 ps} {18990769 ps}} 0
bookmark add wave bookmark7 {{17959609 ps} {22111537 ps}} 0
