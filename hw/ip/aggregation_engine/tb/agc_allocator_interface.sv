import noc_pkg::*;

interface agc_allocator_interface (
    input  logic                                           core_clk,
    input  logic                                           resetn,

    // Allocation request
    input  logic                                           allocation_req_valid,
    input  logic                                           allocation_req_ready,
    input  top_pkg::NSB_AGE_REQ_t                          allocation_req,

    input  logic [noc_pkg::MAX_AGC_COUNT-1:0]                           cores_free,
    input  logic [9:0]                                     layer_config_in_features_count,
    
    // Deallocation request
    input  logic                                           deallocation_valid,
    input  logic [noc_pkg::MAX_AGC_COUNT-1:0]                           deallocation_cores,

    // Generated AGM request
    input  logic [noc_pkg::MAX_AGGREGATION_COLS-1:0]                        agm_req_valid,
    input  logic [noc_pkg::MAX_AGGREGATION_COLS-1:0]                        agm_req_ready,
    input  age_pkg::AGE_AGM_REQ_t                          agm_req

);

endinterface