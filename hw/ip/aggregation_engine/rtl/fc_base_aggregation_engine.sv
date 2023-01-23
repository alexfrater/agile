
// import aggregation_engine_regbank_regs_pkg::*;

module fc_base_aggregation_engine #(
    parameter AXI_ADDR_WIDTH = 32
) (
    input logic core_clk,
    input logic resetn,
    
    // Regbank Slave AXI interface
    input  logic [AXI_ADDR_WIDTH-1:0]                           s_axi_awaddr,
    input  logic [2:0]                                          s_axi_awprot,
    input  logic                                                s_axi_awvalid,
    output logic                                                s_axi_awready,
    input  logic [31:0]                                         s_axi_wdata,
    input  logic [3:0]                                          s_axi_wstrb,
    input  logic                                                s_axi_wvalid,
    output logic                                                s_axi_wready,
    input  logic [AXI_ADDR_WIDTH-1:0]                           s_axi_araddr,
    input  logic [2:0]                                          s_axi_arprot,
    input  logic                                                s_axi_arvalid,
    output logic                                                s_axi_arready,
    output logic [31:0]                                         s_axi_rdata,
    output logic [1:0]                                          s_axi_rresp,
    output logic                                                s_axi_rvalid,
    input  logic                                                s_axi_rready,
    output logic [1:0]                                          s_axi_bresp,
    output logic                                                s_axi_bvalid,
    input  logic                                                s_axi_bready,

    input logic c0_init_calib_complete
);


// ==================================================================================================================================================
// Declarations
// ==================================================================================================================================================

// Regbank
// ------------------------------------------------------------

logic                         matrix_a_address_strobe;
logic                         matrix_b_address_strobe;
logic                         matrix_c_address_strobe;

logic [31:0]                  matrix_a_address_value;
logic [31:0]                  matrix_b_address_value;
logic [31:0]                  matrix_c_address_value;

logic [8:0]                   matrix_a_strobe;
logic [8:0]                   matrix_b_strobe;

logic [8:0] [31:0]            matrix_a_value;
logic [8:0] [31:0]            matrix_b_value;

logic                         config_strobe;

logic [0:0]                   config_matrix_a_valid;
logic [0:0]                   config_matrix_b_valid;
logic [0:0]                   config_start_mult;
logic [0:0]                   config_mult_done;

// ==================================================================================================================================================
// Instances
// ==================================================================================================================================================

// Register Bank
// ----------------------------------------------------

aggregation_engine_regbank_regs #(
    .AXI_ADDR_WIDTH(32),
    .BASEADDR(32'b0) // use regbank parameter
) aggregation_engine_regbank_regs_i (
    .axi_aclk                       (core_clk),
    .axi_aresetn                    (resetn),
    .s_axi_awaddr,
    .s_axi_awprot,
    .s_axi_awvalid,
    .s_axi_awready,
    .s_axi_wdata,
    .s_axi_wstrb,
    .s_axi_wvalid,
    .s_axi_wready,
    .s_axi_araddr,
    .s_axi_arprot,
    .s_axi_arvalid,
    .s_axi_arready,
    .s_axi_rdata,
    .s_axi_rresp,
    .s_axi_rvalid,
    .s_axi_rready,
    .s_axi_bresp,
    .s_axi_bvalid,
    .s_axi_bready,
    .matrix_a_address_strobe,
    .matrix_a_address_value,
    .matrix_b_address_strobe,
    .matrix_b_address_value,
    .matrix_c_address_strobe,
    .matrix_c_address_value,
    .matrix_a_strobe,
    .matrix_a_value,
    .matrix_b_strobe,
    .matrix_b_value,
    .config_strobe,
    .config_matrix_a_valid,
    .config_matrix_b_valid,
    .config_start_mult,
    .config_mult_done 
);

// ==================================================================================================================================================
// Logic
// ==================================================================================================================================================


// ==================================================================================================================================================
// Assertions
// ==================================================================================================================================================

endmodule