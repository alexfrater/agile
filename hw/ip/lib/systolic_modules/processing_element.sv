//////////////////////////////////////////////////////////////////////////////////
// Engineer: 
// 
// Design Name: 
// Create Date:
// Module Name: processing_element
// Tool Versions: 
// Description: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module processing_element #(
    parameter FLOAT_WIDTH = 32
) (
    input  logic                            core_clk,
    input  logic                            resetn,

    input  logic                            pulse_systolic_module,

    input  logic                            pe_forward_in_valid,
    input  logic [FLOAT_WIDTH-1:0]          pe_forward_in,

    input  logic                            pe_down_in_valid,
    input  logic [FLOAT_WIDTH-1:0]          pe_down_in,
    
    output logic                            pe_forward_out_valid,
    output logic [FLOAT_WIDTH-1:0]          pe_forward_out,
    
    output logic                            pe_down_out_valid,
    output logic [FLOAT_WIDTH-1:0]          pe_down_out,

    input  logic                            bias_valid,
    input  logic [FLOAT_WIDTH-1:0]          bias,

    input  logic                            activation_valid,
    input  logic [$bits(top_pkg::ACTIVATION_FUNCTION_e)-1:0] activation,

    input  logic                            shift_valid,
    input  logic [FLOAT_WIDTH-1:0]          shift_data,
    
    output logic [FLOAT_WIDTH-1:0]          pe_acc,

    input  logic [31:0]                     layer_config_leaky_relu_alpha_value
);

// ==================================================================================================================================================
// Declarations
// ==================================================================================================================================================

logic                   update_accumulator;

logic                   overwrite_accumulator;
logic [FLOAT_WIDTH-1:0] overwrite_data;

logic                   bias_out_valid;
logic [FLOAT_WIDTH-1:0] pe_acc_add_bias;

logic                   activated_feature_valid;
logic [FLOAT_WIDTH-1:0] activated_feature;

// ==================================================================================================================================================
// Accumulator
// ==================================================================================================================================================

mac #(
    .FLOAT_WIDTH        (FLOAT_WIDTH)
) mac_i (
    .core_clk,            
    .resetn,
    
    .en                 (update_accumulator),
    .a                  (pe_forward_in),
    .b                  (pe_down_in),

    .overwrite          (overwrite_accumulator),
    .overwrite_data     (overwrite_data),
    
    .acc                (pe_acc)
);

// Bias addition
// -------------------------------------------------------------

fp_add bias_adder (
  .s_axis_a_tvalid              ('1),
  .s_axis_a_tdata               (pe_acc),
  
  .s_axis_b_tvalid              (bias_valid),
  .s_axis_b_tdata               (bias),

  .m_axis_result_tvalid         (bias_out_valid),
  .m_axis_result_tdata          (pe_acc_add_bias)
);

// Activations
// -------------------------------------------------------------

activation_core activation_core_i (
    .core_clk         (core_clk),
    .resetn           (resetn),

    .sel_activation   (activation),

    .in_feature_valid (activation_valid),
    .in_feature       (pe_acc),

    .activated_feature_valid (activated_feature_valid),
    .activated_feature       (activated_feature),

    .layer_config_leaky_relu_alpha_value (layer_config_leaky_relu_alpha_value)
);

// ==================================================================================================================================================
// Logic
// ==================================================================================================================================================

assign update_accumulator = pulse_systolic_module && pe_forward_in_valid && pe_down_in_valid;

// Register incoming (forward/down) features
always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
        pe_forward_out_valid        <= '0;
        pe_forward_out              <= '0;
        
        pe_down_out_valid           <= '0;
        pe_down_out                 <= '0;

    end else if (update_accumulator) begin
        // Register incoming messages when updating the accumulator
        pe_forward_out_valid        <= pe_forward_in_valid;
        pe_forward_out              <= pe_forward_in;

        pe_down_out_valid           <= pe_down_in_valid;
        pe_down_out                 <= pe_down_in;
    end
end

// Overwrite accumulator for activation, bias and shifting
always_comb begin
    overwrite_accumulator = bias_out_valid || activated_feature_valid || shift_valid;

    overwrite_data        = bias_valid ? pe_acc_add_bias
                            : activated_feature_valid ? activated_feature
                            : shift_valid ? shift_data
                            : '0;
end

// ======================================================================================================
// Assertions
// ======================================================================================================

// P_update_acc_both_valid: assert property (
//     @(posedge core_clk) disable iff (!resetn)
//     (!pe_forward_in_valid || !pe_down_in_valid) |=> pe_acc == $past(pe_acc, 1)
// );

endmodule