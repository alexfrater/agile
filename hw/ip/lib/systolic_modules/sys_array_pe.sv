//////////////////////////////////////////////////////////////////////////////////
// Engineer: 
// 
// Design Name: 
// Create Date:
// Module Name: sys_array_pe
// Tool Versions: 
// Description: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module sys_array_pe #(
    parameter FLOAT_WIDTH = 32
) (
    input  logic                            core_clk,
    input  logic                            rstn,

    input  logic [FLOAT_WIDTH-1:0]          pe_forward_in,
    input  logic                            pe_forward_in_valid,
    input  logic [FLOAT_WIDTH-1:0]          pe_down_in,
    input  logic                            pe_down_in_valid,
    
    output logic [FLOAT_WIDTH-1:0]          pe_forward_out,
    output logic                            pe_forward_out_valid,
    output logic [FLOAT_WIDTH-1:0]          pe_down_out,
    output logic                            pe_down_out_valid,

    input  logic                            overwrite,
    input  logic [FLOAT_WIDTH-1:0]          overwrite_data,
    
    output logic [FLOAT_WIDTH-1:0]          pe_acc
);

logic update;

mac #(
    .FLOAT_WIDTH(FLOAT_WIDTH)
) mac_i (
    .core_clk,            
    .rstn,
    
    .en                 (update),
    .a                  (pe_forward_in),
    .b                  (pe_down_in),

    .overwrite          (overwrite),
    .overwrite_data     (overwrite_data),
    
    .acc                (pe_acc)
);

always_comb begin
    
    update = pe_forward_in_valid && pe_down_in_valid;

    pe_forward_out_valid = pe_forward_in_valid;
    pe_down_out_valid    = pe_down_in_valid;
end

always_ff @(posedge core_clk or negedge rstn) begin
    if (!rstn) begin
        pe_forward_out              <= '0;
        pe_down_out                 <= '0;
    end else begin
        // Register incoming messages when updating the accumulator
        pe_forward_out              <= update ? pe_forward_in : pe_forward_out;
        pe_down_out                 <= update ? pe_down_in : pe_down_out;
    end
end

// ======================================================================================================
// Assertions
// ======================================================================================================

// P_update_acc_both_valid: assert property (
//     @(posedge core_clk) disable iff (!rstn)
//     (!pe_forward_in_valid || !pe_down_in_valid) |=> pe_acc == $past(pe_acc, 1)
// );

endmodule