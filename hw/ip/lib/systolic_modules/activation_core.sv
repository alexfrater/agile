import top_pkg::*;

module activation_core #(
    parameter PRECISION = top_pkg::FLOAT_32,
    parameter DATA_WIDTH = 32,
    parameter FLOAT_WIDTH = 32
) (
    input  logic core_clk,
    input  logic resetn,

    input  logic [$bits(ACTIVATION_FUNCTION_e)-1:0] sel_activation,
    
    input  logic                                    in_feature_valid,
    input  logic [DATA_WIDTH-1:0]                   in_feature,
    
    output logic                                    activated_feature_valid,
    output logic [DATA_WIDTH-1:0]                   activated_feature,

    input  logic [DATA_WIDTH-1:0]                   layer_config_leaky_relu_alpha_value
);

logic                  activated_feature_valid_comb;
logic [DATA_WIDTH-1:0] activated_feature_comb;

logic                  leaky_relu_activation_valid_comb;
logic [DATA_WIDTH-1:0] leaky_relu_activation_comb;

assign activated_feature_valid_comb = (sel_activation == top_pkg::NONE) ? in_feature_valid
                               : (sel_activation == top_pkg::RELU) ? in_feature_valid
                               : (sel_activation == top_pkg::LEAKY_RELU) ? leaky_relu_activation_valid_comb
                               : '0;

always_comb begin
    case (sel_activation)

    top_pkg::NONE: begin
        activated_feature_comb = in_feature;
    end

    top_pkg::RELU: begin
        activated_feature_comb = in_feature[FLOAT_WIDTH-1] ? '0 : in_feature;
    end

    top_pkg::LEAKY_RELU: begin
        activated_feature_comb = in_feature[FLOAT_WIDTH-1] ? leaky_relu_activation_comb : in_feature;
    end

    endcase
end

// Leaky ReLU
// -----------------------------------------------------------------

if (PRECISION == top_pkg::FLOAT_32) begin

`ifdef SIMULATION_QUICK
    assign leaky_relu_activation_valid_comb = in_feature_valid;
    assign leaky_relu_activation_comb = in_feature;
    
`else

    logic [DATA_WIDTH-1:0] in_feature_gated;
    assign in_feature_gated = in_feature_valid ? in_feature : '0;

    fp_mult activation_mult (
        .in1       (in_feature_gated),

        .in2       (layer_config_leaky_relu_alpha_value),

        .res  (leaky_relu_activation_comb)
        );
    assign leaky_relu_activation_valid_comb = in_feature_valid;

`endif

    end else begin

        // Fixed point
        always_comb begin
            leaky_relu_activation_valid_comb = in_feature_valid;
            leaky_relu_activation_comb = in_feature * layer_config_leaky_relu_alpha_value;
        end
    end

// Register activated feature
// -----------------------------------------------------------------

always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
        activated_feature_valid <= '0;
        activated_feature <= '0;

    end else begin
        activated_feature_valid <= activated_feature_valid_comb;
        activated_feature       <= activated_feature_comb;
    end

end

endmodule