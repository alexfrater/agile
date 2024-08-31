

module top_wrapper(

    input  wire         sys_clk,
    input  wire         sys_rst,

    input  wire         regbank_clk,
    input  wire         regbank_resetn,





    // AXI-L Register Programming Interface
    input  wire [31:0]  host_axil_awaddr,
    input  wire [2:0]   host_axil_awprot,
    input  wire [0:0]   host_axil_awvalid,
    output wire [0:0]   host_axil_awready,

    input  wire [31:0]  host_axil_wdata,
    input  wire [3:0]   host_axil_wstrb,
    input  wire [0:0]   host_axil_wvalid,
    output wire [0:0]   host_axil_wready,

    output wire [1:0]   host_axil_bresp,
    output wire [0:0]   host_axil_bvalid,
    input  wire [0:0]   host_axil_bready,

    input  wire [31:0]  host_axil_araddr,
    input  wire [2:0]   host_axil_arprot,
    input  wire [0:0]   host_axil_arvalid,
    output wire [0:0]   host_axil_arready,

    output wire [31:0]  host_axil_rdata,
    output wire [1:0]   host_axil_rresp,
    output wire [0:0]   host_axil_rvalid,
    input  wire [0:0]   host_axil_rready,


    // Ample AXI Master Interface
    output wire [3:0]   ample_axi_awid,
    output wire [33:0]  ample_axi_awaddr,
    output wire [7:0]   ample_axi_awlen,
    output wire [2:0]   ample_axi_awsize,
    output wire [1:0]   ample_axi_awburst,
    output wire         ample_axi_awlock,
    output wire [3:0]   ample_axi_awcache,
    output wire [2:0]   ample_axi_awprot,
    output wire [3:0]   ample_axi_awqos,
    output wire         ample_axi_awvalid,
    input  wire         ample_axi_awready,

    output wire [511:0] ample_axi_wdata,
    output wire [63:0]  ample_axi_wstrb,
    output wire         ample_axi_wlast,
    output wire         ample_axi_wvalid,
    input  wire         ample_axi_wready,

    input  wire [3:0]   ample_axi_bid,
    input  wire [1:0]   ample_axi_bresp,
    input  wire         ample_axi_bvalid,
    output wire         ample_axi_bready,

    output wire [3:0]   ample_axi_arid,
    output wire [33:0]  ample_axi_araddr,
    output wire [7:0]   ample_axi_arlen,
    output wire [2:0]   ample_axi_arsize,
    output wire [1:0]   ample_axi_arburst,
    output wire         ample_axi_arlock,
    output wire [3:0]   ample_axi_arcache,
    output wire [2:0]   ample_axi_arprot,
    output wire [3:0]   ample_axi_arqos,
    output wire         ample_axi_arvalid,
    input  wire         ample_axi_arready,

    input  wire [3:0]   ample_axi_rid,
    input  wire [511:0] ample_axi_rdata,
    input  wire [1:0]   ample_axi_rresp,
    input  wire         ample_axi_rlast,
    input  wire         ample_axi_rvalid,
    output wire         ample_axi_rready,

    // Nodeslot AXI Master Interface
    output wire [3:0]   nodeslot_axi_awid,
    output wire [33:0]  nodeslot_axi_awaddr,
    output wire [7:0]   nodeslot_axi_awlen,
    output wire [2:0]   nodeslot_axi_awsize,
    output wire [1:0]   nodeslot_axi_awburst,
    output wire [0:0]   nodeslot_axi_awlock,
    output wire [3:0]   nodeslot_axi_awcache,
    output wire [2:0]   nodeslot_axi_awprot,
    output wire [3:0]   nodeslot_axi_awqos,
    output wire         nodeslot_axi_awvalid,
    input  wire         nodeslot_axi_awready,

    output wire [511:0] nodeslot_axi_wdata,
    output wire [63:0]  nodeslot_axi_wstrb,
    output wire         nodeslot_axi_wlast,
    output wire         nodeslot_axi_wvalid,
    input  wire         nodeslot_axi_wready,

    input  wire [3:0]   nodeslot_axi_bid,
    input  wire [1:0]   nodeslot_axi_bresp,
    input  wire         nodeslot_axi_bvalid,
    output wire         nodeslot_axi_bready,

    output wire [3:0]   nodeslot_axi_arid,
    output wire [33:0]  nodeslot_axi_araddr,
    output wire [7:0]   nodeslot_axi_arlen,
    output wire [2:0]   nodeslot_axi_arsize,
    output wire [1:0]   nodeslot_axi_arburst,
    output wire [0:0]   nodeslot_axi_arlock,
    output wire [3:0]   nodeslot_axi_arcache,
    output wire [2:0]   nodeslot_axi_arprot,
    output wire [3:0]   nodeslot_axi_arqos,
    output wire         nodeslot_axi_arvalid,
    input  wire         nodeslot_axi_arready,

    input  wire [511:0] nodeslot_axi_rdata,
    input  wire [3:0]   nodeslot_axi_rid,
    input  wire [1:0]   nodeslot_axi_rresp,
    input  wire         nodeslot_axi_rlast,
    input  wire         nodeslot_axi_rvalid,
    output wire         nodeslot_axi_rready,





    //////////////////////////DEBUG -ILA//////////////////////////////////
    // NSB -> Aggregation Engine Interface
    output logic                                                debug_nsb_age_req_valid,
    output logic                                                debug_nsb_age_req_ready,
    // output logic [($clog2(MAX_NODESLOT_COUNT) + NODE_ID_WIDTH + $clog2(MESSAGE_CHANNEL_COUNT) + NODE_PRECISION_e_WIDTH + AGGREGATION_FUNCTION_e_WIDTH) - 1:0] debug_nsb_age_req,

    output logic                                                debug_nsb_age_resp_valid, // valid only for now
    // output logic [($clog2(MAX_NODESLOT_COUNT)) - 1:0] debug_nsb_age_resp,


    // NSB -> Transformation Engine Interface
    output logic                                                debug_nsb_fte_req_valid,
    output logic                                                debug_nsb_fte_req_ready,
    // output logic [(MAX_NODESLOT_COUNT + NODE_PRECISION_e_WIDTH + AGGREGATION_BUFFER_SLOTS) - 1:0] debug_nsb_fte_req,

    output logic                                                debug_nsb_fte_resp_valid, // valid only for now
    // output logic [(MAX_NODESLOT_COUNT + NODE_PRECISION_e_WIDTH) - 1:0] debug_nsb_fte_resp,


    // NSB -> Prefetcher Interface
    output logic                                                debug_nsb_prefetcher_req_valid,
    output logic                                                debug_nsb_prefetcher_req_ready,
    // output logic [3 + AXI_ADDRESS_WIDTH + ($clog2(MAX_FEATURE_COUNT) + 1) + ($clog2(MAX_FEATURE_COUNT) + 1) + $clog2(MAX_NODESLOT_COUNT) + NODE_PRECISION_e_WIDTH + $clog2(MAX_NEIGHBOURS) :0]   debug_nsb_prefetcher_req,
    output logic                                                debug_nsb_prefetcher_resp_valid // valid only for now
    // output [$clog2(MAX_NODESLOT_COUNT) + 3 + $clog2(MESSAGE_CHANNEL_COUNT):0]   debug_nsb_prefetcher_resp




//////////////////////////////////////////////////////////////////////



);

// // AXI-L Register Programming Interface
// wire [31:0]  host_axil_awaddr;
// wire [2:0]   host_axil_awprot;
// wire [0:0]   host_axil_awvalid;
// wire [0:0]   host_axil_awready;

// wire [31:0]  host_axil_wdata;
// wire [3:0]   host_axil_wstrb;
// wire [0:0]   host_axil_wvalid;
// wire [0:0]   host_axil_wready;

// wire [1:0]   host_axil_bresp;
// wire [0:0]   host_axil_bvalid;
// wire [0:0]   host_axil_bready;

// wire [31:0]  host_axil_araddr;
// wire [2:0]   host_axil_arprot;
// wire [0:0]   host_axil_arvalid;
// wire [0:0]   host_axil_arready;

// wire [31:0]  host_axil_rdata;
// wire [1:0]   host_axil_rresp;
// wire [0:0]   host_axil_rvalid;
// wire [0:0]   host_axil_rready;


//     // Ample AXI Master Interface
// wire [3:0]   ample_axi_awid;
// wire [33:0]  ample_axi_awaddr;
// wire [7:0]   ample_axi_awlen;
// wire [2:0]   ample_axi_awsize;
// wire [1:0]   ample_axi_awburst;
// wire         ample_axi_awlock;
// wire [3:0]   ample_axi_awcache;
// wire [2:0]   ample_axi_awprot;
// wire [3:0]   ample_axi_awqos;
// wire         ample_axi_awvalid;
// wire         ample_axi_awready;

// wire [511:0] ample_axi_wdata;
// wire [63:0]  ample_axi_wstrb;
// wire         ample_axi_wlast;
// wire         ample_axi_wvalid;
// wire         ample_axi_wready;

// wire [3:0]   ample_axi_bid;
// wire [1:0]   ample_axi_bresp;
// wire         ample_axi_bvalid;
// wire         ample_axi_bready;

// wire [3:0]   ample_axi_arid;
// wire [33:0]  ample_axi_araddr;
// wire [7:0]   ample_axi_arlen;
// wire [2:0]   ample_axi_arsize;
// wire [1:0]   ample_axi_arburst;
// wire         ample_axi_arlock;
// wire [3:0]   ample_axi_arcache;
// wire [2:0]   ample_axi_arprot;
// wire [3:0]   ample_axi_arqos;
// wire         ample_axi_arvalid;
// wire         ample_axi_arready;

// wire [3:0]   ample_axi_rid;
// wire [511:0] ample_axi_rdata;
// wire [1:0]   ample_axi_rresp;
// wire         ample_axi_rlast;
// wire         ample_axi_rvalid;
// wire         ample_axi_rready;

//     // Nodeslot AXI Master Interface
// wire [3:0]   nodeslot_axi_awid;
// wire [33:0]  nodeslot_axi_awaddr;
// wire [7:0]   nodeslot_axi_awlen;
// wire [2:0]   nodeslot_axi_awsize;
// wire [1:0]   nodeslot_axi_awburst;
// wire [0:0]   nodeslot_axi_awlock;
// wire [3:0]   nodeslot_axi_awcache;
// wire [2:0]   nodeslot_axi_awprot;
// wire [3:0]   nodeslot_axi_awqos;
// wire         nodeslot_axi_awvalid;
// wire         nodeslot_axi_awready;

// wire [511:0] nodeslot_axi_wdata;
// wire [63:0]  nodeslot_axi_wstrb;
// wire         nodeslot_axi_wlast;
// wire         nodeslot_axi_wvalid;
// wire         nodeslot_axi_wready;

// wire [3:0]   nodeslot_axi_bid;
// wire [1:0]   nodeslot_axi_bresp;
// wire         nodeslot_axi_bvalid;
// wire         nodeslot_axi_bready;

// wire [3:0]   nodeslot_axi_arid;
// wire [33:0]  nodeslot_axi_araddr;
// wire [7:0]   nodeslot_axi_arlen;
// wire [2:0]   nodeslot_axi_arsize;
// wire [1:0]   nodeslot_axi_arburst;
// wire [0:0]   nodeslot_axi_arlock;
// wire [3:0]   nodeslot_axi_arcache;
// wire [2:0]   nodeslot_axi_arprot;
// wire [3:0]   nodeslot_axi_arqos;
// wire         nodeslot_axi_arvalid;
// wire         nodeslot_axi_arready;

// wire [511:0] nodeslot_axi_rdata;
// wire [3:0]   nodeslot_axi_rid;
// wire [1:0]   nodeslot_axi_rresp;
// wire         nodeslot_axi_rlast;
// wire         nodeslot_axi_rvalid;
// wire         nodeslot_axi_rready;





// ====================================================================================
// TOP RTL
// ====================================================================================

// =============================AMPLE==============================================

  top top_i
  (
      .sys_clk                                (sys_clk),
      .sys_rst                                (sys_rst),

      .regbank_clk                            (regbank_clk),
      .regbank_resetn                         (regbank_resetn),
      
      // AXI-L Register Programming Interface
      .host_axil_awaddr                       (host_axil_awaddr),
      .host_axil_awprot                       (host_axil_awprot),
      .host_axil_awvalid                      (host_axil_awvalid),
      .host_axil_awready                      (host_axil_awready),
      .host_axil_wdata                        (host_axil_wdata),
      .host_axil_wstrb                        (host_axil_wstrb),
      .host_axil_wvalid                       (host_axil_wvalid),
      .host_axil_bready                       (host_axil_bready),
      .host_axil_araddr                       (host_axil_araddr),
      .host_axil_arprot                       (host_axil_arprot),
      .host_axil_arvalid                      (host_axil_arvalid),
      .host_axil_rready                       (host_axil_rready),
      .host_axil_wready                       (host_axil_wready),
      .host_axil_bresp                        (host_axil_bresp),
      .host_axil_bvalid                       (host_axil_bvalid),
      .host_axil_arready                      (host_axil_arready),
      .host_axil_rdata                        (host_axil_rdata),
      .host_axil_rresp                        (host_axil_rresp),
      .host_axil_rvalid                       (host_axil_rvalid),

      // AXI Master -> DDR4 Interface
      .ample_axi_awid                         (ample_axi_awid),
      .ample_axi_awaddr                       (ample_axi_awaddr),
      .ample_axi_awlen                        (ample_axi_awlen),
      .ample_axi_awsize                       (ample_axi_awsize),
      .ample_axi_awburst                      (ample_axi_awburst),
      .ample_axi_awlock                       (ample_axi_awlock),
      .ample_axi_awcache                      (ample_axi_awcache),
      .ample_axi_awprot                       (ample_axi_awprot),
      .ample_axi_awqos                        (ample_axi_awqos),
      .ample_axi_awvalid                      (ample_axi_awvalid),
      .ample_axi_awready                      (ample_axi_awready),
      .ample_axi_wdata                        (ample_axi_wdata),
      .ample_axi_wstrb                        (ample_axi_wstrb),
      .ample_axi_wlast                        (ample_axi_wlast),
      .ample_axi_wvalid                       (ample_axi_wvalid),
      .ample_axi_wready                       (ample_axi_wready),
      .ample_axi_bid                          (ample_axi_bid),
      .ample_axi_bresp                        (ample_axi_bresp),
      .ample_axi_bvalid                       (ample_axi_bvalid),
      .ample_axi_bready                       (ample_axi_bready),
      .ample_axi_arid                         (ample_axi_arid),
      .ample_axi_araddr                       (ample_axi_araddr),
      .ample_axi_arlen                        (ample_axi_arlen),
      .ample_axi_arsize                       (ample_axi_arsize),
      .ample_axi_arburst                      (ample_axi_arburst),
      .ample_axi_arlock                       (ample_axi_arlock),
      .ample_axi_arcache                      (ample_axi_arcache),
      .ample_axi_arprot                       (ample_axi_arprot),
      .ample_axi_arqos                        (ample_axi_arqos),
      .ample_axi_arvalid                      (ample_axi_arvalid),
      .ample_axi_arready                      (ample_axi_arready),
      .ample_axi_rid                          (ample_axi_rid),
      .ample_axi_rdata                        (ample_axi_rdata),
      .ample_axi_rresp                        (ample_axi_rresp),
      .ample_axi_rlast                        (ample_axi_rlast),
      .ample_axi_rvalid                       (ample_axi_rvalid),
      .ample_axi_rready                       (ample_axi_rready),


      .nodeslot_fetch_axi_arid                (nodeslot_axi_arid),
      .nodeslot_fetch_axi_araddr              (nodeslot_axi_araddr),
      .nodeslot_fetch_axi_arlen               (nodeslot_axi_arlen),
      .nodeslot_fetch_axi_arsize              (nodeslot_axi_arsize),
      .nodeslot_fetch_axi_arburst             (nodeslot_axi_arburst),
      .nodeslot_fetch_axi_arlock              (nodeslot_axi_arlock),
      .nodeslot_fetch_axi_arcache             (nodeslot_axi_arcache),
      .nodeslot_fetch_axi_arprot              (nodeslot_axi_arprot),
      .nodeslot_fetch_axi_arqos               (nodeslot_axi_arqos),
      .nodeslot_fetch_axi_arvalid             (nodeslot_axi_arvalid),
      .nodeslot_fetch_axi_arready             (nodeslot_axi_arready),
      .nodeslot_fetch_axi_rid                 (nodeslot_axi_rid),
      .nodeslot_fetch_axi_rdata               (nodeslot_axi_rdata),
      .nodeslot_fetch_axi_rresp               (nodeslot_axi_rresp),
      .nodeslot_fetch_axi_rlast               (nodeslot_axi_rlast),
      .nodeslot_fetch_axi_rvalid              (nodeslot_axi_rvalid),
      .nodeslot_fetch_axi_rready              (nodeslot_axi_rready)
  );




endmodule