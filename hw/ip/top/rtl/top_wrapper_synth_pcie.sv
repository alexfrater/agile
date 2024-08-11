
module top_wrapper (

    input  logic                                sys_clk,
    input  logic                                sys_rst,

    input  logic                                regbank_clk,
    input  logic                                regbank_resetn,


    //XDMA
    input  logic                                sys_clk_gt, //
    input  logic                                sys_rst_n, //



    // DDR4 Channel 0
    input  logic                                c0_sys_clk_p,
    input  logic                                c0_sys_clk_n,
    output logic                                c0_ddr4_act_n,
    output logic [16:0]                         c0_ddr4_adr,
    output logic [1:0]                          c0_ddr4_ba,
    output logic [1:0]                          c0_ddr4_bg,
    output logic [0:0]                          c0_ddr4_cke,
    output logic [0:0]                          c0_ddr4_odt,
    output logic [0:0]                          c0_ddr4_cs_n,
    output logic [0:0]                          c0_ddr4_ck_t,
    output logic [0:0]                          c0_ddr4_ck_c,
    output logic                                c0_ddr4_reset_n,
    output logic                                c0_ddr4_parity,
    inout  logic [71:0]                         c0_ddr4_dq,
    inout  logic [17:0]                         c0_ddr4_dqs_t,
    inout  logic [17:0]                         c0_ddr4_dqs_c,
    output logic                                c0_init_calib_complete,
    output logic                                c0_data_compare_error,

    // DDR4 Channel 1
    input  logic                                c1_sys_clk_p,
    input  logic                                c1_sys_clk_n,
    output logic                                c1_ddr4_act_n,
    output logic [16:0]                         c1_ddr4_adr,
    output logic [1:0]                          c1_ddr4_ba,
    output logic [1:0]                          c1_ddr4_bg,
    output logic [0:0]                          c1_ddr4_cke,
    output logic [0:0]                          c1_ddr4_odt,
    output logic [0:0]                          c1_ddr4_cs_n,
    output logic [0:0]                          c1_ddr4_ck_t,
    output logic [0:0]                          c1_ddr4_ck_c,
    output logic                                c1_ddr4_reset_n,
    output logic                                c1_ddr4_parity,
    inout  logic [71:0]                         c1_ddr4_dq,
    inout  logic [17:0]                         c1_ddr4_dqs_t,
    inout  logic [17:0]                         c1_ddr4_dqs_c,
    output logic                                c1_init_calib_complete,
    output logic                                c1_data_compare_error

);

// ====================================================================================
// Declarations
// ====================================================================================

// AXI-L Register Programming Interface
logic [31 : 0]                 host_axil_awaddr;
logic [2 : 0]                  host_axil_awprot;
logic [0 : 0]                  host_axil_awvalid;
logic [0 : 0]                  host_axil_awready;
logic [31 : 0]                 host_axil_wdata;
logic [3 : 0]                  host_axil_wstrb;
logic [0 : 0]                  host_axil_wvalid;
logic [0 : 0]                  host_axil_bready;
logic [31 : 0]                 host_axil_araddr;
logic [2 : 0]                  host_axil_arprot;
logic [0 : 0]                  host_axil_arvalid;
logic [0 : 0]                  host_axil_rready;
logic [0 : 0]                  host_axil_wready;
logic [1 : 0]                  host_axil_bresp;
logic [0 : 0]                  host_axil_bvalid;
logic [0 : 0]                  host_axil_arready;
logic [31 : 0]                 host_axil_rdata;
logic [1 : 0]                  host_axil_rresp;
logic [0 : 0]                  host_axil_rvalid;

// AXI Master Interface
logic  [7:0]                   c0_ddr4_s_axi_awid;
logic  [33:0]                  c0_ddr4_s_axi_awaddr;
logic  [7:0]                   c0_ddr4_s_axi_awlen;
logic  [2:0]                   c0_ddr4_s_axi_awsize;
logic  [1:0]                   c0_ddr4_s_axi_awburst;
logic  [0:0]                   c0_ddr4_s_axi_awlock;
logic  [3:0]                   c0_ddr4_s_axi_awcache;
logic  [2:0]                   c0_ddr4_s_axi_awprot;
logic  [3:0]                   c0_ddr4_s_axi_awqos;
logic                          c0_ddr4_s_axi_awvalid;
logic                          c0_ddr4_s_axi_awready;
logic  [511:0]                 c0_ddr4_s_axi_wdata;
logic  [63:0]                  c0_ddr4_s_axi_wstrb;
logic                          c0_ddr4_s_axi_wlast;
logic                          c0_ddr4_s_axi_wvalid;
logic                          c0_ddr4_s_axi_wready;
logic [7:0]                    c0_ddr4_s_axi_bid;
logic [1:0]                    c0_ddr4_s_axi_bresp;
logic                          c0_ddr4_s_axi_bvalid;
logic                          c0_ddr4_s_axi_bready;
logic  [7:0]                   c0_ddr4_s_axi_arid;
logic  [33:0]                  c0_ddr4_s_axi_araddr;
logic  [7:0]                   c0_ddr4_s_axi_arlen;
logic  [2:0]                   c0_ddr4_s_axi_arsize;
logic  [1:0]                   c0_ddr4_s_axi_arburst;
logic  [0:0]                   c0_ddr4_s_axi_arlock;
logic  [3:0]                   c0_ddr4_s_axi_arcache;
logic  [2:0]                   c0_ddr4_s_axi_arprot;
logic  [3:0]                   c0_ddr4_s_axi_arqos;
logic                          c0_ddr4_s_axi_arvalid;
logic                          c0_ddr4_s_axi_arready;
logic [7:0]                    c0_ddr4_s_axi_rid;
logic [511:0]                  c0_ddr4_s_axi_rdata;
logic [1:0]                    c0_ddr4_s_axi_rresp;
logic                          c0_ddr4_s_axi_rlast;
logic                          c0_ddr4_s_axi_rvalid;
logic                          c0_ddr4_s_axi_rready;

// AXI Master Interface
logic  [7:0]                   c1_ddr4_s_axi_awid;
logic  [33:0]                  c1_ddr4_s_axi_awaddr;
logic  [7:0]                   c1_ddr4_s_axi_awlen;
logic  [2:0]                   c1_ddr4_s_axi_awsize;
logic  [1:0]                   c1_ddr4_s_axi_awburst;
logic  [0:0]                   c1_ddr4_s_axi_awlock;
logic  [3:0]                   c1_ddr4_s_axi_awcache;
logic  [2:0]                   c1_ddr4_s_axi_awprot;
logic  [3:0]                   c1_ddr4_s_axi_awqos;
logic                          c1_ddr4_s_axi_awvalid;
logic                          c1_ddr4_s_axi_awready;
logic  [511:0]                 c1_ddr4_s_axi_wdata;
logic  [63:0]                  c1_ddr4_s_axi_wstrb;
logic                          c1_ddr4_s_axi_wlast;
logic                          c1_ddr4_s_axi_wvalid;
logic                          c1_ddr4_s_axi_wready;
logic [7:0]                    c1_ddr4_s_axi_bid;
logic [1:0]                    c1_ddr4_s_axi_bresp;
logic                          c1_ddr4_s_axi_bvalid;
logic                          c1_ddr4_s_axi_bready;
logic  [7:0]                   c1_ddr4_s_axi_arid;
logic  [33:0]                  c1_ddr4_s_axi_araddr;
logic  [7:0]                   c1_ddr4_s_axi_arlen;
logic  [2:0]                   c1_ddr4_s_axi_arsize;
logic  [1:0]                   c1_ddr4_s_axi_arburst;
logic  [0:0]                   c1_ddr4_s_axi_arlock;
logic  [3:0]                   c1_ddr4_s_axi_arcache;
logic  [2:0]                   c1_ddr4_s_axi_arprot;
logic  [3:0]                   c1_ddr4_s_axi_arqos;
logic                          c1_ddr4_s_axi_arvalid;
logic                          c1_ddr4_s_axi_arready;
logic [7:0]                    c1_ddr4_s_axi_rid;
logic [511:0]                  c1_ddr4_s_axi_rdata;
logic [1:0]                    c1_ddr4_s_axi_rresp;
logic                          c1_ddr4_s_axi_rlast;
logic                          c1_ddr4_s_axi_rvalid;
logic                          c1_ddr4_s_axi_rready;

logic c0_ddr4_aresetn;
logic c0_ddr4_reset_n_int;
logic c0_ddr4_clk;
logic c0_ddr4_rst;
logic dbg_clk_0;
logic [511:0] dbg_bus_1;

logic c1_ddr4_aresetn;
logic c1_ddr4_reset_n_int;
logic c1_ddr4_clk;
logic c1_ddr4_rst;
logic dbg_clk_1;
logic [511:0] dbg_bus_1;

assign c0_ddr4_reset_n = c0_ddr4_reset_n_int;
assign c1_ddr4_reset_n = c1_ddr4_reset_n_int;

// ====================================================================================
// TOP RTL
// ====================================================================================

// ====================================================================================
// Declarations
// ====================================================================================

// Prefetcher Read Master/s -> DRAM C0 (Read Only) /HBM - Set to 1 when using RAM(U250)
logic [HBM_BANKS-1:0]  [7:0]                   read_master_axi_arid;
logic [HBM_BANKS-1:0]  [33:0]                  read_master_axi_araddr;
logic [HBM_BANKS-1:0]  [7:0]                   read_master_axi_arlen;
logic [HBM_BANKS-1:0]  [2:0]                   read_master_axi_arsize;
logic [HBM_BANKS-1:0]  [1:0]                   read_master_axi_arburst;
logic [HBM_BANKS-1:0]  [0:0]                   read_master_axi_arlock;
logic [HBM_BANKS-1:0]  [3:0]                   read_master_axi_arcache;
logic [HBM_BANKS-1:0]  [2:0]                   read_master_axi_arprot;
logic [HBM_BANKS-1:0]  [3:0]                   read_master_axi_arqos;
logic [HBM_BANKS-1:0]                          read_master_axi_arvalid;
logic [HBM_BANKS-1:0]                          read_master_axi_arready;
logic [HBM_BANKS-1:0] [7:0]                    read_master_axi_rid;
logic [HBM_BANKS-1:0] [511:0]                  read_master_axi_rdata;
logic [HBM_BANKS-1:0] [1:0]                    read_master_axi_rresp;
logic [HBM_BANKS-1:0]                          read_master_axi_rlast;
logic [HBM_BANKS-1:0]                          read_master_axi_rvalid;
logic [HBM_BANKS-1:0]                          read_master_axi_rready;


// Transformation Engine -> DRAM C0 (Write Only)
logic  [7:0]                   transformation_engine_axi_awid;
logic  [33:0]                  transformation_engine_axi_awaddr;
logic  [7:0]                   transformation_engine_axi_awlen;
logic  [2:0]                   transformation_engine_axi_awsize;
logic  [1:0]                   transformation_engine_axi_awburst;
logic  [0:0]                   transformation_engine_axi_awlock;
logic  [3:0]                   transformation_engine_axi_awcache;
logic  [2:0]                   transformation_engine_axi_awprot;
logic  [3:0]                   transformation_engine_axi_awqos;
logic                          transformation_engine_axi_awvalid;
logic                          transformation_engine_axi_awready;
logic  [511:0]                 transformation_engine_axi_wdata;
logic  [63:0]                  transformation_engine_axi_wstrb;
logic                          transformation_engine_axi_wlast;
logic                          transformation_engine_axi_wvalid;
logic                          transformation_engine_axi_wready;
logic [7:0]                    transformation_engine_axi_bid;
logic [1:0]                    transformation_engine_axi_bresp;
logic                          transformation_engine_axi_bvalid;
logic                          transformation_engine_axi_bready;
// logic  [7:0]                   transformation_engine_axi_arid;
// logic  [33:0]                  transformation_engine_axi_araddr;
// logic  [7:0]                   transformation_engine_axi_arlen;
// logic  [2:0]                   transformation_engine_axi_arsize;
// logic  [1:0]                   transformation_engine_axi_arburst;
// logic  [0:0]                   transformation_engine_axi_arlock;
// logic  [3:0]                   transformation_engine_axi_arcache;
// logic  [2:0]                   transformation_engine_axi_arprot;
// logic  [3:0]                   transformation_engine_axi_arqos;
// logic                          transformation_engine_axi_arvalid;
// logic                          transformation_engine_axi_arready;
// logic [7:0]                    transformation_engine_axi_rid;
// logic [511:0]                  transformation_engine_axi_rdata;
// logic [1:0]                    transformation_engine_axi_rresp;
// logic                          transformation_engine_axi_rlast;
// logic                          transformation_engine_axi_rvalid;
// logic                          transformation_engine_axi_rready;

logic [33:0]                       weight_bank_axi_araddr;
logic [1:0]                        weight_bank_axi_arburst;
logic [3:0]                        weight_bank_axi_arcache;
logic [3:0]                        weight_bank_axi_arid;
logic [7:0]                        weight_bank_axi_arlen;
logic [0:0]                        weight_bank_axi_arlock;
logic [2:0]                        weight_bank_axi_arprot;
logic [3:0]                        weight_bank_axi_arqos;
logic [2:0]                        weight_bank_axi_arsize;
logic                              weight_bank_axi_arvalid;
logic                              weight_bank_axi_arready;
logic [3:0]                        weight_bank_axi_rid;
logic [511:0]                      weight_bank_axi_rdata;
logic [1:0]                        weight_bank_axi_rresp;
logic                              weight_bank_axi_rlast;
logic                              weight_bank_axi_rvalid;
logic                              weight_bank_axi_rready;
// logic [33:0]                       weight_bank_axi_awaddr;
// logic [1:0]                        weight_bank_axi_awburst;
// logic [3:0]                        weight_bank_axi_awcache;
// logic [3:0]                        weight_bank_axi_awid;
// logic [7:0]                        weight_bank_axi_awlen;
// logic [0:0]                        weight_bank_axi_awlock;
// logic [2:0]                        weight_bank_axi_awprot;
// logic [3:0]                        weight_bank_axi_awqos;
// logic                              weight_bank_axi_awready;
// logic [2:0]                        weight_bank_axi_awsize;
// logic                              weight_bank_axi_awvalid;
// logic [3:0]                        weight_bank_axi_bid;
// logic                              weight_bank_axi_bready;
// logic [1:0]                        weight_bank_axi_bresp;
// logic                              weight_bank_axi_bvalid;

// logic [511:0]                      weight_bank_axi_wdata;
// logic                              weight_bank_axi_wlast;
// logic                              weight_bank_axi_wready;
// logic [63:0]                       weight_bank_axi_wstrb;
// logic                              weight_bank_axi_wvalid;


// Nodeslot prefetcher -> DRAM C1
logic  [7:0]                   nodeslot_fetch_axi_arid;
logic  [33:0]                  nodeslot_fetch_axi_araddr;
logic  [7:0]                   nodeslot_fetch_axi_arlen;
logic  [2:0]                   nodeslot_fetch_axi_arsize;
logic  [1:0]                   nodeslot_fetch_axi_arburst;
logic  [0:0]                   nodeslot_fetch_axi_arlock;
logic  [3:0]                   nodeslot_fetch_axi_arcache;
logic  [2:0]                   nodeslot_fetch_axi_arprot;
logic  [3:0]                   nodeslot_fetch_axi_arqos;
logic                          nodeslot_fetch_axi_arvalid;
logic                          nodeslot_fetch_axi_arready;
logic [7:0]                    nodeslot_fetch_axi_rid;
logic [511:0]                  nodeslot_fetch_axi_rdata;
logic [1:0]                    nodeslot_fetch_axi_rresp;
logic                          nodeslot_fetch_axi_rlast;
logic                          nodeslot_fetch_axi_rvalid;
logic                          nodeslot_fetch_axi_rready;

top top_i
(
    .sys_clk                              (sys_clk),
    .sys_rst                              (sys_rst),

    .regbank_clk                          (regbank_clk),
    .regbank_resetn                       (regbank_resetn),
    
    // AXI-L Register Programming Interface
    .host_axil_awaddr                     (host_axil_awaddr),
    .host_axil_awprot                     (host_axil_awprot),
    .host_axil_awvalid                    (host_axil_awvalid),
    .host_axil_awready                    (host_axil_awready),
    .host_axil_wdata                      (host_axil_wdata),
    .host_axil_wstrb                      (host_axil_wstrb),
    .host_axil_wvalid                     (host_axil_wvalid),
    .host_axil_bready                     (host_axil_bready),
    .host_axil_araddr                     (host_axil_araddr),
    .host_axil_arprot                     (host_axil_arprot),
    .host_axil_arvalid                    (host_axil_arvalid),
    .host_axil_rready                     (host_axil_rready),
    .host_axil_wready                     (host_axil_wready),
    .host_axil_bresp                      (host_axil_bresp),
    .host_axil_bvalid                     (host_axil_bvalid),
    .host_axil_arready                    (host_axil_arready),
    .host_axil_rdata                      (host_axil_rdata),
    .host_axil_rresp                      (host_axil_rresp),
    .host_axil_rvalid                     (host_axil_rvalid),

    // AXI Master -> DDR4 Interface READ Only
    // .read_master_axi_awid                   (c0_ddr4_s_axi_awid),
    // .read_master_axi_awaddr                 (c0_ddr4_s_axi_awaddr),
    // .read_master_axi_awlen                  (c0_ddr4_s_axi_awlen),
    // .read_master_axi_awsize                 (c0_ddr4_s_axi_awsize),
    // .read_master_axi_awburst                (c0_ddr4_s_axi_awburst),
    // .read_master_axi_awlock                 (c0_ddr4_s_axi_awlock),
    // .read_master_axi_awcache                (c0_ddr4_s_axi_awcache),
    // .read_master_axi_awprot                 (c0_ddr4_s_axi_awprot),
    // .read_master_axi_awqos                  (c0_ddr4_s_axi_awqos),
    // .read_master_axi_awvalid                (c0_ddr4_s_axi_awvalid),
    // .read_master_axi_awready                (c0_ddr4_s_axi_awready),
    // .read_master_axi_wdata                  (c0_ddr4_s_axi_wdata),
    // .read_master_axi_wstrb                  (c0_ddr4_s_axi_wstrb),
    // .read_master_axi_wlast                  (c0_ddr4_s_axi_wlast),
    // .read_master_axi_wvalid                 (c0_ddr4_s_axi_wvalid),
    // .read_master_axi_wready                 (c0_ddr4_s_axi_wready),
    // .read_master_axi_bid                    (c0_ddr4_s_axi_bid),
    // .read_master_axi_bresp                  (c0_ddr4_s_axi_bresp),
    // .read_master_axi_bvalid                 (c0_ddr4_s_axi_bvalid),
    // .read_master_axi_bready                 (c0_ddr4_s_axi_bready),

    // Read Master -> DRAM Channel 0 (Read Only)
    .read_master_axi_arid,
    .read_master_axi_araddr,
    .read_master_axi_arlen,
    .read_master_axi_arsize,
    .read_master_axi_arburst,
    .read_master_axi_arlock,
    .read_master_axi_arcache,
    .read_master_axi_arprot,
    .read_master_axi_arqos,
    .read_master_axi_arvalid,
    .read_master_axi_arready,
    .read_master_axi_rid,
    .read_master_axi_rdata,
    .read_master_axi_rresp,
    .read_master_axi_rlast,
    .read_master_axi_rvalid,
    .read_master_axi_rready,

    //Feature Transformation Engine -> DRAM Channel 0 (Write Only)
    .transformation_engine_axi_awid,
    .transformation_engine_axi_awaddr,
    .transformation_engine_axi_awlen,
    .transformation_engine_axi_awsize,
    .transformation_engine_axi_awburst,
    .transformation_engine_axi_awlock,
    .transformation_engine_axi_awcache,
    .transformation_engine_axi_awprot,
    .transformation_engine_axi_awqos,
    .transformation_engine_axi_awvalid,
    .transformation_engine_axi_awready,
    .transformation_engine_axi_wdata,
    .transformation_engine_axi_wstrb,
    .transformation_engine_axi_wlast,
    .transformation_engine_axi_wvalid,
    .transformation_engine_axi_wready,
    .transformation_engine_axi_bid,
    .transformation_engine_axi_bresp,
    .transformation_engine_axi_bvalid,
    .transformation_engine_axi_bready,

    //Weight Bank  -> DRAM Channel 0 (Read Only)

    .weight_bank_axi_araddr,
    .weight_bank_axi_arburst,
    .weight_bank_axi_arcache,
    .weight_bank_axi_arid,
    .weight_bank_axi_arlen,
    .weight_bank_axi_arlock,
    .weight_bank_axi_arprot,
    .weight_bank_axi_arqos,
    .weight_bank_axi_arsize,
    .weight_bank_axi_arvalid,
    .weight_bank_axi_arready,
    .weight_bank_axi_rdata,
    .weight_bank_axi_rid,
    .weight_bank_axi_rlast,
    .weight_bank_axi_rready,
    .weight_bank_axi_rresp,
    .weight_bank_axi_rvalid,


    // Nodeslot Prefetcher  -> DRAM Channel 1 (Read Only)
    .nodeslot_fetch_axi_arid                (c1_ddr4_s_axi_arid),
    .nodeslot_fetch_axi_araddr              (c1_ddr4_s_axi_araddr),
    .nodeslot_fetch_axi_arlen               (c1_ddr4_s_axi_arlen),
    .nodeslot_fetch_axi_arsize              (c1_ddr4_s_axi_arsize),
    .nodeslot_fetch_axi_arburst             (c1_ddr4_s_axi_arburst),
    .nodeslot_fetch_axi_arlock              (c1_ddr4_s_axi_arlock),
    .nodeslot_fetch_axi_arcache             (c1_ddr4_s_axi_arcache),
    .nodeslot_fetch_axi_arprot              (c1_ddr4_s_axi_arprot),
    .nodeslot_fetch_axi_arqos               (c1_ddr4_s_axi_arqos),
    .nodeslot_fetch_axi_arvalid             (c1_ddr4_s_axi_arvalid),
    .nodeslot_fetch_axi_arready             (c1_ddr4_s_axi_arready),
    .nodeslot_fetch_axi_rid                 (c1_ddr4_s_axi_rid),
    .nodeslot_fetch_axi_rdata               (c1_ddr4_s_axi_rdata),
    .nodeslot_fetch_axi_rresp               (c1_ddr4_s_axi_rresp),
    .nodeslot_fetch_axi_rlast               (c1_ddr4_s_axi_rlast),
    .nodeslot_fetch_axi_rvalid              (c1_ddr4_s_axi_rvalid),
    .nodeslot_fetch_axi_rready              (c1_ddr4_s_axi_rready)
);


//Interconnect

// C0 Read: Weights, Features, PCIe
// C0 Write: Transformation Engine, PCIe


// C1 Read: Nodeslot Prefetcher
// C1 Write: PCIe

xdma_0 host_interface (
  .sys_clk(sys_clk),                                    // input wire sys_clk
  .sys_clk_gt(sys_clk_gt),                              // input wire sys_clk_gt
  .sys_rst_n(sys_rst_n),                                // input wire sys_rst_n
  .user_lnk_up(user_lnk_up),                            // output wire user_lnk_up
  .pci_exp_txp(pci_exp_txp),                            // output wire [15 : 0] pci_exp_txp
  .pci_exp_txn(pci_exp_txn),                            // output wire [15 : 0] pci_exp_txn
  .pci_exp_rxp(pci_exp_rxp),                            // input wire [15 : 0] pci_exp_rxp
  .pci_exp_rxn(pci_exp_rxn),                            // input wire [15 : 0] pci_exp_rxn
  .axi_aclk(axi_aclk),                                  // output wire axi_aclk
  .axi_aresetn(axi_aresetn),                            // output wire axi_aresetn
  .usr_irq_req(usr_irq_req),                            // input wire [0 : 0] usr_irq_req
  .usr_irq_ack(usr_irq_ack),                            // output wire [0 : 0] usr_irq_ack
  .msi_enable(msi_enable),                              // output wire msi_enable
  .msi_vector_width(msi_vector_width),                  // output wire [2 : 0] msi_vector_width
  .m_axi_awready(m_axi_awready),                        // input wire m_axi_awready
  .m_axi_wready(m_axi_wready),                          // input wire m_axi_wready
  .m_axi_bid(m_axi_bid),                                // input wire [3 : 0] m_axi_bid
  .m_axi_bresp(m_axi_bresp),                            // input wire [1 : 0] m_axi_bresp
  .m_axi_bvalid(m_axi_bvalid),                          // input wire m_axi_bvalid
  .m_axi_arready(m_axi_arready),                        // input wire m_axi_arready
  .m_axi_rid(m_axi_rid),                                // input wire [3 : 0] m_axi_rid
  .m_axi_rdata(m_axi_rdata),                            // input wire [511 : 0] m_axi_rdata
  .m_axi_rresp(m_axi_rresp),                            // input wire [1 : 0] m_axi_rresp
  .m_axi_rlast(m_axi_rlast),                            // input wire m_axi_rlast
  .m_axi_rvalid(m_axi_rvalid),                          // input wire m_axi_rvalid
  .m_axi_awid(m_axi_awid),                              // output wire [3 : 0] m_axi_awid
  .m_axi_awaddr(m_axi_awaddr),                          // output wire [63 : 0] m_axi_awaddr
  .m_axi_awlen(m_axi_awlen),                            // output wire [7 : 0] m_axi_awlen
  .m_axi_awsize(m_axi_awsize),                          // output wire [2 : 0] m_axi_awsize
  .m_axi_awburst(m_axi_awburst),                        // output wire [1 : 0] m_axi_awburst
  .m_axi_awprot(m_axi_awprot),                          // output wire [2 : 0] m_axi_awprot
  .m_axi_awvalid(m_axi_awvalid),                        // output wire m_axi_awvalid
  .m_axi_awlock(m_axi_awlock),                          // output wire m_axi_awlock
  .m_axi_awcache(m_axi_awcache),                        // output wire [3 : 0] m_axi_awcache
  .m_axi_wdata(m_axi_wdata),                            // output wire [511 : 0] m_axi_wdata
  .m_axi_wstrb(m_axi_wstrb),                            // output wire [63 : 0] m_axi_wstrb
  .m_axi_wlast(m_axi_wlast),                            // output wire m_axi_wlast
  .m_axi_wvalid(m_axi_wvalid),                          // output wire m_axi_wvalid
  .m_axi_bready(m_axi_bready),                          // output wire m_axi_bready
  .m_axi_arid(m_axi_arid),                              // output wire [3 : 0] m_axi_arid
  .m_axi_araddr(m_axi_araddr),                          // output wire [63 : 0] m_axi_araddr
  .m_axi_arlen(m_axi_arlen),                            // output wire [7 : 0] m_axi_arlen
  .m_axi_arsize(m_axi_arsize),                          // output wire [2 : 0] m_axi_arsize
  .m_axi_arburst(m_axi_arburst),                        // output wire [1 : 0] m_axi_arburst
  .m_axi_arprot(m_axi_arprot),                          // output wire [2 : 0] m_axi_arprot
  .m_axi_arvalid(m_axi_arvalid),                        // output wire m_axi_arvalid
  .m_axi_arlock(m_axi_arlock),                          // output wire m_axi_arlock
  .m_axi_arcache(m_axi_arcache),                        // output wire [3 : 0] m_axi_arcache
  .m_axi_rready(m_axi_rready),                          // output wire m_axi_rready
  .m_axil_awaddr(m_axil_awaddr),                        // output wire [31 : 0] m_axil_awaddr
  .m_axil_awprot(m_axil_awprot),                        // output wire [2 : 0] m_axil_awprot
  .m_axil_awvalid(m_axil_awvalid),                      // output wire m_axil_awvalid
  .m_axil_awready(m_axil_awready),                      // input wire m_axil_awready
  .m_axil_wdata(m_axil_wdata),                          // output wire [31 : 0] m_axil_wdata
  .m_axil_wstrb(m_axil_wstrb),                          // output wire [3 : 0] m_axil_wstrb
  .m_axil_wvalid(m_axil_wvalid),                        // output wire m_axil_wvalid
  .m_axil_wready(m_axil_wready),                        // input wire m_axil_wready
  .m_axil_bvalid(m_axil_bvalid),                        // input wire m_axil_bvalid
  .m_axil_bresp(m_axil_bresp),                          // input wire [1 : 0] m_axil_bresp
  .m_axil_bready(m_axil_bready),                        // output wire m_axil_bready
  .m_axil_araddr(m_axil_araddr),                        // output wire [31 : 0] m_axil_araddr
  .m_axil_arprot(m_axil_arprot),                        // output wire [2 : 0] m_axil_arprot
  .m_axil_arvalid(m_axil_arvalid),                      // output wire m_axil_arvalid
  .m_axil_arready(m_axil_arready),                      // input wire m_axil_arready
  .m_axil_rdata(m_axil_rdata),                          // input wire [31 : 0] m_axil_rdata
  .m_axil_rresp(m_axil_rresp),                          // input wire [1 : 0] m_axil_rresp
  .m_axil_rvalid(m_axil_rvalid),                        // input wire m_axil_rvalid
  .m_axil_rready(m_axil_rready),                        // output wire m_axil_rready
  .cfg_mgmt_addr(cfg_mgmt_addr),                        // input wire [18 : 0] cfg_mgmt_addr
  .cfg_mgmt_write(cfg_mgmt_write),                      // input wire cfg_mgmt_write
  .cfg_mgmt_write_data(cfg_mgmt_write_data),            // input wire [31 : 0] cfg_mgmt_write_data
  .cfg_mgmt_byte_enable(cfg_mgmt_byte_enable),          // input wire [3 : 0] cfg_mgmt_byte_enable
  .cfg_mgmt_read(cfg_mgmt_read),                        // input wire cfg_mgmt_read
  .cfg_mgmt_read_data(cfg_mgmt_read_data),              // output wire [31 : 0] cfg_mgmt_read_data
  .cfg_mgmt_read_write_done(cfg_mgmt_read_write_done)  // output wire cfg_mgmt_read_write_done
);

//C0 Interconnect

//S0 Pcie R/W
//S1 Transformation Engine W
//S2 Weight R
//S3 Feature R

c0_interconnect_v1 c0_interconnect_i (
  .INTERCONNECT_ACLK        (INTERCONNECT_ACLK),        // input wire INTERCONNECT_ACLK
  .INTERCONNECT_ARESETN     (INTERCONNECT_ARESETN),  // input wire INTERCONNECT_ARESETN
  
  .S00_AXI_ARESET_OUT_N     (),  // output wire S00_AXI_ARESET_OUT_N
  .S00_AXI_ACLK             (),                  // input wire S00_AXI_ACLK
  .S00_AXI_AWID             (),                  // input wire [0 : 0] S00_AXI_AWID
  .S00_AXI_AWADDR           (),              // input wire [33 : 0] S00_AXI_AWADDR
  .S00_AXI_AWLEN            (),                // input wire [7 : 0] S00_AXI_AWLEN
  .S00_AXI_AWSIZE           (),               // input wire [2 : 0] S00_AXI_AWSIZE
  .S00_AXI_AWBURST          (),            // input wire [1 : 0] S00_AXI_AWBURST
  .S00_AXI_AWLOCK(),              // input wire S00_AXI_AWLOCK
  .S00_AXI_AWCACHE(),            // input wire [3 : 0] S00_AXI_AWCACHE
  .S00_AXI_AWPROT(),              // input wire [2 : 0] S00_AXI_AWPROT
  .S00_AXI_AWQOS(),                // input wire [3 : 0] S00_AXI_AWQOS
  .S00_AXI_AWVALID(),            // input wire S00_AXI_AWVALID
  .S00_AXI_AWREADY(),            // output wire S00_AXI_AWREADY
  .S00_AXI_WDATA(),                // input wire [511 : 0] S00_AXI_WDATA
  .S00_AXI_WSTRB(),                // input wire [63 : 0] S00_AXI_WSTRB
  .S00_AXI_WLAST(),                // input wire S00_AXI_WLAST
  .S00_AXI_WVALID(),              // input wire S00_AXI_WVALID
  .S00_AXI_WREADY(),              // output wire S00_AXI_WREADY
  .S00_AXI_BID(),                    // output wire [0 : 0] S00_AXI_BID
  .S00_AXI_BRESP(),                // output wire [1 : 0] S00_AXI_BRESP
  .S00_AXI_BVALID(),              // output wire S00_AXI_BVALID
  .S00_AXI_BREADY(),              // input wire S00_AXI_BREADY
  .S00_AXI_ARID(),                  // input wire [0 : 0] S00_AXI_ARID
  .S00_AXI_ARADDR(),              // input wire [33 : 0] S00_AXI_ARADDR
  .S00_AXI_ARLEN(),                // input wire [7 : 0] S00_AXI_ARLEN
  .S00_AXI_ARSIZE(),              // input wire [2 : 0] S00_AXI_ARSIZE
  .S00_AXI_ARBURST(),            // input wire [1 : 0] S00_AXI_ARBURST
  .S00_AXI_ARLOCK(),              // input wire S00_AXI_ARLOCK
  .S00_AXI_ARCACHE(),            // input wire [3 : 0] S00_AXI_ARCACHE
  .S00_AXI_ARPROT(),              // input wire [2 : 0] S00_AXI_ARPROT
  .S00_AXI_ARQOS(),                // input wire [3 : 0] S00_AXI_ARQOS
  .S00_AXI_ARVALID(),            // input wire S00_AXI_ARVALID
  .S00_AXI_ARREADY(),            // output wire S00_AXI_ARREADY
  .S00_AXI_RID(),                    // output wire [0 : 0] S00_AXI_RID
  .S00_AXI_RDATA(),                // output wire [511 : 0] S00_AXI_RDATA
  .S00_AXI_RRESP(),                // output wire [1 : 0] S00_AXI_RRESP
  .S00_AXI_RLAST(),                // output wire S00_AXI_RLAST
  .S00_AXI_RVALID(),              // output wire S00_AXI_RVALID
  .S00_AXI_RREADY(),              // input wire S00_AXI_RREADY


  .S01_AXI_ARESET_OUT_N                 (S01_AXI_ARESET_OUT_N),  // output wire S01_AXI_ARESET_OUT_N
  .S01_AXI_ACLK                 (S01_AXI_ACLK),                  // input wire S01_AXI_ACLK
  .S01_AXI_AWID                 (transformation_engine_axi_awid),                  // input wire [0 : 0] S01_AXI_AWID
  .S01_AXI_AWADDR                 (transformation_engine_axi_awaddr),              // input wire [33 : 0] S01_AXI_AWADDR
  .S01_AXI_AWLEN                  (transformation_engine_axi_awlen),                // input wire [7 : 0] S01_AXI_AWLEN
  .S01_AXI_AWSIZE                 (transformation_engine_axi_awsize),              // input wire [2 : 0] S01_AXI_AWSIZE
  .S01_AXI_AWBURST                  (transformation_engine_axi_awburst),            // input wire [1 : 0] S01_AXI_AWBURST
  .S01_AXI_AWLOCK                 (transformation_engine_axi_awlock),              // input wire S01_AXI_AWLOCK
  .S01_AXI_AWCACHE                  (transformation_engine_axi_awcache),            // input wire [3 : 0] S01_AXI_AWCACHE
  .S01_AXI_AWPROT                 (transformation_engine_axi_awprot),              // input wire [2 : 0] S01_AXI_AWPROT
  .S01_AXI_AWQOS                  (transformation_engine_axi_awqos),                // input wire [3 : 0] S01_AXI_AWQOS
  .S01_AXI_AWVALID                  (transformation_engine_axi_awvalid),            // input wire S01_AXI_AWVALID
  .S01_AXI_AWREADY                  (transformation_engine_axi_awready),            // output wire S01_AXI_AWREADY
  .S01_AXI_WDATA                  (transformation_engine_axi_wdata),                // input wire [511 : 0] S01_AXI_WDATA
  .S01_AXI_WSTRB                  (transformation_engine_axi_wstrb),                // input wire [63 : 0] S01_AXI_WSTRB
  .S01_AXI_WLAST(transformation_engine_axi_wlast),                // input wire S01_AXI_WLAST
  .S01_AXI_WVALID(transformation_engine_axi_wvalid),              // input wire S01_AXI_WVALID
  .S01_AXI_WREADY(transformation_engine_axi_wready),              // output wire S01_AXI_WREADY
  .S01_AXI_BID(transformation_engine_axi_bid),                    // output wire [0 : 0] S01_AXI_BID
  .S01_AXI_BRESP(transformation_engine_axi_bresp),                // output wire [1 : 0] S01_AXI_BRESP
  .S01_AXI_BVALID(transformation_engine_axi_bvalid),              // output wire S01_AXI_BVALID
  .S01_AXI_BREADY(transformation_engine_axi_bready),              // input wire S01_AXI_BREADY


  //Weight Bank
  .S02_AXI_ARESET_OUT_N(S02_AXI_ARESET_OUT_N),  // output wire S02_AXI_ARESET_OUT_N
  .S02_AXI_ACLK(S02_AXI_ACLK),                  // input wire S02_AXI_ACLK
  .S02_AXI_ARID(weight_bank_axi_arid),                  // input wire [0 : 0] S02_AXI_ARID
  .S02_AXI_ARADDR(weight_bank_axi_araddr),              // input wire [33 : 0] S02_AXI_ARADDR
  .S02_AXI_ARLEN(weight_bank_axi_arlen),                // input wire [7 : 0] S02_AXI_ARLEN
  .S02_AXI_ARSIZE(weight_bank_axi_arsize),              // input wire [2 : 0] S02_AXI_ARSIZE
  .S02_AXI_ARBURST(weight_bank_axi_arburst),            // input wire [1 : 0] S02_AXI_ARBURST
  .S02_AXI_ARLOCK(weight_bank_axi_arlock),              // input wire S02_AXI_ARLOCK
  .S02_AXI_ARCACHE(weight_bank_axi_arcache),            // input wire [3 : 0] S02_AXI_ARCACHE
  .S02_AXI_ARPROT(weight_bank_axi_arprot),              // input wire [2 : 0] S02_AXI_ARPROT
  .S02_AXI_ARQOS(weight_bank_axi_arqos),                // input wire [3 : 0] S02_AXI_ARQOS
  .S02_AXI_ARVALID(weight_bank_axi_arvalid),            // input wire S02_AXI_ARVALID
  .S02_AXI_ARREADY(weight_bank_axi_arready),            // output wire S02_AXI_ARREADY
  .S02_AXI_RID(weight_bank_axi_rid),                    // output wire [0 : 0] S02_AXI_RID
  .S02_AXI_RDATA(weight_bank_axi_rdata),                // output wire [511 : 0] S02_AXI_RDATA
  .S02_AXI_RRESP(weight_bank_axi_rresp),                // output wire [1 : 0] S02_AXI_RRESP
  .S02_AXI_RLAST(weight_bank_axi_rlast),                // output wire S02_AXI_RLAST
  .S02_AXI_RVALID(weight_bank_axi_rvalid),              // output wire S02_AXI_RVALID
  .S02_AXI_RREADY(weight_bank_axi_rready),              // input wire S02_AXI_RREADY

  //Feature Bank
  .S03_AXI_ARID(read_master_axi_arid),                  // input wire [0 : 0] S03_AXI_ARID
  .S03_AXI_ARADDR(read_master_axi_araddr),              // input wire [33 : 0] S03_AXI_ARADDR
  .S03_AXI_ARLEN(read_master_axi_arlen),                // input wire [7 : 0] S03_AXI_ARLEN
  .S03_AXI_ARSIZE(read_master_axi_arsize),              // input wire [2 : 0] S03_AXI_ARSIZE
  .S03_AXI_ARBURST(read_master_axi_arburst),            // input wire [1 : 0] S03_AXI_ARBURST
  .S03_AXI_ARLOCK(read_master_axi_arlock),              // input wire S03_AXI_ARLOCK
  .S03_AXI_ARCACHE(read_master_axi_arcache),            // input wire [3 : 0] S03_AXI_ARCACHE
  .S03_AXI_ARPROT(read_master_axi_arprot),              // input wire [2 : 0] S03_AXI_ARPROT
  .S03_AXI_ARQOS(read_master_axi_arqos),                // input wire [3 : 0] S03_AXI_ARQOS
  .S03_AXI_ARVALID(read_master_axi_arvalid),            // input wire S03_AXI_ARVALID
  .S03_AXI_ARREADY(read_master_axi_arready),            // output wire S03_AXI_ARREADY
  .S03_AXI_RID(read_master_axi_rid),                    // output wire [0 : 0] S03_AXI_RID
  .S03_AXI_RDATA(read_master_axi_rdata),                // output wire [511 : 0] S03_AXI_RDATA
  .S03_AXI_RRESP(read_master_axi_rresp),                // output wire [1 : 0] S03_AXI_RRESP
  .S03_AXI_RLAST(read_master_axi_rlast),                // output wire S03_AXI_RLAST
  .S03_AXI_RVALID(read_master_axi_rvalid),              // output wire S03_AXI_RVALID
  .S03_AXI_RREADY(read_master_axi_rready),              // input wire S03_AXI_RREADY


  //C0 In/Out
  .M00_AXI_ARESET_OUT_N(c0_ddr4_aresetn),
  .M00_AXI_ACLK(c0_ddr4_clk),
  .M00_AXI_AWID(c0_ddr4_s_axi_awid),
  .M00_AXI_AWADDR(c0_ddr4_s_axi_awaddr),
  .M00_AXI_AWLEN(c0_ddr4_s_axi_awlen),
  .M00_AXI_AWSIZE(c0_ddr4_s_axi_awsize),
  .M00_AXI_AWBURST(c0_ddr4_s_axi_awburst),
  .M00_AXI_AWLOCK(c0_ddr4_s_axi_awlock),
  .M00_AXI_AWCACHE(c0_ddr4_s_axi_awcache),
  .M00_AXI_AWPROT(c0_ddr4_s_axi_awprot),
  .M00_AXI_AWQOS(c0_ddr4_s_axi_awqos),
  .M00_AXI_AWVALID(c0_ddr4_s_axi_awvalid),
  .M00_AXI_AWREADY(c0_ddr4_s_axi_awready),
  .M00_AXI_WDATA(c0_ddr4_s_axi_wdata),
  .M00_AXI_WSTRB(c0_ddr4_s_axi_wstrb),
  .M00_AXI_WLAST(c0_ddr4_s_axi_wlast),
  .M00_AXI_WVALID(c0_ddr4_s_axi_wvalid),
  .M00_AXI_WREADY(c0_ddr4_s_axi_wready),
  .M00_AXI_BID(c0_ddr4_s_axi_bid),
  .M00_AXI_BRESP(c0_ddr4_s_axi_bresp),
  .M00_AXI_BVALID(c0_ddr4_s_axi_bvalid),
  .M00_AXI_BREADY(c0_ddr4_s_axi_bready),
  .M00_AXI_ARID(c0_ddr4_s_axi_arid),
  .M00_AXI_ARADDR(c0_ddr4_s_axi_araddr),
  .M00_AXI_ARLEN(c0_ddr4_s_axi_arlen),
  .M00_AXI_ARSIZE(c0_ddr4_s_axi_arsize),
  .M00_AXI_ARBURST(c0_ddr4_s_axi_arburst),
  .M00_AXI_ARLOCK(c0_ddr4_s_axi_arlock),
  .M00_AXI_ARCACHE(c0_ddr4_s_axi_arcache),
  .M00_AXI_ARPROT(c0_ddr4_s_axi_arprot),
  .M00_AXI_ARQOS(c0_ddr4_s_axi_arqos),
  .M00_AXI_ARVALID(c0_ddr4_s_axi_arvalid),
  .M00_AXI_ARREADY(c0_ddr4_s_axi_arready),
  .M00_AXI_RID(c0_ddr4_s_axi_rid),
  .M00_AXI_RDATA(c0_ddr4_s_axi_rdata),
  .M00_AXI_RRESP(c0_ddr4_s_axi_rresp),
  .M00_AXI_RLAST(c0_ddr4_s_axi_rlast),
  .M00_AXI_RVALID(c0_ddr4_s_axi_rvalid),
  .M00_AXI_RREADY(c0_ddr4_s_axi_rready)

);


// ====================================================================================
// DDR4 Controller
// ====================================================================================


ddr4_0 u_ddr4_0
(
    .sys_rst                          (sys_rst),

    .c0_sys_clk_p                     (c0_sys_clk_p),
    .c0_sys_clk_n                     (c0_sys_clk_n),
    .c0_init_calib_complete           (c0_init_calib_complete),
    .c0_ddr4_act_n                    (c0_ddr4_act_n),
    .c0_ddr4_adr                      (c0_ddr4_adr),
    .c0_ddr4_ba                       (c0_ddr4_ba),
    .c0_ddr4_bg                       (c0_ddr4_bg),
    .c0_ddr4_cke                      (c0_ddr4_cke),
    .c0_ddr4_odt                      (c0_ddr4_odt),
    .c0_ddr4_cs_n                     (c0_ddr4_cs_n),
    .c0_ddr4_ck_t                     (c0_ddr4_ck_t),
    .c0_ddr4_ck_c                     (c0_ddr4_ck_c),
    .c0_ddr4_reset_n                  (c0_ddr4_reset_n_int),

    .c0_ddr4_parity                   (c0_ddr4_parity),
    .c0_ddr4_dq                       (c0_ddr4_dq),
    .c0_ddr4_dqs_c                    (c0_ddr4_dqs_c),
    .c0_ddr4_dqs_t                    (c0_ddr4_dqs_t),

    .c0_ddr4_ui_clk                   (c0_ddr4_clk),
    .c0_ddr4_ui_clk_sync_rst          (c0_ddr4_rst),
    .addn_ui_clkout1                  (),
    .dbg_clk                          (dbg_clk_0),

    // AXI CTRL port
    .c0_ddr4_s_axi_ctrl_awvalid       (1'b0),
    .c0_ddr4_s_axi_ctrl_awready       (),
    .c0_ddr4_s_axi_ctrl_awaddr        (32'b0),
    // Slave Interface Write Data Ports
    .c0_ddr4_s_axi_ctrl_wvalid        (1'b0),
    .c0_ddr4_s_axi_ctrl_wready        (),
    .c0_ddr4_s_axi_ctrl_wdata         (32'b0),
    // Slave Interface Write Response Ports
    .c0_ddr4_s_axi_ctrl_bvalid        (),
    .c0_ddr4_s_axi_ctrl_bready        (1'b1),
    .c0_ddr4_s_axi_ctrl_bresp         (),
    // Slave Interface Read Address Ports
    .c0_ddr4_s_axi_ctrl_arvalid       (1'b0),
    .c0_ddr4_s_axi_ctrl_arready       (),
    .c0_ddr4_s_axi_ctrl_araddr        (32'b0),
    // Slave Interface Read Data Ports
    .c0_ddr4_s_axi_ctrl_rvalid        (),
    .c0_ddr4_s_axi_ctrl_rready        (1'b1),
    .c0_ddr4_s_axi_ctrl_rdata         (),
    .c0_ddr4_s_axi_ctrl_rresp         (),


    // Interrupt output
    .c0_ddr4_interrupt                (),

    // Slave Interface AXI ports
    .c0_ddr4_aresetn                     (c0_ddr4_aresetn),
    .c0_ddr4_s_axi_awid                  (c0_ddr4_s_axi_awid),
    .c0_ddr4_s_axi_awaddr                (c0_ddr4_s_axi_awaddr),
    .c0_ddr4_s_axi_awlen                 (c0_ddr4_s_axi_awlen),
    .c0_ddr4_s_axi_awsize                (c0_ddr4_s_axi_awsize),
    .c0_ddr4_s_axi_awburst               (c0_ddr4_s_axi_awburst),
    .c0_ddr4_s_axi_awlock                (c0_ddr4_s_axi_awlock),
    .c0_ddr4_s_axi_awcache               (c0_ddr4_s_axi_awcache),
    .c0_ddr4_s_axi_awprot                (c0_ddr4_s_axi_awprot),
    .c0_ddr4_s_axi_awqos                 (c0_ddr4_s_axi_awqos),
    .c0_ddr4_s_axi_awvalid               (c0_ddr4_s_axi_awvalid),
    .c0_ddr4_s_axi_awready               (c0_ddr4_s_axi_awready),
    .c0_ddr4_s_axi_wdata                 (c0_ddr4_s_axi_wdata),
    .c0_ddr4_s_axi_wstrb                 (c0_ddr4_s_axi_wstrb),
    .c0_ddr4_s_axi_wlast                 (c0_ddr4_s_axi_wlast),
    .c0_ddr4_s_axi_wvalid                (c0_ddr4_s_axi_wvalid),
    .c0_ddr4_s_axi_wready                (c0_ddr4_s_axi_wready),
    .c0_ddr4_s_axi_bid                   (c0_ddr4_s_axi_bid),
    .c0_ddr4_s_axi_bresp                 (c0_ddr4_s_axi_bresp),
    .c0_ddr4_s_axi_bvalid                (c0_ddr4_s_axi_bvalid),
    .c0_ddr4_s_axi_bready                (c0_ddr4_s_axi_bready),

    
    .c0_ddr4_s_axi_arid                  (c0_ddr4_s_axi_arid),
    .c0_ddr4_s_axi_araddr                (c0_ddr4_s_axi_araddr),
    .c0_ddr4_s_axi_arlen                 (c0_ddr4_s_axi_arlen),
    .c0_ddr4_s_axi_arsize                (c0_ddr4_s_axi_arsize),
    .c0_ddr4_s_axi_arburst               (c0_ddr4_s_axi_arburst),
    .c0_ddr4_s_axi_arlock                (c0_ddr4_s_axi_arlock),
    .c0_ddr4_s_axi_arcache               (c0_ddr4_s_axi_arcache),
    .c0_ddr4_s_axi_arprot                (c0_ddr4_s_axi_arprot),
    .c0_ddr4_s_axi_arqos                 (c0_ddr4_s_axi_arqos),
    .c0_ddr4_s_axi_arvalid               (c0_ddr4_s_axi_arvalid),
    .c0_ddr4_s_axi_arready               (c0_ddr4_s_axi_arready),
    .c0_ddr4_s_axi_rid                   (c0_ddr4_s_axi_rid),
    .c0_ddr4_s_axi_rdata                 (c0_ddr4_s_axi_rdata),
    .c0_ddr4_s_axi_rresp                 (c0_ddr4_s_axi_rresp),
    .c0_ddr4_s_axi_rlast                 (c0_ddr4_s_axi_rlast),
    .c0_ddr4_s_axi_rvalid                (c0_ddr4_s_axi_rvalid),
    .c0_ddr4_s_axi_rready                (c0_ddr4_s_axi_rready),

    // Debug Port
    .dbg_bus         (dbg_bus_0)                                             

);





//Connections to DDR4 Controller c1







ddr4_1 u_ddr4_1
(
    .sys_rst                          (sys_rst),

    .c0_sys_clk_p                     (c1_sys_clk_p),
    .c0_sys_clk_n                     (c1_sys_clk_n),
    .c0_init_calib_complete           (c1_init_calib_complete),
    .c0_ddr4_act_n                    (c1_ddr4_act_n),
    .c0_ddr4_adr                      (c1_ddr4_adr),
    .c0_ddr4_ba                       (c1_ddr4_ba),
    .c0_ddr4_bg                       (c1_ddr4_bg),
    .c0_ddr4_cke                      (c1_ddr4_cke),
    .c0_ddr4_odt                      (c1_ddr4_odt),
    .c0_ddr4_cs_n                     (c1_ddr4_cs_n),
    .c0_ddr4_ck_t                     (c1_ddr4_ck_t),
    .c0_ddr4_ck_c                     (c1_ddr4_ck_c),
    .c0_ddr4_reset_n                  (c1_ddr4_reset_n_int),

    .c0_ddr4_parity                   (c1_ddr4_parity),
    .c0_ddr4_dq                       (c1_ddr4_dq),
    .c0_ddr4_dqs_c                    (c1_ddr4_dqs_c),
    .c0_ddr4_dqs_t                    (c1_ddr4_dqs_t),

    .c0_ddr4_ui_clk                   (c1_ddr4_clk),
    .c0_ddr4_ui_clk_sync_rst          (c1_ddr4_rst),
    .addn_ui_clkout1                  (),
    .dbg_clk                          (dbg_clk_1),

    // AXI CTRL port
    .c0_ddr4_s_axi_ctrl_awvalid       (1'b0),
    .c0_ddr4_s_axi_ctrl_awready       (),
    .c0_ddr4_s_axi_ctrl_awaddr        (32'b0),
    // Slave Interface Write Data Ports
    .c0_ddr4_s_axi_ctrl_wvalid        (1'b0),
    .c0_ddr4_s_axi_ctrl_wready        (),
    .c0_ddr4_s_axi_ctrl_wdata         (32'b0),
    // Slave Interface Write Response Ports
    .c0_ddr4_s_axi_ctrl_bvalid        (),
    .c0_ddr4_s_axi_ctrl_bready        (1'b1),
    .c0_ddr4_s_axi_ctrl_bresp         (),
    // Slave Interface Read Address Ports
    .c0_ddr4_s_axi_ctrl_arvalid       (1'b0),
    .c0_ddr4_s_axi_ctrl_arready       (),
    .c0_ddr4_s_axi_ctrl_araddr        (32'b0),
    // Slave Interface Read Data Ports
    .c0_ddr4_s_axi_ctrl_rvalid        (),
    .c0_ddr4_s_axi_ctrl_rready        (1'b1),
    .c0_ddr4_s_axi_ctrl_rdata         (),
    .c0_ddr4_s_axi_ctrl_rresp         (),


    // Interrupt output
    .c0_ddr4_interrupt                (),

    // Slave Interface AXI ports
    .c0_ddr4_aresetn                     (c1_ddr4_aresetn),
    .c0_ddr4_s_axi_awid                  (c1_ddr4_s_axi_awid),
    .c0_ddr4_s_axi_awaddr                (c1_ddr4_s_axi_awaddr),
    .c0_ddr4_s_axi_awlen                 (c1_ddr4_s_axi_awlen),
    .c0_ddr4_s_axi_awsize                (c1_ddr4_s_axi_awsize),
    .c0_ddr4_s_axi_awburst               (c1_ddr4_s_axi_awburst),
    .c0_ddr4_s_axi_awlock                (c1_ddr4_s_axi_awlock),
    .c0_ddr4_s_axi_awcache               (c1_ddr4_s_axi_awcache),
    .c0_ddr4_s_axi_awprot                (c1_ddr4_s_axi_awprot),
    .c0_ddr4_s_axi_awqos                 (c1_ddr4_s_axi_awqos),
    .c0_ddr4_s_axi_awvalid               (c1_ddr4_s_axi_awvalid),
    .c0_ddr4_s_axi_awready               (c1_ddr4_s_axi_awready),
    .c0_ddr4_s_axi_wdata                 (c1_ddr4_s_axi_wdata),
    .c0_ddr4_s_axi_wstrb                 (c1_ddr4_s_axi_wstrb),
    .c0_ddr4_s_axi_wlast                 (c1_ddr4_s_axi_wlast),
    .c0_ddr4_s_axi_wvalid                (c1_ddr4_s_axi_wvalid),
    .c0_ddr4_s_axi_wready                (c1_ddr4_s_axi_wready),
    .c0_ddr4_s_axi_bid                   (c1_ddr4_s_axi_bid),
    .c0_ddr4_s_axi_bresp                 (c1_ddr4_s_axi_bresp),
    .c0_ddr4_s_axi_bvalid                (c1_ddr4_s_axi_bvalid),
    .c0_ddr4_s_axi_bready                (c1_ddr4_s_axi_bready),
    .c0_ddr4_s_axi_arid                  (c1_ddr4_s_axi_arid),
    .c0_ddr4_s_axi_araddr                (c1_ddr4_s_axi_araddr),
    .c0_ddr4_s_axi_arlen                 (c1_ddr4_s_axi_arlen),
    .c0_ddr4_s_axi_arsize                (c1_ddr4_s_axi_arsize),
    .c0_ddr4_s_axi_arburst               (c1_ddr4_s_axi_arburst),
    .c0_ddr4_s_axi_arlock                (c1_ddr4_s_axi_arlock),
    .c0_ddr4_s_axi_arcache               (c1_ddr4_s_axi_arcache),
    .c0_ddr4_s_axi_arprot                (c1_ddr4_s_axi_arprot),
    .c0_ddr4_s_axi_arqos                 (c1_ddr4_s_axi_arqos),
    .c0_ddr4_s_axi_arvalid               (c1_ddr4_s_axi_arvalid),
    .c0_ddr4_s_axi_arready               (c1_ddr4_s_axi_arready),
    .c0_ddr4_s_axi_rid                   (c1_ddr4_s_axi_rid),
    .c0_ddr4_s_axi_rdata                 (c1_ddr4_s_axi_rdata),
    .c0_ddr4_s_axi_rresp                 (c1_ddr4_s_axi_rresp),
    .c0_ddr4_s_axi_rlast                 (c1_ddr4_s_axi_rlast),
    .c0_ddr4_s_axi_rvalid                (c1_ddr4_s_axi_rvalid),
    .c0_ddr4_s_axi_rready                (c1_ddr4_s_axi_rready),

    // Debug Port
    .dbg_bus         (dbg_bus_1)                                             

);

always @(posedge c0_ddr4_clk) begin
  c0_ddr4_aresetn <= ~c0_ddr4_rst;
end

always @(posedge c1_ddr4_clk) begin
  c1_ddr4_aresetn <= ~c1_ddr4_rst;
end

endmodule