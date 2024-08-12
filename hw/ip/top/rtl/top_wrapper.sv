
module top_wrapper (

    input  logic                                sys_clk,
    input  logic                                sys_rst,

    input  logic                                regbank_clk,
    input  logic                                regbank_resetn,

    // AXI-L interface to Host
    input  logic [0 : 0]                  host_axil_awvalid,
    output logic [0 : 0]                  host_axil_awready,
    input  logic [31 : 0]                 host_axil_awaddr,
    input  logic [2 : 0]                  host_axil_awprot,

    input  logic [0 : 0]                  host_axil_wvalid,
    output logic [0 : 0]                  host_axil_wready,
    input  logic [31 : 0]                 host_axil_wdata,
    input  logic [3 : 0]                  host_axil_wstrb,

    output logic [0 : 0]                  host_axil_bvalid,
    input  logic [0 : 0]                  host_axil_bready,
    output logic [1 : 0]                  host_axil_bresp,

    input  logic [0 : 0]                  host_axil_arvalid,
    output logic [0 : 0]                  host_axil_arready,
    input  logic [31 : 0]                 host_axil_araddr,
    input  logic [2 : 0]                  host_axil_arprot,

    output logic [0 : 0]                  host_axil_rvalid,
    input  logic [0 : 0]                  host_axil_rready,
    output logic [31 : 0]                 host_axil_rdata,
    output logic [1 : 0]                  host_axil_rresp     

);



// ====================================================================================
// Declarations
// ====================================================================================

  // AXI-L Register Programming Interface
  // logic [31 : 0]                 host_axil_awaddr;
  // logic [2 : 0]                  host_axil_awprot;
  // logic [0 : 0]                  host_axil_awvalid;
  // logic [0 : 0]                  host_axil_awready;
  // logic [31 : 0]                 host_axil_wdata;
  // logic [3 : 0]                  host_axil_wstrb;
  // logic [0 : 0]                  host_axil_wvalid;
  // logic [0 : 0]                  host_axil_bready;
  // logic [31 : 0]                 host_axil_araddr;
  // logic [2 : 0]                  host_axil_arprot;
  // logic [0 : 0]                  host_axil_arvalid;
  // logic [0 : 0]                  host_axil_rready;
  // logic [0 : 0]                  host_axil_wready;
  // logic [1 : 0]                  host_axil_bresp;
  // logic [0 : 0]                  host_axil_bvalid;
  // logic [0 : 0]                  host_axil_arready;
  // logic [31 : 0]                 host_axil_rdata;
  // logic [1 : 0]                  host_axil_rresp;
  // logic [0 : 0]                  host_axil_rvalid;



  // =============================Ample Axi Interface==============================================
  logic [3:0]         ample_axi_awid;
  logic [31:0]        ample_axi_awaddr;
  logic [7:0]         ample_axi_awlen;
  logic [2:0]         ample_axi_awsize;
  logic [1:0]         ample_axi_awburst;
  logic               ample_axi_awlock;
  logic [3:0]         ample_axi_awcache;
  logic [2:0]         ample_axi_awprot;
  logic [3:0]         ample_axi_awqos;
  logic               ample_axi_awvalid;
  logic               ample_axi_awready;
  logic [511:0]       ample_axi_wdata;
  logic [63:0]        ample_axi_wstrb;
  logic               ample_axi_wlast;
  logic               ample_axi_wvalid;
  logic               ample_axi_wready;
  logic [3:0]         ample_axi_bid;
  logic [1:0]         ample_axi_bresp;
  logic               ample_axi_bvalid;
  logic               ample_axi_bready;
  logic [3:0]         ample_axi_arid;
  logic [31:0]        ample_axi_araddr;
  logic [7:0]         ample_axi_arlen;
  logic [2:0]         ample_axi_arsize;
  logic [1:0]         ample_axi_arburst;
  logic               ample_axi_arlock;
  logic [3:0]         ample_axi_arcache;
  logic [2:0]         ample_axi_arprot;
  logic [3:0]         ample_axi_arqos;
  logic               ample_axi_arvalid;
  logic               ample_axi_arready;
  logic [3:0]         ample_axi_rid;
  logic [511:0]       ample_axi_rdata;
  logic [1:0]         ample_axi_rresp;
  logic               ample_axi_rlast;
  logic               ample_axi_rvalid;
  logic               ample_axi_rready;

  // =====================================Nodeslot AXI Signals========================================
  logic [3:0]         nodeslot_axi_awid;
  logic [33:0]        nodeslot_axi_awaddr;
  logic [7:0]         nodeslot_axi_awlen;
  logic [2:0]         nodeslot_axi_awsize;
  logic [1:0]         nodeslot_axi_awburst;
  logic [0:0]         nodeslot_axi_awlock;
  logic [3:0]         nodeslot_axi_awcache;
  logic [2:0]         nodeslot_axi_awprot;
  logic [3:0]         nodeslot_axi_awqos;
  logic               nodeslot_axi_awvalid;
  logic               nodeslot_axi_awready;
  logic [511:0]       nodeslot_axi_wdata;
  logic [63:0]        nodeslot_axi_wstrb;
  logic               nodeslot_axi_wlast;
  logic               nodeslot_axi_wvalid;
  logic               nodeslot_axi_wready;
  logic [3:0]         nodeslot_axi_bid;
  logic [1:0]         nodeslot_axi_bresp;
  logic               nodeslot_axi_bvalid;
  logic               nodeslot_axi_bready;
  logic [3:0]         nodeslot_axi_arid;
  logic [33:0]        nodeslot_axi_araddr;
  logic [7:0]         nodeslot_axi_arlen;
  logic [2:0]         nodeslot_axi_arsize;
  logic [1:0]         nodeslot_axi_arburst;
  logic [0:0]         nodeslot_axi_arlock;
  logic [3:0]         nodeslot_axi_arcache;
  logic [2:0]         nodeslot_axi_arprot;
  logic [3:0]         nodeslot_axi_arqos;
  logic               nodeslot_axi_arvalid;
  logic               nodeslot_axi_arready;
  logic [511:0]       nodeslot_axi_rdata;
  logic [3:0]         nodeslot_axi_rid;
  logic [1:0]         nodeslot_axi_rresp;
  logic               nodeslot_axi_rlast;
  logic               nodeslot_axi_rvalid;
  logic               nodeslot_axi_rready;


  assign nodeslot_axi_awid      = 4'b0;
  assign nodeslot_axi_awaddr    = 34'b0;
  assign nodeslot_axi_awlen     = 8'b0;
  assign nodeslot_axi_awsize    = 3'b0;
  assign nodeslot_axi_awburst   = 2'b0;
  assign nodeslot_axi_awlock    = 1'b0;
  assign nodeslot_axi_awcache   = 4'b0;
  assign nodeslot_axi_awprot    = 3'b0;
  assign nodeslot_axi_awqos     = 4'b0;
  assign nodeslot_axi_awvalid   = 1'b0;
  // assign nodeslot_axi_awready   = 1'b0;
  assign nodeslot_axi_wdata     = 512'b0;
  assign nodeslot_axi_wstrb     = 64'b0;
  assign nodeslot_axi_wlast     = 1'b0;
  assign nodeslot_axi_wvalid    = 1'b0;
  // assign nodeslot_axi_wready    = 1'b0;
  // assign nodeslot_axi_bid       = 4'b0;
  // assign nodeslot_axi_bresp     = 2'b0;
  // assign nodeslot_axi_bvalid    = 1'b0;
  assign nodeslot_axi_bready    = 1'b0;





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




    // ==============================Simulated Memory=========================================
    // DRAM model for feature memory
    axi_ram #(
        .DATA_WIDTH(512),
        .ADDR_WIDTH(34),
        .ID_WIDTH(8),
        .DATA_FILE  ("$WORKAREA/hw/sim/layer_config/nodeslots.mem")

    ) dram_c0_sim (
        .clk                    (sys_clk),
        .rst                    (sys_rst),
        .s_axi_awid             (ample_axi_awid),
        .s_axi_awaddr           (ample_axi_awaddr),
        .s_axi_awlen            (ample_axi_awlen),
        .s_axi_awsize           (ample_axi_awsize),
        .s_axi_awburst          (ample_axi_awburst),
        .s_axi_awlock           (ample_axi_awlock),
        .s_axi_awcache          (ample_axi_awcache),
        .s_axi_awprot           (ample_axi_awprot),
        .s_axi_awvalid          (ample_axi_awvalid),
        .s_axi_awready          (ample_axi_awready),
        .s_axi_wdata            (ample_axi_wdata),
        .s_axi_wstrb            (ample_axi_wstrb),
        .s_axi_wlast            (ample_axi_wlast),
        .s_axi_wvalid           (ample_axi_wvalid),
        .s_axi_wready           (ample_axi_wready),
        .s_axi_bid              (ample_axi_bid),
        .s_axi_bresp            (ample_axi_bresp),
        .s_axi_bvalid           (ample_axi_bvalid),
        .s_axi_bready           (ample_axi_bready),
        .s_axi_arid             (ample_axi_arid),
        .s_axi_araddr           (ample_axi_araddr),
        .s_axi_arlen            (ample_axi_arlen),
        .s_axi_arsize           (ample_axi_arsize),
        .s_axi_arburst          (ample_axi_arburst),
        .s_axi_arlock           (ample_axi_arlock),
        .s_axi_arcache          (ample_axi_arcache),
        .s_axi_arprot           (ample_axi_arprot),
        .s_axi_arvalid          (ample_axi_arvalid),
        .s_axi_arready          (ample_axi_arready),
        .s_axi_rid              (ample_axi_rid),
        .s_axi_rdata            (ample_axi_rdata),
        .s_axi_rresp            (ample_axi_rresp),
        .s_axi_rlast            (ample_axi_rlast),
        .s_axi_rvalid           (ample_axi_rvalid),
        .s_axi_rready           (ample_axi_rready)
    );
    
    // ==============================Nodeslot Memory========================================

    // DRAM model for nodeslot programming
    axi_ram #(
        .DATA_WIDTH (512),
        .ADDR_WIDTH (34),
        .ID_WIDTH   (8),
        .DATA_FILE  ("$WORKAREA/hw/sim/layer_config/nodeslots.mem")
    ) dram_c1_sim (
        .clk                    (sys_clk),
        .rst                    (sys_rst),

        .s_axi_awid             (nodeslot_axi_awid    ),
        .s_axi_awaddr           (nodeslot_axi_awaddr  ),
        .s_axi_awlen            (nodeslot_axi_awlen   ),
        .s_axi_awsize           (nodeslot_axi_awsize  ),
        .s_axi_awburst          (nodeslot_axi_awburst ),
        .s_axi_awlock           (nodeslot_axi_awlock  ),
        .s_axi_awcache          (nodeslot_axi_awcache ),
        .s_axi_awprot           (nodeslot_axi_awprot  ),
        .s_axi_awvalid          (nodeslot_axi_awvalid ),
        .s_axi_awready          (nodeslot_axi_awready ),
        .s_axi_wdata            (nodeslot_axi_wdata   ),
        .s_axi_wstrb            (nodeslot_axi_wstrb   ),
        .s_axi_wlast            (nodeslot_axi_wlast   ),
        .s_axi_wvalid           (nodeslot_axi_wvalid  ),
        .s_axi_wready           (nodeslot_axi_wready  ),
        .s_axi_bid              (nodeslot_axi_bid     ),
        .s_axi_bresp            (nodeslot_axi_bresp   ),
        .s_axi_bvalid           (nodeslot_axi_bvalid  ),
        .s_axi_bready           (nodeslot_axi_bready  ),
        .s_axi_arid             (nodeslot_axi_arid    ),
        .s_axi_araddr           (nodeslot_axi_araddr  ),
        .s_axi_arlen            (nodeslot_axi_arlen   ),
        .s_axi_arsize           (nodeslot_axi_arsize  ),
        .s_axi_arburst          (nodeslot_axi_arburst ),
        .s_axi_arlock           (nodeslot_axi_arlock  ),
        .s_axi_arcache          (nodeslot_axi_arcache ),
        .s_axi_arprot           (nodeslot_axi_arprot  ),
        .s_axi_arvalid          (nodeslot_axi_arvalid ),
        .s_axi_arready          (nodeslot_axi_arready ),
        .s_axi_rid              (nodeslot_axi_rid     ),
        .s_axi_rdata            (nodeslot_axi_rdata   ),
        .s_axi_rresp            (nodeslot_axi_rresp   ),
        .s_axi_rlast            (nodeslot_axi_rlast   ),
        .s_axi_rvalid           (nodeslot_axi_rvalid  ),
        .s_axi_rready           (nodeslot_axi_rready  )
    );



endmodule