
module top_wrapper (

    input  logic                                sys_clk,
    input  logic                                sys_rst,

    input  logic                                regbank_clk,
    input  logic                                regbank_resetn,

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
    output logic                                c1_data_compare_error,


    // PCIe DMA Interface
    input  logic                                xdma_sys_clk,             // input wire sys_clk
    input  logic                                xdma_sys_clk_gt,          // input wire sys_clk_gt
    input  logic                                xdma_sys_rst_n,           // input wire sys_rst_n
    output logic                                xdma_user_lnk_up,         // output wire user_lnk_up
    output logic [15:0]                         xdma_pci_exp_txp,          // output wire [15 : 0] pci_exp_txp
    output logic [15:0]                         xdma_pci_exp_txn,          // output wire [15 : 0] pci_exp_txn
    input  logic [15:0]                         xdma_pci_exp_rxp,          // input wire [15 : 0] pci_exp_rxp
    input  logic [15:0]                         xdma_pci_exp_rxn,          // input wire [15 : 0] pci_exp_rxn
    output logic                                xdma_axi_aclk,             // output wire axi_aclk
    output logic                                xdma_axi_aresetn,          // output wire axi_aresetn
    input  logic [0:0]                          xdma_usr_irq_req,          // input wire [0 : 0] usr_irq_req
    output logic [0:0]                          xdma_usr_irq_ack,          // output wire [0 : 0] usr_irq_ack
    output logic                                xdma_msi_enable,           // output wire msi_enable
    output logic [2:0]                          xdma_msi_vector_width,     // output wire [2 : 0] msi_vector_width

    

);



// ====================================================================================
// Declarations
// ====================================================================================

  // =============================PCIe DMA Signals==============================================

  // AXI Master Interface signals
  logic           xdma_m_axi_awready;                
  logic           xdma_m_axi_wready;    
  logic [3:0]     xdma_m_axi_bid;      
  logic [1:0]     xdma_m_axi_bresp;                  
  logic           xdma_m_axi_bvalid;    
  logic           xdma_m_axi_arready;                
  logic [3:0]     xdma_m_axi_rid;      
  logic [511:0]   xdma_m_axi_rdata;                  
  logic [1:0]     xdma_m_axi_rresp;                  
  logic           xdma_m_axi_rlast;          
  logic           xdma_m_axi_rvalid;          
  logic [3:0]     xdma_m_axi_awid;                   
  logic [63:0]    xdma_m_axi_awaddr;                 
  logic [7:0]     xdma_m_axi_awlen;                   
  logic [2:0]     xdma_m_axi_awsize;                  
  logic [1:0]     xdma_m_axi_awburst;                 
  logic [2:0]     xdma_m_axi_awprot;                  
  logic           xdma_m_axi_awvalid;                 
  logic           xdma_m_axi_awlock;                 
  logic [3:0]     xdma_m_axi_awcache;                 
  logic [511:0]   xdma_m_axi_wdata;                   
  logic [63:0]    xdma_m_axi_wstrb;                   
  logic           xdma_m_axi_wlast;                           
  logic           xdma_m_axi_wvalid;                 
  logic           xdma_m_axi_bready;                 
  logic [3:0]     xdma_m_axi_arid;                   
  logic [63:0]    xdma_m_axi_araddr;                  
  logic [7:0]     xdma_m_axi_arlen;                   
  logic [2:0]     xdma_m_axi_arsize;                  
  logic [1:0]     xdma_m_axi_arburst;                 
  logic [2:0]     xdma_m_axi_arprot;                  
  logic           xdma_m_axi_arvalid;                 
  logic           xdma_m_axi_arlock;                 
  logic [3:0]     xdma_m_axi_arcache;                 
  logic           xdma_m_axi_rready;                 


  // Configuration Management Interface signals
  logic [18:0]    xdma_cfg_mgmt_addr;                  
  logic           xdma_cfg_mgmt_write;                
  logic [31:0]    xdma_cfg_mgmt_write_data;            
  logic [3:0]     xdma_cfg_mgmt_byte_enable;            
  logic           xdma_cfg_mgmt_read;                
  logic [31:0]    xdma_cfg_mgmt_read_data;             
  logic           xdma_cfg_mgmt_read_write_done;    



  // Logic signals for XDMA Memory AXI (Slave 00)
  logic [3:0]     xdma_mem_axi_awid;
  logic [33:0]    xdma_mem_axi_awaddr;
  logic [7:0]     xdma_mem_axi_awlen;
  logic [2:0]     xdma_mem_axi_awsize;
  logic [1:0]     xdma_mem_axi_awburst;
  logic [0:0]     xdma_mem_axi_awlock;
  logic [3:0]     xdma_mem_axi_awcache;
  logic [2:0]     xdma_mem_axi_awprot;
  logic [3:0]     xdma_mem_axi_awqos;
  logic           xdma_mem_axi_awvalid;
  logic           xdma_mem_axi_awready;
  logic [511:0]   xdma_mem_axi_wdata;
  logic [63:0]    xdma_mem_axi_wstrb;
  logic           xdma_mem_axi_wlast;
  logic           xdma_mem_axi_wvalid;
  logic           xdma_mem_axi_wready;
  logic [3:0]     xdma_mem_axi_bid;
  logic [1:0]     xdma_mem_axi_bresp;
  logic           xdma_mem_axi_bvalid;
  logic           xdma_mem_axi_bready;
  logic [3:0]     xdma_mem_axi_arid;
  logic [33:0]    xdma_mem_axi_araddr;
  logic [7:0]     xdma_mem_axi_arlen;
  logic [2:0]     xdma_mem_axi_arsize;
  logic [1:0]     xdma_mem_axi_arburst;
  logic [0:0]     xdma_mem_axi_arlock;
  logic [3:0]     xdma_mem_axi_arcache;
  logic [2:0]     xdma_mem_axi_arprot;
  logic [3:0]     xdma_mem_axi_arqos;
  logic           xdma_mem_axi_arvalid;
  logic           xdma_mem_axi_arready;
  logic [511:0]   xdma_mem_axi_rdata;
  logic [3:0]     xdma_mem_axi_rid;
  logic [1:0]     xdma_mem_axi_rresp;
  logic           xdma_mem_axi_rlast;
  logic           xdma_mem_axi_rvalid;
  logic           xdma_mem_axi_rready;


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

  // DDR4 C0 AXI Interface
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

  logic                          c0_ddr4_aresetn;
  logic                          c0_ddr4_reset_n_int;
  logic                          c0_ddr4_clk;
  logic                          c0_ddr4_rst;
  logic                          dbg_clk_0;
  logic [511:0]                  dbg_bus_1;

  logic                          c1_ddr4_aresetn;
  logic                          c1_ddr4_reset_n_int;
  logic                          c1_ddr4_clk;
  logic                          c1_ddr4_rst;
  logic                          dbg_clk_1;
  logic [511:0]                  dbg_bus_1;

  assign c0_ddr4_reset_n = c0_ddr4_reset_n_int;
  assign c1_ddr4_reset_n = c1_ddr4_reset_n_int;


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
      .nodeslot_fetch_axi_rready              (nodeslot_axi_rready),
  );



  //PCIe DMA
`ifdef SYNTHESIS

   xdma_0 xdma_i (
    .sys_clk                    (sys_clk),                                    // input wire sys_clk
    .sys_clk_gt                 (sys_clk_gt),                              // input wire sys_clk_gt
    .sys_rst_n                  (sys_rst_n),                                // input wire sys_rst_n
    .user_lnk_up                (xdma_user_lnk_up),                       // output wire user_lnk_up
    .pci_exp_txp                (xdma_pci_exp_txp),                       // output wire [15 : 0] pci_exp_txp
    .pci_exp_txn                (xdma_pci_exp_txn),                       // output wire [15 : 0] pci_exp_txn
    .pci_exp_rxp                (xdma_pci_exp_rxp),                       // input wire [15 : 0] pci_exp_rxp
    .pci_exp_rxn                (xdma_pci_exp_rxn),                       // input wire [15 : 0] pci_exp_rxn
    .axi_aclk                   (xdma_axi_aclk),                             // output wire axi_aclk
    .axi_aresetn                (xdma_axi_aresetn),                       // output wire axi_aresetn
    .usr_irq_req                (xdma_usr_irq_req),                       // input wire [0 : 0] usr_irq_req
    .usr_irq_ack                (xdma_usr_irq_ack),                       // output wire [0 : 0] usr_irq_ack
    .msi_enable                 (xdma_msi_enable),                         // output wire msi_enable
    .msi_vector_width           (xdma_msi_vector_width),             // output wire [2 : 0] msi_vector_width

    // AXI Master Interfac      e
    .m_axi_awready              (xdma_m_axi_awready),                   // input wire m_axi_awready
    .m_axi_wready               (xdma_m_axi_wready),                     // input wire m_axi_wready
    .m_axi_bid                  (xdma_m_axi_bid),                           // input wire [3 : 0] m_axi_bid
    .m_axi_bresp                (xdma_m_axi_bresp),                       // input wire [1 : 0] m_axi_bresp
    .m_axi_bvalid               (xdma_m_axi_bvalid),                     // input wire m_axi_bvalid
    .m_axi_arready              (xdma_m_axi_arready),                   // input wire m_axi_arready
    .m_axi_rid                  (xdma_m_axi_rid),                           // input wire [3 : 0] m_axi_rid
    .m_axi_rdata                (xdma_m_axi_rdata),                       // input wire [511 : 0] m_axi_rdata
    .m_axi_rresp                (xdma_m_axi_rresp),                       // input wire [1 : 0] m_axi_rresp
    .m_axi_rlast                (xdma_m_axi_rlast),                       // input wire m_axi_rlast
    .m_axi_rvalid               (xdma_m_axi_rvalid),                     // input wire m_axi_rvalid
    .m_axi_awid                 (xdma_m_axi_awid),                         // output wire [3 : 0] m_axi_awid
    .m_axi_awaddr               (xdma_m_axi_awaddr),                     // output wire [63 : 0] m_axi_awaddr
    .m_axi_awlen                (xdma_m_axi_awlen),                       // output wire [7 : 0] m_axi_awlen
    .m_axi_awsize               (xdma_m_axi_awsize),                     // output wire [2 : 0] m_axi_awsize
    .m_axi_awburst              (xdma_m_axi_awburst),                   // output wire [1 : 0] m_axi_awburst
    .m_axi_awprot               (xdma_m_axi_awprot),                     // output wire [2 : 0] m_axi_awprot
    .m_axi_awvalid              (xdma_m_axi_awvalid),                   // output wire m_axi_awvalid
    .m_axi_awlock               (xdma_m_axi_awlock),                     // output wire m_axi_awlock
    .m_axi_awcache              (xdma_m_axi_awcache),                   // output wire [3 : 0] m_axi_awcache
    .m_axi_wdata                (xdma_m_axi_wdata),                       // output wire [511 : 0] m_axi_wdata
    .m_axi_wstrb                (xdma_m_axi_wstrb),                       // output wire [63 : 0] m_axi_wstrb
    .m_axi_wlast                (xdma_m_axi_wlast),                       // output wire m_axi_wlast
    .m_axi_wvalid               (xdma_m_axi_wvalid),                     // output wire m_axi_wvalid
    .m_axi_bready               (xdma_m_axi_bready),                     // output wire m_axi_bready
    .m_axi_arid                 (xdma_m_axi_arid),                         // output wire [3 : 0] m_axi_arid
    .m_axi_araddr               (xdma_m_axi_araddr),                     // output wire [63 : 0] m_axi_araddr
    .m_axi_arlen                (xdma_m_axi_arlen),                       // output wire [7 : 0] m_axi_arlen
    .m_axi_arsize               (xdma_m_axi_arsize),                     // output wire [2 : 0] m_axi_arsize
    .m_axi_arburst              (xdma_m_axi_arburst),                   // output wire [1 : 0] m_axi_arburst
    .m_axi_arprot               (xdma_m_axi_arprot),                     // output wire [2 : 0] m_axi_arprot
    .m_axi_arvalid              (xdma_m_axi_arvalid),                   // output wire m_axi_arvalid
    .m_axi_arlock               (xdma_m_axi_arlock),                     // output wire m_axi_arlock
    .m_axi_arcache              (xdma_m_axi_arcache),                   // output wire [3 : 0] m_axi_arcache
    .m_axi_rready               (xdma_m_axi_rready),                     // output wire m_axi_rready

    // AXI Lite Master Int      
    .m_axil_awaddr              (host_axil_awaddr),                   // output wire [31 : 0] m_axil_awaddr
    .m_axil_awprot              (host_axil_awprot),                   // output wire [2 : 0] m_axil_awprot
    .m_axil_awvalid             (host_axil_awvalid),                 // output wire m_axil_awvalid
    .m_axil_awready             (host_axil_awready),                 // input wire m_axil_awready
    .m_axil_wdata               (host_axil_wdata),                     // output wire [31 : 0] m_axil_wdata
    .m_axil_wstrb               (host_axil_wstrb),                     // output wire [3 : 0] m_axil_wstrb
    .m_axil_wvalid              (host_axil_wvalid),                   // output wire m_axil_wvalid
    .m_axil_wready              (host_axil_wready),                   // input wire m_axil_wready
    .m_axil_bvalid              (host_axil_bvalid),                   // input wire m_axil_bvalid
    .m_axil_bresp               (host_axil_bresp),                     // input wire [1 : 0] m_axil_bresp
    .m_axil_bready              (host_axil_bready),                   // output wire m_axil_bready
    .m_axil_araddr              (host_axil_araddr),                   // output wire [31 : 0] m_axil_araddr
    .m_axil_arprot              (host_axil_arprot),                   // output wire [2 : 0] m_axil_arprot
    .m_axil_arvalid             (host_axil_arvalid),                 // output wire m_axil_arvalid
    .m_axil_arready             (host_axil_arready),                 // input wire m_axil_arready
    .m_axil_rdata               (host_axil_rdata),                     // input wire [31 : 0] m_axil_rdata
    .m_axil_rresp               (host_axil_rresp),                     // input wire [1 : 0] m_axil_rresp
    .m_axil_rvalid              (host_axil_rvalid),                   // input wire m_axil_rvalid
    .m_axil_rready              (host_axil_rready),                   // output wire m_axil_rready

    // Configuration Manag      ement Interface
    .cfg_mgmt_addr              (xdma_cfg_mgmt_addr),                   // input wire [18 : 0] cfg_mgmt_addr
    .cfg_mgmt_write             (xdma_cfg_mgmt_write),                 // input wire cfg_mgmt_write
    .cfg_mgmt_write_data        (xdma_cfg_mgmt_write_data),       // input wire [31 : 0] cfg_mgmt_write_data
    .cfg_mgmt_byte_enable       (xdma_cfg_mgmt_byte_enable),     // input wire [3 : 0] cfg_mgmt_byte_enable
    .cfg_mgmt_read              (xdma_cfg_mgmt_read),                   // input wire cfg_mgmt_read
    .cfg_mgmt_read_data         (xdma_cfg_mgmt_read_data),         // output wire [31 : 0] cfg_mgmt_read_data
    .cfg_mgmt_read_write_done   (xdma_cfg_mgmt_read_write_done)  // output wire cfg_mgmt_read_write_done
  );


`else 
  //Allow AXI-L to float - used in cocotb
  //Set AXI-Signals to zero
  assign xdma_m_axi_awid    = 4'b0000;
  assign xdma_m_axi_awaddr  = 64'b0;
  assign xdma_m_axi_awlen   = 8'b00000000;
  assign xdma_m_axi_awsize  = 3'b000;
  assign xdma_m_axi_awburst = 2'b00;
  assign xdma_m_axi_awprot  = 3'b000;
  assign xdma_m_axi_awvalid = 1'b0;
  assign xdma_m_axi_awlock  = 1'b0;
  assign xdma_m_axi_awcache = 4'b0000;
  assign xdma_m_axi_wdata   = 512'b0;
  assign xdma_m_axi_wstrb   = 64'b0;
  assign xdma_m_axi_wlast   = 1'b0;
  assign xdma_m_axi_wvalid  = 1'b0;
  assign xdma_m_axi_bready  = 1'b0;
  assign xdma_m_axi_arid    = 4'b0000;
  assign xdma_m_axi_araddr  = 64'b0;
  assign xdma_m_axi_arlen   = 8'b00000000;
  assign xdma_m_axi_arsize  = 3'b000;
  assign xdma_m_axi_arburst = 2'b00;
  assign xdma_m_axi_arprot  = 3'b000;
  assign xdma_m_axi_arvalid = 1'b0;
  assign xdma_m_axi_arlock  = 1'b0;
  assign xdma_m_axi_arcache = 4'b0000;
  assign xdma_m_axi_rready  = 1'b0;


  `endif



  axi_interconnect_wrap_1x2 #
  (
      .DATA_WIDTH(512),
      .ADDR_WIDTH(34),
      .STRB_WIDTH(DATA_WIDTH/8),
      .ID_WIDTH(4)
  ) xdma_write_interconnect_i
  (
      .clk                    (clk),
      .rst                    (rst),

      /*
      * AXI slave interface
      */
      .s00_axi_awid           (xdma_m_axi_awid),
      .s00_axi_awaddr         (xdma_m_axi_awaddr),
      .s00_axi_awlen          (xdma_m_axi_awlen),
      .s00_axi_awsize         (xdma_m_axi_awsize),
      .s00_axi_awburst        (xdma_m_axi_awburst),
      .s00_axi_awlock         (xdma_m_axi_awlock),
      .s00_axi_awcache        (xdma_m_axi_awcache),
      .s00_axi_awprot         (xdma_m_axi_awprot),
      .s00_axi_awqos          (xdma_m_axi_awqos),
      .s00_axi_awvalid        (xdma_m_axi_awvalid),
      .s00_axi_awready        (xdma_m_axi_awready),
      .s00_axi_wdata          (xdma_m_axi_wdata),
      .s00_axi_wstrb          (xdma_m_axi_wstrb),
      .s00_axi_wlast          (xdma_m_axi_wlast),
      .s00_axi_wvalid         (xdma_m_axi_wvalid),
      .s00_axi_wready         (xdma_m_axi_wready),
      .s00_axi_bid            (xdma_m_axi_bid),
      .s00_axi_bresp          (xdma_m_axi_bresp),
      .s00_axi_bvalid         (xdma_m_axi_bvalid),
      .s00_axi_bready         (xdma_m_axi_bready),
      .s00_axi_arid           (xdma_m_axi_arid),
      .s00_axi_araddr         (xdma_m_axi_araddr),
      .s00_axi_arlen          (xdma_m_axi_arlen),
      .s00_axi_arsize         (xdma_m_axi_arsize),
      .s00_axi_arburst        (xdma_m_axi_arburst),
      .s00_axi_arlock         (xdma_m_axi_arlock),
      .s00_axi_arcache        (xdma_m_axi_arcache),
      .s00_axi_arprot         (xdma_m_axi_arprot),
      .s00_axi_arqos          (xdma_m_axi_arqos),
      .s00_axi_arvalid        (xdma_m_axi_arvalid),
      .s00_axi_arready        (xdma_m_axi_arready),
      .s00_axi_rid            (xdma_m_axi_rid),
      .s00_axi_rdata          (xdma_m_axi_rdata),
      .s00_axi_rresp          (xdma_m_axi_rresp),
      .s00_axi_rlast          (xdma_m_axi_rlast),
      .s00_axi_rvalid         (xdma_m_axi_rvalid),
      .s00_axi_rready         (xdma_m_axi_rready),
      /*
      * AXI master interface for Memory (Slave 00)
      */
      .m00_axi_awid           (xdma_mem_axi_awid),
      .m00_axi_awaddr         (xdma_mem_axi_awaddr),
      .m00_axi_awlen          (xdma_mem_axi_awlen),
      .m00_axi_awsize         (xdma_mem_axi_awsize),
      .m00_axi_awburst        (xdma_mem_axi_awburst),
      .m00_axi_awlock         (xdma_mem_axi_awlock),
      .m00_axi_awcac          (xdma_mem_axi_awcache),
      .m00_axi_awprot         (xdma_mem_axi_awprot),
      .m00_axi_awqos          (xdma_mem_axi_awqos),
      .m00_axi_awregion       (4'd0), 
      .m00_axi_awuser         (1'b0),
      .m00_axi_awvalid        (xdma_mem_axi_awvalid),
      .m00_axi_awready        (xdma_mem_axi_awready),
      .m00_axi_wdata          (xdma_mem_axi_wdata),
      .m00_axi_wstrb          (xdma_mem_axi_wstrb),
      .m00_axi_wlast          (xdma_mem_axi_wlast),
      .m00_axi_wuser          (1'b0), 
      .m00_axi_wvalid         (xdma_mem_axi_wvalid),
      .m00_axi_wready         (xdma_mem_axi_wready),
      .m00_axi_bid            (xdma_mem_axi_bid),
      .m00_axi_bresp          (xdma_mem_axi_bresp),
      .m00_axi_buser          (1'b0), 
      .m00_axi_bvalid         (xdma_mem_axi_bvalid),
      .m00_axi_bready         (xdma_mem_axi_bready),
      .m00_axi_arid           (xdma_mem_axi_arid),
      .m00_axi_araddr         (xdma_mem_axi_araddr),
      .m00_axi_arlen          (xdma_mem_axi_arlen),
      .m00_axi_arsize         (xdma_mem_axi_arsize),
      .m00_axi_arburst        (xdma_mem_axi_arburst),
      .m00_axi_arlock         (xdma_mem_axi_arlock),
      .m00_axi_arcache        (xdma_mem_axi_arcache),
      .m00_axi_arprot         (xdma_mem_axi_arprot),
      .m00_axi_arqos          (xdma_mem_axi_arqos),
      .m00_axi_arregion       (4'd0), 
      .m00_axi_aruser         (1'b0), 
      .m00_axi_arvalid        (xdma_mem_axi_arvalid),
      .m00_axi_arready        (xdma_mem_axi_arready),
      .m00_axi_rid            (xdma_mem_axi_rid),
      .m00_axi_rdata          (xdma_mem_axi_rdata),
      .m00_axi_rresp          (xdma_mem_axi_rresp),
      .m00_axi_rlast          (xdma_mem_axi_rlast),
      .m00_axi_ruser          (1'b0), 
      .m00_axi_rvalid         (xdma_mem_axi_rvalid),
      .m00_axi_rready         (xdma_mem_axi_rready),

      /*
      * AXI master interface for NodeSlot (Slave 01)
      */
      .m01_axi_awid           (nodeslot_axi_awid),
      .m01_axi_awaddr         (nodeslot_axi_awaddr),
      .m01_axi_awlen          (nodeslot_axi_awlen),
      .m01_axi_awsize         (nodeslot_axi_awsize),
      .m01_axi_awburst        (nodeslot_axi_awburst),
      .m01_axi_awlock         (nodeslot_axi_awlock),
      .m01_axi_awcache        (nodeslot_axi_awcache),
      .m01_axi_awprot         (nodeslot_axi_awprot),
      .m01_axi_awqos          (nodeslot_axi_awqos),
      .m01_axi_awregion       (4'd0), 
      .m01_axi_awuser         (1'b0), 
      .m01_axi_awvalid        (nodeslot_axi_awvalid),
      .m01_axi_awready        (nodeslot_axi_awready),
      .m01_axi_wdata          (nodeslot_axi_wdata),
      .m01_axi_wstrb          (nodeslot_axi_wstrb),
      .m01_axi_wlast          (nodeslot_axi_wlast),
      .m01_axi_wuser          (1'b0), 
      .m01_axi_wvalid         (nodeslot_axi_wvalid),
      .m01_axi_wready         (nodeslot_axi_wready),
      .m01_axi_bid            (nodeslot_axi_bid),
      .m01_axi_bresp          (nodeslot_axi_bresp),
      .m01_axi_buser          (1'b0), 
      .m01_axi_bvalid         (nodeslot_axi_bvalid),
      .m01_axi_bready         (nodeslot_axi_bready),

);


//AMPLE and XDMA DDR4 C0interconnect
axi_interconnect_wrap_2x1 #
    (
    .DATA_WIDTH(512),
    .ADDR_WIDTH(34),
    .STRB_WIDTH(DATA_WIDTH/8),
    .ID_WIDTH(4)
    ) xdma_write_interconnect_i
    (
      .clk                    (clk),
      .rst                    (rst),

      //Ample Memory Interface
      .s00_axi_awid           (ample_axi_awid),
      .s00_axi_awaddr         (ample_axi_awaddr),
      .s00_axi_awlen          (ample_axi_awlen),
      .s00_axi_awsize         (ample_axi_awsize),
      .s00_axi_awburst        (ample_axi_awburst),
      .s00_axi_awlock         (ample_axi_awlock),
      .s00_axi_awcache        (ample_axi_awcache),
      .s00_axi_awprot         (ample_axi_awprot),
      .s00_axi_awqos          (ample_axi_awqos),
      .s00_axi_awvalid        (ample_axi_awvalid),
      .s00_axi_awready        (ample_axi_awready),
      .s00_axi_wdata          (ample_axi_wdata),
      .s00_axi_wstrb          (ample_axi_wstrb),
      .s00_axi_wlast          (ample_axi_wlast),
      .s00_axi_wvalid         (ample_axi_wvalid),
      .s00_axi_wready         (ample_axi_wready),
      .s00_axi_bid            (ample_axi_bid),
      .s00_axi_bresp          (ample_axi_bresp),
      .s00_axi_bvalid         (ample_axi_bvalid),
      .s00_axi_bready         (ample_axi_bready),
      .s00_axi_arid           (ample_axi_arid),
      .s00_axi_araddr         (ample_axi_araddr),
      .s00_axi_arlen          (ample_axi_arlen),
      .s00_axi_arsize         (ample_axi_arsize),
      .s00_axi_arburst        (ample_axi_arburst),
      .s00_axi_arlock         (ample_axi_arlock),
      .s00_axi_arcache        (ample_axi_arcache),
      .s00_axi_arprot         (ample_axi_arprot),
      .s00_axi_arqos          (ample_axi_arqos),
      .s00_axi_arvalid        (ample_axi_arvalid),
      .s00_axi_arready        (ample_axi_arready),
      .s00_axi_rid            (ample_axi_rid),
      .s00_axi_rdata          (ample_axi_rdata),
      .s00_axi_rresp          (ample_axi_rresp),
      .s00_axi_rlast          (ample_axi_rlast),
      .s00_axi_rvalid         (ample_axi_rvalid),
      .s00_axi_rready         (ample_axi_rready),


      //XDMA Memory Interface
      .s01_axi_awid           (xdma_mem_axi_awid),
      .s01_axi_awaddr         (xdma_mem_axi_awaddr),
      .s01_axi_awlen          (xdma_mem_axi_awlen),
      .s01_axi_awsize         (xdma_mem_axi_awsize),
      .s01_axi_awburst        (xdma_mem_axi_awburst),
      .s01_axi_awlock         (xdma_mem_axi_awlock),
      .s01_axi_awcac          (xdma_mem_axi_awcache),
      .s01_axi_awprot         (xdma_mem_axi_awprot),
      .s01_axi_awqos          (xdma_mem_axi_awqos),
      .s01_axi_awregion       (4'd0), 
      .s01_axi_awuser         (1'b0),
      .s01_axi_awvalid        (xdma_mem_axi_awvalid),
      .s01_axi_awready        (xdma_mem_axi_awready),
      .s01_axi_wdata          (xdma_mem_axi_wdata),
      .s01_axi_wstrb          (xdma_mem_axi_wstrb),
      .s01_axi_wlast          (xdma_mem_axi_wlast),
      .s01_axi_wuser          (1'b0), 
      .s01_axi_wvalid         (xdma_mem_axi_wvalid),
      .s01_axi_wready         (xdma_mem_axi_wready),
      .s01_axi_bid            (xdma_mem_axi_bid),
      .s01_axi_bresp          (xdma_mem_axi_bresp),
      .s01_axi_buser          (1'b0), 
      .s01_axi_bvalid         (xdma_mem_axi_bvalid),
      .s01_axi_bready         (xdma_mem_axi_bready),
      .s01_axi_arid           (xdma_mem_axi_arid),
      .s01_axi_araddr         (xdma_mem_axi_araddr),
      .s01_axi_arlen          (xdma_mem_axi_arlen),
      .s01_axi_arsize         (xdma_mem_axi_arsize),
      .s01_axi_arburst        (xdma_mem_axi_arburst),
      .s01_axi_arlock         (xdma_mem_axi_arlock),
      .s01_axi_arcache        (xdma_mem_axi_arcache),
      .s01_axi_arprot         (xdma_mem_axi_arprot),
      .s01_axi_arqos          (xdma_mem_axi_arqos),
      .s01_axi_arregion       (4'd0), 
      .s01_axi_aruser         (1'b0), 
      .s01_axi_arvalid        (xdma_mem_axi_arvalid),
      .s01_axi_arready        (xdma_mem_axi_arready),
      .s01_axi_rid            (xdma_mem_axi_rid),
      .s01_axi_rdata          (xdma_mem_axi_rdata),
      .s01_axi_rresp          (xdma_mem_axi_rresp),
      .s01_axi_rlast          (xdma_mem_axi_rlast),
      .s01_axi_ruser          (1'b0), 
      .s01_axi_rvalid         (xdma_mem_axi_rvalid),
      .s01_axi_rready         (xdma_mem_axi_rready),


      .m00_axi_awid           (c0_ddr4_s_axi_awid),
      .m00_axi_awaddr         (c0_ddr4_s_axi_awaddr),
      .m00_axi_awlen          (c0_ddr4_s_axi_awlen),
      .m00_axi_awsize         (c0_ddr4_s_axi_awsize),
      .m00_axi_awburst        (c0_ddr4_s_axi_awburst),
      .m00_axi_awlock         (c0_ddr4_s_axi_awlock),
      .m00_axi_awcache        (c0_ddr4_s_axi_awcache),
      .m00_axi_awprot         (c0_ddr4_s_axi_awprot),
      .m00_axi_awqos          (c0_ddr4_s_axi_awqos),
      .m00_axi_awregion       (4'd0),  
      .m00_axi_awuser         (1'b0),  
      .m00_axi_awvalid        (c0_ddr4_s_axi_awvalid),
      .m00_axi_awready        (c0_ddr4_s_axi_awready),
      .m00_axi_wdata          (c0_ddr4_s_axi_wdata),
      .m00_axi_wstrb          (c0_ddr4_s_axi_wstrb),
      .m00_axi_wlast          (c0_ddr4_s_axi_wlast),
      .m00_axi_wuser          (1'b0), 
      .m00_axi_wvalid         (c0_ddr4_s_axi_wvalid),
      .m00_axi_wready         (c0_ddr4_s_axi_wready),
      .m00_axi_bid            (c0_ddr4_s_axi_bid),
      .m00_axi_bresp          (c0_ddr4_s_axi_bresp),
      .m00_axi_buser          (1'b0),  
      .m00_axi_bvalid         (c0_ddr4_s_axi_bvalid),
      .m00_axi_bready         (c0_ddr4_s_axi_bready),
      .m00_axi_arid           (c0_ddr4_s_axi_arid),
      .m00_axi_araddr         (c0_ddr4_s_axi_araddr),
      .m00_axi_arlen          (c0_ddr4_s_axi_arlen),
      .m00_axi_arsize         (c0_ddr4_s_axi_arsize),
      .m00_axi_arburst        (c0_ddr4_s_axi_arburst),
      .m00_axi_arlock         (c0_ddr4_s_axi_arlock),
      .m00_axi_arcache        (c0_ddr4_s_axi_arcache),
      .m00_axi_arprot         (c0_ddr4_s_axi_arprot),
      .m00_axi_arqos          (c0_ddr4_s_axi_arqos),
      .m00_axi_arregion       (4'd0),  
      .m00_axi_aruser         (1'b0),  
      .m00_axi_arvalid        (c0_ddr4_s_axi_arvalid),
      .m00_axi_arready        (c0_ddr4_s_axi_arready),
      .m00_axi_rid            (c0_ddr4_s_axi_rid),
      .m00_axi_rdata          (c0_ddr4_s_axi_rdata),
      .m00_axi_rresp          (c0_ddr4_s_axi_rresp),
      .m00_axi_rlast          (c0_ddr4_s_axi_rlast),
      .m00_axi_ruser          (1'b0),  
      .m00_axi_rvalid         (c0_ddr4_s_axi_rvalid),
      .m00_axi_rready         (c0_ddr4_s_axi_rready)

);


//DDR4 Controllers

`ifdef SYNTHESIS
    // ====================================================================================
    // DDR4 Controller
    // ====================================================================================

    //Feature,weight and output memory

    ddr4_0 u_ddr4_0
    (
        .sys_rst                             (sys_rst),

        .c0_sys_clk_p                        (c0_sys_clk_p),
        .c0_sys_clk_n                        (c0_sys_clk_n),
        .c0_init_calib_complete              (c0_init_calib_complete),
        .c0_ddr4_act_n                       (c0_ddr4_act_n),
        .c0_ddr4_adr                         (c0_ddr4_adr),
        .c0_ddr4_ba                          (c0_ddr4_ba),
        .c0_ddr4_bg                          (c0_ddr4_bg),
        .c0_ddr4_cke                         (c0_ddr4_cke),
        .c0_ddr4_odt                         (c0_ddr4_odt),
        .c0_ddr4_cs_n                        (c0_ddr4_cs_n),
        .c0_ddr4_ck_t                        (c0_ddr4_ck_t),
        .c0_ddr4_ck_c                        (c0_ddr4_ck_c),
        .c0_ddr4_reset_n                     (c0_ddr4_reset_n_int),

        .c0_ddr4_parity                      (c0_ddr4_parity),
        .c0_ddr4_dq                          (c0_ddr4_dq),
        .c0_ddr4_dqs_c                       (c0_ddr4_dqs_c),
        .c0_ddr4_dqs_t                       (c0_ddr4_dqs_t),

        .c0_ddr4_ui_clk                      (c0_ddr4_clk),
        .c0_ddr4_ui_clk_sync_rst             (c0_ddr4_rst),
        .addn_ui_clkout1                     (),
        .dbg_clk                             (dbg_clk_0),

        // AXI CTRL port
        .c0_ddr4_s_axi_ctrl_awvalid          (1'b0),
        .c0_ddr4_s_axi_ctrl_awready          (),
        .c0_ddr4_s_axi_ctrl_awaddr           (32'b0),
        // Slave Interface Write Data Ports
        .c0_ddr4_s_axi_ctrl_wvalid           (1'b0),
        .c0_ddr4_s_axi_ctrl_wready           (),
        .c0_ddr4_s_axi_ctrl_wdata            (32'b0),
        // Slave Interface Write Response Ports
        .c0_ddr4_s_axi_ctrl_bvalid           (),
        .c0_ddr4_s_axi_ctrl_bready           (1'b1),
        .c0_ddr4_s_axi_ctrl_bresp            (),
        // Slave Interface Read Address Ports
        .c0_ddr4_s_axi_ctrl_arvalid          (1'b0),
        .c0_ddr4_s_axi_ctrl_arready          (),
        .c0_ddr4_s_axi_ctrl_araddr           (32'b0),
        // Slave Interface Read Data Ports
        .c0_ddr4_s_axi_ctrl_rvalid           (),
        .c0_ddr4_s_axi_ctrl_rready           (1'b1),
        .c0_ddr4_s_axi_ctrl_rdata            (),
        .c0_ddr4_s_axi_ctrl_rresp            (),


        // Interrupt output
        .c0_ddr4_interrupt                   (),

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

    //Nodeslot memory
    ddr4_1 u_ddr4_1 (
        .sys_rst                             (sys_rst),

        .c0_sys_clk_p                        (c1_sys_clk_p),
        .c0_sys_clk_n                        (c1_sys_clk_n),
        .c0_init_calib_complete              (c1_init_calib_complete),
        .c0_ddr4_act_n                       (c1_ddr4_act_n),
        .c0_ddr4_adr                         (c1_ddr4_adr),
        .c0_ddr4_ba                          (c1_ddr4_ba),
        .c0_ddr4_bg                          (c1_ddr4_bg),
        .c0_ddr4_cke                         (c1_ddr4_cke),
        .c0_ddr4_odt                         (c1_ddr4_odt),
        .c0_ddr4_cs_n                        (c1_ddr4_cs_n),
        .c0_ddr4_ck_t                        (c1_ddr4_ck_t),
        .c0_ddr4_ck_c                        (c1_ddr4_ck_c),
        .c0_ddr4_reset_n                     (c1_ddr4_reset_n_int),

        .c0_ddr4_parity                      (c1_ddr4_parity),
        .c0_ddr4_dq                          (c1_ddr4_dq),
        .c0_ddr4_dqs_c                       (c1_ddr4_dqs_c),
        .c0_ddr4_dqs_t                       (c1_ddr4_dqs_t),

        .c0_ddr4_ui_clk                      (c1_ddr4_clk),
        .c0_ddr4_ui_clk_sync_rst             (c1_ddr4_rst),
        .addn_ui_clkout1                     (),
        .dbg_clk                             (dbg_clk_1),

        // AXI CTRL port
        .c0_ddr4_s_axi_ctrl_awvalid          (1'b0),
        .c0_ddr4_s_axi_ctrl_awready          (),
        .c0_ddr4_s_axi_ctrl_awaddr           (32'b0),
        .c0_ddr4_s_axi_ctrl_wvalid           (1'b0),
        .c0_ddr4_s_axi_ctrl_wready           (),
        .c0_ddr4_s_axi_ctrl_wdata            (32'b0),
        .c0_ddr4_s_axi_ctrl_bvalid           (),
        .c0_ddr4_s_axi_ctrl_bready           (1'b1),
        .c0_ddr4_s_axi_ctrl_bresp            (),
        .c0_ddr4_s_axi_ctrl_arvalid          (1'b0),
        .c0_ddr4_s_axi_ctrl_arready          (),
        .c0_ddr4_s_axi_ctrl_araddr           (32'b0),
        .c0_ddr4_s_axi_ctrl_rvalid           (),
        .c0_ddr4_s_axi_ctrl_rready           (1'b1),
        .c0_ddr4_s_axi_ctrl_rdata            (),
        .c0_ddr4_s_axi_ctrl_rresp            (),

        // Interrupt output
        .c0_ddr4_interrupt                   (),

        // Slave Interface AXI ports (Connect NodeSlot AXI to DDR4)
        .c0_ddr4_aresetn                     (c1_ddr4_aresetn),
        .c0_ddr4_s_axi_awid                  (nodeslot_axi_awid),
        .c0_ddr4_s_axi_awaddr                (nodeslot_axi_awaddr),
        .c0_ddr4_s_axi_awlen                 (nodeslot_axi_awlen),
        .c0_ddr4_s_axi_awsize                (nodeslot_axi_awsize),
        .c0_ddr4_s_axi_awburst               (nodeslot_axi_awburst),
        .c0_ddr4_s_axi_awlock                (nodeslot_axi_awlock),
        .c0_ddr4_s_axi_awcache               (nodeslot_axi_awcache),
        .c0_ddr4_s_axi_awprot                (nodeslot_axi_awprot),
        .c0_ddr4_s_axi_awqos                 (nodeslot_axi_awqos),
        .c0_ddr4_s_axi_awvalid               (nodeslot_axi_awvalid),
        .c0_ddr4_s_axi_awready               (nodeslot_axi_awready),
        .c0_ddr4_s_axi_wdata                 (nodeslot_axi_wdata),
        .c0_ddr4_s_axi_wstrb                 (nodeslot_axi_wstrb),
        .c0_ddr4_s_axi_wlast                 (nodeslot_axi_wlast),
        .c0_ddr4_s_axi_wvalid                (nodeslot_axi_wvalid),
        .c0_ddr4_s_axi_wready                (nodeslot_axi_wready),
        .c0_ddr4_s_axi_bid                   (nodeslot_axi_bid),
        .c0_ddr4_s_axi_bresp                 (nodeslot_axi_bresp),
        .c0_ddr4_s_axi_bvalid                (nodeslot_axi_bvalid),
        .c0_ddr4_s_axi_bready                (nodeslot_axi_bready),
        .c0_ddr4_s_axi_arid                  (nodeslot_axi_arid),
        .c0_ddr4_s_axi_araddr                (nodeslot_axi_araddr),
        .c0_ddr4_s_axi_arlen                 (nodeslot_axi_arlen),
        .c0_ddr4_s_axi_arsize                (nodeslot_axi_arsize),
        .c0_ddr4_s_axi_arburst               (nodeslot_axi_arburst),
        .c0_ddr4_s_axi_arlock                (nodeslot_axi_arlock),
        .c0_ddr4_s_axi_arcache               (nodeslot_axi_arcache),
        .c0_ddr4_s_axi_arprot                (nodeslot_axi_arprot),
        .c0_ddr4_s_axi_arqos                 (nodeslot_axi_arqos),
        .c0_ddr4_s_axi_arvalid               (nodeslot_axi_arvalid),
        .c0_ddr4_s_axi_arready               (nodeslot_axi_arready),
        .c0_ddr4_s_axi_rid                   (nodeslot_axi_rid),
        .c0_ddr4_s_axi_rdata                 (nodeslot_axi_rdata),
        .c0_ddr4_s_axi_rresp                 (nodeslot_axi_rresp),
        .c0_ddr4_s_axi_rlast                 (nodeslot_axi_rlast),
        .c0_ddr4_s_axi_rvalid                (nodeslot_axi_rvalid),
        .c0_ddr4_s_axi_rready                (nodeslot_axi_rready),

        .dbg_bus                            (dbg_bus_1)
    );


    always @(posedge c0_ddr4_clk) begin
      c0_ddr4_aresetn <= ~c0_ddr4_rst;
    end

    always @(posedge c1_ddr4_clk) begin
      c1_ddr4_aresetn <= ~c1_ddr4_rst;
    end


//RAM SIM
`else

    // ==============================Simulated Memory=========================================
    // DRAM model for feature memory
    axi_ram #(
        .DATA_WIDTH(512),
        .ADDR_WIDTH(34),
        .ID_WIDTH(8)
    ) dram_c0_sim (
        .clk                    (sys_clk),
        .rst                    (sys_rst),
        .s_axi_awid             (c0_ddr4_s_axi_awid),
        .s_axi_awaddr           (c0_ddr4_s_axi_awaddr),
        .s_axi_awlen            (c0_ddr4_s_axi_awlen),
        .s_axi_awsize           (c0_ddr4_s_axi_awsize),
        .s_axi_awburst          (c0_ddr4_s_axi_awburst),
        .s_axi_awlock           (c0_ddr4_s_axi_awlock),
        .s_axi_awcache          (c0_ddr4_s_axi_awcache),
        .s_axi_awprot           (c0_ddr4_s_axi_awprot),
        .s_axi_awvalid          (c0_ddr4_s_axi_awvalid),
        .s_axi_awready          (c0_ddr4_s_axi_awready),
        .s_axi_wdata            (c0_ddr4_s_axi_wdata),
        .s_axi_wstrb            (c0_ddr4_s_axi_wstrb),
        .s_axi_wlast            (c0_ddr4_s_axi_wlast),
        .s_axi_wvalid           (c0_ddr4_s_axi_wvalid),
        .s_axi_wready           (c0_ddr4_s_axi_wready),
        .s_axi_bid              (c0_ddr4_s_axi_bid),
        .s_axi_bresp            (c0_ddr4_s_axi_bresp),
        .s_axi_bvalid           (c0_ddr4_s_axi_bvalid),
        .s_axi_bready           (c0_ddr4_s_axi_bready),

        .s_axi_arid             (c0_ddr4_s_axi_arid),
        .s_axi_araddr           (c0_ddr4_s_axi_araddr),
        .s_axi_arlen            (c0_ddr4_s_axi_arlen),
        .s_axi_arsize           (c0_ddr4_s_axi_arsize),
        .s_axi_arburst          (c0_ddr4_s_axi_arburst),
        .s_axi_arlock           (c0_ddr4_s_axi_arlock),
        .s_axi_arcache          (c0_ddr4_s_axi_arcache),
        .s_axi_arprot           (c0_ddr4_s_axi_arprot),
        .s_axi_arvalid          (c0_ddr4_s_axi_arvalid),
        .s_axi_arready          (c0_ddr4_s_axi_arready),
        .s_axi_rid              (c0_ddr4_s_axi_rid),
        .s_axi_rdata            (c0_ddr4_s_axi_rdata),
        .s_axi_rresp            (c0_ddr4_s_axi_rresp),
        .s_axi_rlast            (c0_ddr4_s_axi_rlast),
        .s_axi_rvalid           (c0_ddr4_s_axi_rvalid),
        .s_axi_rready           (c0_ddr4_s_axi_rready)
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


`endif

endmodule