NDContentPage.OnToolTipsLoaded({133:"<div class=\"NDToolTip TClass LSystemVerilog\"><div class=\"TTSummary\">The xil_seq_item_pull_port#(REQ,RSP) class is extends from xil_sqr_if_base. It inherits all these variables and functions of xil_sqr_if_base.</div></div>",367:"<div class=\"NDToolTip TType LSystemVerilog\"><div class=\"TTSummary\">Xilinx AXI VIP Interger unsigned data type</div></div>",388:"<div class=\"NDToolTip TType LSystemVerilog\"><div class=\"TTSummary\">This policy type informs the driver if the driver can re-order transactions that it is still processing.&nbsp; For READ transactions, the driver can return the RDATA beats from different RID\'s in a different order than they were received.&nbsp; For WRITE transactions, the driver uses this policy to determine if it can return BRESP\'s in a different order than they were received.</div></div>",516:"<div class=\"NDToolTip TClass LSystemVerilog\"><div class=\"TTSummary\">The axi_transaction class is the base class of AXI protocol. It inherits all the methods of xil_sequence_item.</div></div>",631:"<div class=\"NDToolTip TClass LSystemVerilog\"><div class=\"TTSummary\">AXI VIF Proxy Class. It has virtual interface for AXI VIP interface.</div></div>",747:"<div class=\"NDToolTip TClass LSystemVerilog\"><div class=\"TTSummary\">AXI Ready generation class.</div></div>",760:"<div class=\"NDToolTip TFunction LSystemVerilog\"><div id=\"NDPrototype760\" class=\"NDPrototype WideForm CStyle\"><table><tr><td class=\"PBeforeParameters\"><span class=\"SHKeyword\">function void</span> set_arready_gen(</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"PModifierQualifier first\">input&nbsp;</td><td class=\"PType\">axi_ready_gen_t&nbsp;</td><td class=\"PName last\">new_method</td></tr></table></td><td class=\"PAfterParameters\">);</td></tr></table></div><div class=\"TTSummary\">Sets arready of the AXI slave read driver. There are three ways for arready generation in AXI slave read driver.</div></div>",763:"<div class=\"NDToolTip TFunction LSystemVerilog\"><div id=\"NDPrototype763\" class=\"NDPrototype WideForm CStyle\"><table><tr><td class=\"PBeforeParameters\">task send_arready(</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"PModifierQualifier first\">input&nbsp;</td><td class=\"PType\">axi_ready_gen&nbsp;</td><td class=\"PName last\">t</td></tr></table></td><td class=\"PAfterParameters\">);</td></tr></table></div><div class=\"TTSummary\">Sends the ready structure to the slave read driver for controlling the ARREADY channel. This is blocking process which will not return till this ready is being sent out.</div></div>",1011:"<div class=\"NDToolTip TFunction LSystemVerilog\"><div id=\"NDPrototype1011\" class=\"NDPrototype WideForm CStyle\"><table><tr><td class=\"PBeforeParameters\"><span class=\"SHKeyword\">function void</span> set_forward_progress_timeout_value (</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"PModifierQualifier first\">input&nbsp;</td><td class=\"PType\">xil_axi_uint&nbsp;</td><td class=\"PName last\">new_timeout</td></tr></table></td><td class=\"PAfterParameters\">);</td></tr></table></div><div class=\"TTSummary\">Sets the number of cycles that the driver will wait until it will flag a watch dog error of the axi_slv_rd_driver. Default value is 50000. Setting this to a very large value will cause a hung simulation to continue for a longer time.&nbsp; Setting this to a very small number may not allow enough time for simulation to respond.</div></div>",1017:"<div class=\"NDToolTip TClass LSystemVerilog\"><div class=\"TTSummary\">AXI Slave Read Driver Class. It does below: Receives AR Command from the interface and then passes that command to the user environment. The user will then create a READ transaction and pass it back to the driver to drive the R channel.</div></div>"});