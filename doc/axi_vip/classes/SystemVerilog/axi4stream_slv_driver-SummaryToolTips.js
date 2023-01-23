NDSummary.OnToolTipsLoaded("SystemVerilogClass:axi4stream_slv_driver",{301:"<div class=\"NDToolTip TClass LSystemVerilog\"><div class=\"TTSummary\">AXI4STREAM Slave Driver Class. It receives TREADY transaction from the user enviroment and drives the TREADY signal if HAS_TREADY of the VIP is on, else TREADY is set to high all the time.</div></div>",303:"<div class=\"NDToolTip TInformation LSystemVerilog\"><div class=\"TTSummary\">axi4stream_vif_proxy `XIL_AXI4STREAM_PARAM_ORDER&nbsp; vif_proxy; AXI4STREAM VIF Proxy Class.</div></div>",305:"<div class=\"NDToolTip TFunction LSystemVerilog\"><div id=\"NDPrototype305\" class=\"NDPrototype WideForm CStyle\"><table><tr><td class=\"PBeforeParameters\"><span class=\"SHKeyword\">function new</span>(</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"PModifierQualifier first\">input&nbsp;</td><td class=\"PType\"><span class=\"SHKeyword\">string</span>&nbsp;</td><td class=\"PName\">name&nbsp;</td><td class=\"PDefaultValueSeparator\">=&nbsp;</td><td class=\"PDefaultValue last\"><span class=\"SHString\">&quot;unnamed_axi4stream_slv_driver&quot;</span></td></tr></table></td><td class=\"PAfterParameters\">);</td></tr></table></div><div class=\"TTSummary\">Constructor to create a new axi4stream slave driver object,~name~ is the name of the instance.</div></div>",306:"<div class=\"NDToolTip TFunction LSystemVerilog\"><div id=\"NDPrototype306\" class=\"NDPrototype WideForm CStyle\"><table><tr><td class=\"PBeforeParameters\"><span class=\"SHKeyword\">function void</span> set_vif(</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"PModifierQualifier first\">axi4stream_vif_proxy `</td><td class=\"PType\">XIL_AXI4STREAM_PARAM_ORDER&nbsp;</td><td class=\"PName last\">vif</td></tr></table></td><td class=\"PAfterParameters\">);</td></tr></table></div><div class=\"TTSummary\">Assigns the virtual interface of the driver.</div></div>",307:"<div class=\"NDToolTip TFunction LSystemVerilog\"><div id=\"NDPrototype307\" class=\"NDPrototype NoParameterForm\"><span class=\"SHKeyword\">virtual</span> task run_phase();</div><div class=\"TTSummary\">Start control processes for operation of axi4stream_slv_driver.</div></div>",308:"<div class=\"NDToolTip TFunction LSystemVerilog\"><div id=\"NDPrototype308\" class=\"NDPrototype NoParameterForm\"><span class=\"SHKeyword\">virtual</span> task stop_phase();</div><div class=\"TTSummary\">Stops all control processes of axi4stream_slv_driver.</div></div>",309:"<div class=\"NDToolTip TFunction LSystemVerilog\"><div id=\"NDPrototype309\" class=\"NDPrototype WideForm CStyle\"><table><tr><td class=\"PBeforeParameters\">task send_tready(</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"PModifierQualifier first\">input&nbsp;</td><td class=\"PType\">axi4stream_ready_gen&nbsp;</td><td class=\"PName last\">t</td></tr></table></td><td class=\"PAfterParameters\">);</td></tr></table></div><div class=\"TTSummary\">Send ready object to the driver when HAS_TREADY is on.</div></div>",310:"<div class=\"NDToolTip TFunction LSystemVerilog\"><div id=\"NDPrototype310\" class=\"NDPrototype WideForm CStyle\"><table><tr><td class=\"PBeforeParameters\"><span class=\"SHKeyword\">virtual function</span> axi4stream_ready_gen create_ready (</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"PType first\"><span class=\"SHKeyword\">string</span>&nbsp;</td><td class=\"PName\">name&nbsp;</td><td class=\"PDefaultValueSeparator\">=&nbsp;</td><td class=\"PDefaultValue last\"><span class=\"SHString\">&quot;unnamed_ready&quot;</span></td></tr></table></td><td class=\"PAfterParameters\">);</td></tr></table></div><div class=\"TTSummary\">Returns Ready class that has been &quot;newed&quot;.</div></div>"});