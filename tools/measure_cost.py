import torch
from fvcore.nn import FlopCountAnalysis
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer

cfg = get_cfg(); cfg.merge_from_file("/root/project/SeCap-AGPReID-main/logs/CARGO/DCA-NUM-36/config.yaml"); cfg.freeze()
model = DefaultTrainer.build_model(cfg).eval().cuda()

dummy = torch.randn(1, 3, 256, 128).cuda()
bb = model.backbone
flops = FlopCountAnalysis(bb, dummy)
print("GFLOPs:", flops.total()/1e9)
# 关键: 打印 by_module, 确认真的算到了 blocks/attention, 不是只算了个别层
print(flops.by_module_and_operator())