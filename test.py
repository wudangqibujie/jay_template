# from torch.utils.tensorboard import SummaryWriter
#
#
# writer = SummaryWriter(log_dir="log_file", comment="LR_0.1_BATCH_16")
# for epoch in range(100):
#     writer.add_scalar('mAP', epoch * 0.1, epoch)
#     writer.add_scalar('loss/loss1', epoch * 0.12, epoch)
#     writer.add_scalar('loss/loss2', epoch * 0.31, epoch)
#     writer.add_scalar('loss/loss3', epoch * 0.14, epoch)

# import logging
# import logging.config as logging_config
# from utils import read_json
# config = read_json("logger/logger_config.json")
# logging_config.dictConfig(config)
#
# logger = logging.getLogger("train")
# logger.setLevel(logging.INFO)
# logger.info("test")
# logger.info('    {:15s}: {}'.format(str("BMW"), 0.12))


with open('./config/ocr_config/char_std_5990.txt', encoding="utf-8") as f:
    data = f.readlines()
    alphabet = [x.rstrip() for x in data]
    alphabet = ''.join(alphabet)
print(alphabet)