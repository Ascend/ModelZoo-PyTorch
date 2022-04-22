# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

from apex import amp
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.config import cfg


class GetFPS(object):
    def __init__(self, max_len):
        self.max_len = max_len
        self.all_fps = []
        self.mean_fps = 0

    def update(self, batch_time):
        self.all_fps.append(1 / batch_time)
        if len(self.all_fps) > self.max_len:
            del self.all_fps[0]
        self.mean_fps = self.get_mean(self.all_fps)

    def get_mean(self, all_fps):
        all_fps = sorted(all_fps)
        length = len(all_fps)
        if length <= 3:
            avg = sum(all_fps) / len(all_fps)

            return avg
        else:
            del_num = max(1, int(length / 10))
            cut = all_fps[del_num:length - del_num]
            avg = sum(cut) / len(cut)

            return avg


def do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        isAMP,
        local_rank
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    batch_size = cfg.SOLVER.IMS_PER_BATCH
    meters = MetricLogger(bs=batch_size, delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    get_fps = GetFPS(500)
    iteration = 0
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):

        if local_rank == 0:
            print('=====iter%d' % iteration)

        data_time = time.time() - end
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device, non_blocking=True)
        targets = [target.to(device, non_blocking=True) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        meters.update(loss=losses, **loss_dict)

        optimizer.zero_grad()
        if isAMP:
            with amp.scale_loss(losses, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        get_fps.update(batch_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if local_rank == 0:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    # memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    memory=0
                )
            )
        if iteration % checkpoint_period == 0 and iteration > 0:
            checkpointer.save("model_%07d" % (iteration + 1), **arguments)
    checkpointer.save("model_%07d" % iteration, **arguments)
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    if local_rank == 0:
        logger.info("Total training time: %s" % total_time_str)
        logger.info("step_fps: %.4f" % get_fps.mean_fps)
