import argparse
import logging
import os
import pprint
import shutil
import time
from tqdm import tqdm
import torch
import torch.optim
from torch.cuda.amp import autocast, GradScaler
import yaml
from datasets.data_builder import build_dataloader
from easydict import EasyDict
from models.model_helper import ModelHelper
from tensorboardX import SummaryWriter
from utils.criterion_helper import build_criterion
from utils.eval_helper import dump, log_metrics, merge_together, performances
from utils.lr_helper import get_scheduler
from utils.misc_helper import AverageMeter, create_logger, get_current_time, load_state, save_checkpoint, set_random_seed, update_config
from utils.optimizer_helper import get_optimizer
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="UniAD Framework")
parser.add_argument("--config", default="./train_config.yaml")
parser.add_argument("-e", "--evaluate", action="store_true")

def main():
    global args, config, key_metric, best_metric
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.port = config.get("port", None)
    rank = 0
    config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    config.evaluator.eval_dir = os.path.join(config.exp_path, config.evaluator.save_dir)
    config = update_config(config)

    if rank == 0:
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)

        current_time = get_current_time()
        tb_logger = SummaryWriter(config.log_path + "/events_dec/" + current_time)
        logger = create_logger(
            "global_logger", config.log_path + "/dec_{}.log".format(current_time)
        )
        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))
    else:
        tb_logger = None

    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)

    # create model
    model = ModelHelper(config.net)
    model.cuda()

    layers = []
    for module in config.net:
        layers.append(module["name"])
    frozen_layers = config.get("frozen_layers", [])
    active_layers = list(set(layers) ^ set(frozen_layers))
    if rank == 0:
        logger.info("layers: {}".format(layers))
        logger.info("active layers: {}".format(active_layers))

    optimizer = get_optimizer(model.parameters(), config.trainer.optimizer)
    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)
    scaler = GradScaler()

    key_metric = config.evaluator["key_metric"]
    best_metric = 0
    last_epoch = 0

    auto_resume = config.saver.get("auto_resume", True)
    resume_model = config.saver.get("resume_model", None)
    load_path = config.saver.get("load_path", None)

    if resume_model and not resume_model.startswith("/"):
        resume_model = os.path.join(config.exp_path, resume_model)
    lastest_model = os.path.join(config.save_path, "ckpt_best.pth.tar")
    if auto_resume and os.path.exists(lastest_model):
        resume_model = lastest_model
    if resume_model:
        best_metric, last_epoch = load_state(resume_model, model, optimizer=optimizer)
    elif load_path:
        if not load_path.startswith("/"):
            load_path = os.path.join(config.exp_path, load_path)
        load_state(load_path, model)

    train_loader, val_loader = build_dataloader(config.dataset, distributed=False)

    if args.evaluate:
        validate(val_loader, model)
        tb_logger.close()
        return

    criterion = build_criterion(config.criterion)

    for epoch in range(last_epoch, config.trainer.max_epoch):
        last_iter = epoch * len(train_loader)
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            lr_scheduler,
            epoch,
            last_iter,
            tb_logger,
            criterion,
            frozen_layers,
            scaler=scaler,
        )
        lr_scheduler.step()

        if (epoch + 1) % config.trainer.val_freq_epoch == 0:
            ret_metrics = validate(val_loader, model)
            if rank == 0:
                ret_key_metric = ret_metrics[key_metric]
                is_best = ret_key_metric >= best_metric
                best_metric = max(ret_key_metric, best_metric)
                logger.info(f"best_metric = {best_metric}")
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": config.net,
                        "state_dict": model.state_dict(),
                        "best_metric": best_metric,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best,
                    config,
                )


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    start_iter,
    tb_logger,
    criterion,
    frozen_layers,
    scaler=None,
):

    batch_time = AverageMeter(config.trainer.print_freq_step)
    data_time = AverageMeter(config.trainer.print_freq_step)
    losses = AverageMeter(config.trainer.print_freq_step)

    model.train()
    for layer in frozen_layers:
        module = getattr(model, layer)
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    world_size = 1
    rank = 0
    logger = logging.getLogger("global_logger")
    end = time.time()
    
    if scaler is None:
        scaler = GradScaler()

    for i, input in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.trainer.max_epoch}", ncols=100)):
        curr_step = start_iter + i
        current_lr = lr_scheduler.get_last_lr()[0]

        data_time.update(time.time() - end)
        
        # # plt show 8x1 grid of images
        # for i in range(8):
        #     plt.subplot(1, 8, i + 1)
        #     plt.imshow(input["image"][i].permute(1, 2, 0).numpy())
        #     plt.title(f"Label: {input['label'][i]}")
        # plt.show()

        with autocast():
            outputs = model(input)
            loss = 0
            for name, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                loss += weight * criterion_loss(outputs)
                
        reduced_loss = loss.clone()
        reduced_loss = reduced_loss / world_size
        losses.update(reduced_loss.item())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # update
        if config.trainer.get("clip_max_norm", None):
            max_norm = config.trainer.clip_max_norm
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
        scaler.step(optimizer)
        scaler.update()
        batch_time.update(time.time() - end)

        if (curr_step + 1) % config.trainer.print_freq_step == 0 and rank == 0:
            tb_logger.add_scalar("loss_train", losses.avg, curr_step + 1)
            tb_logger.add_scalar("lr", current_lr, curr_step + 1)
            tb_logger.flush()

            logger.info(
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                "LR {lr:.5f}\t".format(
                    epoch + 1,
                    config.trainer.max_epoch,
                    curr_step + 1,
                    len(train_loader) * config.trainer.max_epoch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=current_lr,
                )
            )

        end = time.time()
    print(f"Loss: {losses.avg} at epoch {epoch+1}")


def validate(val_loader, model):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)

    model.eval()
    rank = 0 # dist.get_rank()
    logger = logging.getLogger("global_logger")
    criterion = build_criterion(config.criterion)
    end = time.time()

    if rank == 0:
        os.makedirs(config.evaluator.eval_dir, exist_ok=True)

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            outputs = model(input)
            dump(config.evaluator.eval_dir, outputs)

            loss = 0
            for name, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                loss += weight * criterion_loss(outputs)
            num = len(outputs["filename"])
            losses.update(loss.item(), num)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % config.trainer.print_freq_step == 0 and rank == 0:
                logger.info(
                    "Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        i + 1, len(val_loader), batch_time=batch_time
                    )
                )


    total_num = torch.Tensor([losses.count]).cuda()
    loss_sum = torch.Tensor([losses.avg * losses.count]).cuda()
    final_loss = loss_sum.item() / total_num.item()

    ret_metrics = {} 
    if rank == 0:
        logger.info("Gathering final results ...")
        # total loss
        logger.info(" * Loss {:.5f}\ttotal_num={}".format(final_loss, total_num.item()))
        fileinfos, preds, masks = merge_together(config.evaluator.eval_dir)
        shutil.rmtree(config.evaluator.eval_dir)
        # evaluate, log & vis
        ret_metrics = performances(fileinfos, preds, masks, config.evaluator.metrics)
        log_metrics(ret_metrics, config.evaluator.metrics)
        
    model.train()
    return ret_metrics


if __name__ == "__main__":
    main()
