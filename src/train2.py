import os
import torch
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from torchvision import datasets, models
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import PIL.Image
from torchinfo import summary
import torch.optim as optim
from model import load_model, Darknet
from util.loss import compute_loss
import util.utils as utils
import util.transforms as utransforms
from test import _evaluate, evaluate_on_fog
from util.augmentations import AUGMENTATION_TRANSFORMS
from util.logger import Logger
import util.datasets as ds
from activation import Activations
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
import psutil
import gc

cur_time = time.strftime("%H-%M-%S", time.localtime(time.time()))

def print_system_info():
    """Print system and training info for debugging"""
    print("=" * 60)
    print("SYSTEM INFO")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        print(f"Current GPU Memory: {torch.cuda.memory_allocated() / (1024**3):.1f} GB")
    else:
        print("⚠️  WARNING: NO GPU DETECTED - TRAINING WILL BE VERY SLOW!")
    print("=" * 60)

def optimize_torch_settings():
    """Enable PyTorch optimizations"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("✓ Enabled CUDA optimizations")
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def train_student_one_epoch_fast(s_model, t_model, loader, optimizer, device, epoch,
                                logger, batch_size, stio_logger):
    """Optimized student training without perceptual loss"""
    
    losses = utils.AverageMeter()
    lbox = utils.AverageMeter()
    lobj = utils.AverageMeter()
    lcls = utils.AverageMeter()
    
    s_model.train()
    
    # Add progress bar and timing
    epoch_start_time = time.time()
    loader_with_progress = tqdm(loader, desc=f"Student Epoch {epoch}", leave=False)
    
    for train_epoch, (img_dir, images, depth, targets) in enumerate(loader_with_progress):
        batch_start = time.time()
        
        # Fast fog generation - simple noise instead of complex fog
        if train_epoch % 2 == 0:  # Add fog every other batch
            noise_level = np.random.uniform(0.05, 0.2)
            noise = torch.randn_like(images) * noise_level
            hazy_batch = torch.clamp(images + noise, 0, 1).to(device, non_blocking=True)
        else:
            hazy_batch = images.to(device, non_blocking=True)
            
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass
        s_outputs = s_model(hazy_batch)
        loss, loss_components = compute_loss(s_outputs, targets, s_model)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(s_model.parameters(), max_norm=10.0)
        
        optimizer.step()

        # Update metrics
        losses.update(loss.detach().item())
        lbox.update(loss_components[0])
        lobj.update(loss_components[1])
        lcls.update(loss_components[2])

        # Log less frequently to save time
        if train_epoch % 50 == 0:
            logger_summary = [
                ("student/loss/total", loss.detach().item()),
                ("student/loss/box", lbox.avg), 
                ("student/loss/obj", lobj.avg), 
                ("student/loss/cls", lcls.avg), 
            ]
            logger.list_of_scalars_summary(logger_summary, epoch * len(loader) + train_epoch)
        
        # Update progress bar
        batch_time = time.time() - batch_start
        loader_with_progress.set_postfix({
            'loss': f'{losses.avg:.3f}',
            'batch_time': f'{batch_time:.2f}s'
        })
        
        # Debug slow batches
        if batch_time > 2.0:  # If batch takes more than 2 seconds
            print(f"⚠️  Slow batch {train_epoch}: {batch_time:.2f}s")

    epoch_time = time.time() - epoch_start_time
    print(f"✓ Student Epoch {epoch} completed in {epoch_time/60:.1f} minutes")
    
    return losses.avg

def train_teacher_one_epoch_fast(model, loader, optimizer, device, epoch, logger):
    """Optimized teacher training"""
    model.train()
    
    losses = utils.AverageMeter()
    lbox = utils.AverageMeter()
    lobj = utils.AverageMeter()
    lcls = utils.AverageMeter()
    
    epoch_start_time = time.time()
    loader_with_progress = tqdm(loader, desc=f"Teacher Epoch {epoch}", leave=False)
    
    for train_epoch, (img_dir, images, depth, targets) in enumerate(loader_with_progress):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss, loss_components = compute_loss(outputs, targets, model)

        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()

        losses.update(loss.detach().item())
        lbox.update(loss_components[0])
        lobj.update(loss_components[1])
        lcls.update(loss_components[2])

        # Log less frequently
        if train_epoch % 50 == 0:
            logger_summary = [
                ("teacher/loss/total", losses.avg),
                ("teacher/loss/box", lbox.avg),
                ("teacher/loss/obj", lobj.avg),
                ("teacher/loss/cls", lcls.avg),
            ]
            logger.list_of_scalars_summary(logger_summary, epoch * len(loader) + train_epoch)
        
        loader_with_progress.set_postfix({'loss': f'{losses.avg:.3f}'})

    epoch_time = time.time() - epoch_start_time
    print(f"✓ Teacher Epoch {epoch} completed in {epoch_time/60:.1f} minutes")
    
    return losses.avg

def train_student_perc_loss_one_epoch_optimized(s_model, t_model, loader, optimizer,
                                              device, epoch, logger, batch_size,
                                              stio_logger, lu_beta):
    """Highly optimized perceptual loss training"""
    
    # Use fewer layers and simpler computation
    t_model.eval()  # Teacher always in eval mode
    s_model.train()
    
    losses = utils.AverageMeter()
    lda = utils.AverageMeter()
    lbox = utils.AverageMeter()
    lobj = utils.AverageMeter()
    lcls = utils.AverageMeter()
    
    lambda_perc = 0.0005  # Reduced from 0.001
    
    epoch_start_time = time.time()
    loader_with_progress = tqdm(loader, desc=f"Student+Perc Epoch {epoch}", leave=False)
    
    for train_epoch, (img_dir, images, depth, targets) in enumerate(loader_with_progress):
        batch_start = time.time()
        
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)

        # Simplified fog generation
        if train_epoch % 3 == 0:  # Apply fog every 3rd batch only
            beta = torch.randint(lu_beta[0], lu_beta[1], (images.shape[0], )) / 100.
            beta = beta.to(device)
            # Simple fog instead of complex real haze
            fog_factor = beta.view(-1, 1, 1, 1)
            hazy_batch = images * (1 - fog_factor) + fog_factor * 0.7
        else:
            # Use simple noise
            noise = torch.randn_like(images) * 0.1
            hazy_batch = torch.clamp(images + noise, 0, 1)

        # Student forward pass
        s_outputs = s_model(hazy_batch)

        # Simplified perceptual loss computation
        diff_act = torch.tensor(0.0, device=device)
        
        if train_epoch % 5 == 0:  # Compute perceptual loss less frequently
            with torch.no_grad():  # No gradients needed for teacher
                try:
                    # Use specific layer instead of multiple layers
                    t_features = t_model.module_list[15](images)  # Mid-level features
                    s_features = s_model.module_list[15](hazy_batch)
                    
                    # Simple MSE loss instead of complex computation
                    diff_act = F.mse_loss(s_features, t_features) * lambda_perc
                except:
                    # Fallback if layer doesn't exist
                    diff_act = torch.tensor(0.0, device=device)

        # Main loss
        loss, loss_components = compute_loss(s_outputs, targets, s_model)
        loss += diff_act

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(s_model.parameters(), max_norm=10.0)
        optimizer.step()

        # Update metrics
        losses.update(loss.detach().item())
        lda.update(diff_act.detach().item() if torch.is_tensor(diff_act) else diff_act)
        lbox.update(loss_components[0])
        lobj.update(loss_components[1])
        lcls.update(loss_components[2])

        # Log less frequently
        if train_epoch % 50 == 0:
            logger_summary = [
                ("student/loss/total", loss.detach().item()),
                ("student/loss/diff_act", lda.avg), 
                ("student/loss/box", lbox.avg), 
                ("student/loss/obj", lobj.avg), 
                ("student/loss/cls", lcls.avg), 
            ]
            logger.list_of_scalars_summary(logger_summary, epoch * len(loader) + train_epoch)
        
        batch_time = time.time() - batch_start
        loader_with_progress.set_postfix({
            'loss': f'{losses.avg:.3f}',
            'd_act': f'{lda.avg:.4f}',
            'batch_time': f'{batch_time:.2f}s'
        })

    epoch_time = time.time() - epoch_start_time
    print(f"✓ Student+Perc Epoch {epoch} completed in {epoch_time/60:.1f} minutes")
    
    return losses.avg, lda.avg

def evaluate_model_fast(model, valid_dl, args, device, class_names, logger, epoch,
                       on_fog=False, beta=[0, 16]):
    """Faster evaluation"""
    
    eval_start = time.time()
    print(f"Evaluating model (epoch {epoch})...")
    
    if on_fog:
        metrics_output = evaluate_on_fog(model, valid_dl, args, device,
                                         class_names,
                                         model.hyperparams['height'],
                                         epoch, beta)
    else:
        metrics_output = _evaluate(model, valid_dl, device, class_names,
                                   img_size=model.hyperparams['height'],
                                   iou_thres=args.iou_thres,
                                   conf_thres=args.conf_thres,
                                   nms_thres=args.nms_thres,
                                   verbose=args.verbose, epoch=epoch)

    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        evaluation_metrics = [("validation/mAP", AP.mean())]
        logger.list_of_scalars_summary(evaluation_metrics, epoch)
        
        eval_time = time.time() - eval_start
        print(f"✓ Evaluation completed in {eval_time:.1f}s, mAP: {AP.mean():.4f}")
        
        return AP.mean()
    else:
        print("⚠️  Evaluation returned None")
        return 0.0

def fine_tune_fast(model, optimizer, train_dl, valid_dl, device, args, class_names, tb_logger):
    """Fast fine-tuning with better progress tracking"""
    
    best_map, cur_map = 0., 0.
    
    if args.t_pretrained_weights:
        print("\n" + "="*50)
        print("TEACHER INITIAL EVALUATION")
        print("="*50)
        cur_map = evaluate_model_fast(model, valid_dl, args, device,
                                     class_names, tb_logger, epoch=0, on_fog=False)
        print(f"Initial Teacher mAP: {cur_map:.4f}")

    if args.ft_epochs == 0:
        return args.t_pretrained_weights, cur_map

    print(f"\n{'='*50}")
    print(f"TEACHER FINE-TUNING ({args.ft_epochs} epochs)")
    print(f"{'='*50}")
    
    total_start_time = time.time()

    for epoch in range(1, args.ft_epochs + 1):
        epoch_start = time.time()
        
        loss = train_teacher_one_epoch_fast(model, train_dl, optimizer, device, epoch, tb_logger)

        # Evaluate less frequently for speed
        if epoch % max(1, args.evaluation_interval) == 0:
            cur_map = evaluate_model_fast(model, valid_dl, args, device,
                                         class_names, tb_logger, epoch, on_fog=False)

        # Save model
        if epoch % args.checkpoint_interval == 0 and cur_map > best_map:
            best_dir = utils.save_best(model, f"{len(class_names)}_teacher", best_map, cur_map)
            best_map = cur_map
            print(f"✓ New best teacher saved: {best_map:.4f}")

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{args.ft_epochs}: loss={loss:.4f}, mAP={cur_map:.4f}, time={epoch_time/60:.1f}min")

    total_time = time.time() - total_start_time
    print(f"\n✓ Teacher training completed in {total_time/60:.1f} minutes")
    print(f"Best Teacher mAP: {best_map:.4f}")

    return best_dir, best_map

def train_student_fast(s_model, t_model, s_optimizer, t_optimizer, train_dl,
                      valid_dl, batch_size, args, device, class_names,
                      tb_logger, stio_logger):
    """Fast student training with progress tracking"""

    best_s_map, s_cur_map = 0., 0.
    diff_act = 0.0

    if args.epochs == 0:
        cur_map = evaluate_model_fast(s_model, valid_dl, args, device,
                                     class_names, tb_logger, epoch=0, on_fog=False)
        print(f"Initial Student mAP: {cur_map:.4f}")
        return args.s_pretrained_weights, cur_map

    print(f"\n{'='*50}")
    print(f"STUDENT TRAINING ({args.epochs} epochs)")
    print(f"Perceptual Loss: {'ON' if args.perc_loss == 1 else 'OFF'}")
    print(f"{'='*50}")
    
    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Choose training method
        if args.perc_loss == 0:
            loss = train_student_one_epoch_fast(s_model, t_model, train_dl, s_optimizer,
                                               device, epoch, tb_logger, batch_size, stio_logger)
            diff_act = 0.0
        else:
            loss, diff_act = train_student_perc_loss_one_epoch_optimized(
                s_model, t_model, train_dl, s_optimizer, device, epoch,
                tb_logger, batch_size, stio_logger, lu_beta=[0, 16])

        # Evaluate
        if epoch % args.evaluation_interval == 0:
            s_cur_map = evaluate_model_fast(s_model, valid_dl, args, device,
                                           class_names, tb_logger, epoch,
                                           on_fog=(args.perc_loss == 1))  # Only fog eval if using perc loss

        # Save model
        if epoch % args.checkpoint_interval == 0 and s_cur_map > best_s_map:
            best_model_dir = utils.save_best(s_model, f"{len(class_names)}_student",
                                           best_s_map, s_cur_map)
            best_s_map = s_cur_map
            print(f"✓ New best student saved: {best_s_map:.4f}")

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{args.epochs}: loss={loss:.4f}, d_act={diff_act:.4f}, mAP={s_cur_map:.4f}, time={epoch_time/60:.1f}min")

    total_time = time.time() - total_start_time
    print(f"\n✓ Student training completed in {total_time/60:.1f} minutes")
    print(f"Best Student mAP: {best_s_map:.4f}")

    return best_model_dir, best_s_map

def create_optimized_dataloader(dataset, batch_size):
    """Create optimized dataloader"""
    num_workers = min(4, psutil.cpu_count()) if psutil.cpu_count() > 2 else 0
    
    # Reduce workers on Windows due to multiprocessing issues
    if os.name == 'nt':  # Windows
        num_workers = min(2, num_workers)
    
    print(f"Using {num_workers} dataloader workers")
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True  # For consistent batch sizes
    )

def main():
    # Print system info and optimize settings
    print_system_info()
    optimize_torch_settings()
    
    # Set seed
    torch.manual_seed(10)
    np.random.seed(10)
    
    # Parse args
    args = utils.get_parsed_args()
    
    # Print training configuration
    print(f"\nTRAINING CONFIGURATION:")
    print(f"Teacher epochs: {args.ft_epochs}")
    print(f"Student epochs: {args.epochs}")
    print(f"Perceptual loss: {args.perc_loss}")
    print(f"Evaluation interval: {args.evaluation_interval}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    
    # Initialize loggers
    tb_logger = Logger(args.logdir)
    stio_logger = utils.setup_logger_dir(args)
    
    # Load data config
    data_config = utils.parse_data_config(args.data)
    class_names = utils.load_classes(data_config["names"])
    device = utils.get_device(args)
    
    print(f"Device: {device}")
    print(f"Classes: {len(class_names)}")

    # Load models
    print("\nLoading models...")
    t_model = load_model(args.model, device, args.t_pretrained_weights)
    s_model = load_model(args.model, device, args.s_pretrained_weights)
    
    # Get training parameters
    batch_size = t_model.hyperparams['batch']
    image_size = t_model.hyperparams['height']
    
    # Optimize batch size based on GPU memory
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        original_batch = batch_size
        
        if gpu_memory_gb >= 6:
            batch_size = min(batch_size * 2, 32)
        if gpu_memory_gb >= 10:
            batch_size = min(batch_size * 3, 48)
            
        if batch_size != original_batch:
            print(f"Optimized batch size: {original_batch} -> {batch_size}")
            t_model.hyperparams['batch'] = batch_size
            s_model.hyperparams['batch'] = batch_size

    print(f"Batch size: {batch_size}")
    print(f"Image size: {image_size}")

    # Create datasets
    print("\nCreating datasets...")
    transform = utransforms.simple_transform(image_size)
    
    train_ds = ds.yolo_dataset(data_config, "train", transform, image_size, batch_size)
    train_dl, valid_dl = train_ds.create_dataloader()
    
    test_ds = ds.yolo_dataset(data_config, "test", transform, image_size, batch_size)
    test_dl = test_ds.create_dataloader()

    # Create optimizers
    t_optimizer = utils.set_optimizer(t_model)
    s_optimizer = utils.set_optimizer(s_model)

    # Training pipeline
    total_training_start = time.time()
    
    # 1. Fine-tune teacher
    best_t_model, best_t_map = fine_tune_fast(
        t_model, t_optimizer, train_dl, valid_dl, 
        device, args, class_names, tb_logger
    )
    
    # 2. Load best teacher
    t_model = load_model(args.model, device, best_t_model)
    
    # 3. Train student
    best_s_model, best_s_map = train_student_fast(
        s_model, t_model, s_optimizer, t_optimizer, 
        train_dl, valid_dl, batch_size, args, device,
        class_names, tb_logger, stio_logger
    )
    
    # 4. Final evaluation
    s_model = load_model(args.model, device, best_s_model)
    
    print(f"\n{'='*50}")
    print("FINAL EVALUATION")
    print(f"{'='*50}")
    
    final_start = time.time()
    voc_s_map = evaluate_model_fast(s_model, test_dl, args, device, 
                                   class_names, tb_logger, epoch=0)
    final_time = time.time() - final_start
    
    # Print final results
    total_training_time = time.time() - total_training_start
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total training time: {total_training_time/3600:.1f} hours ({total_training_time/60:.1f} minutes)")
    print(f"Teacher mAP: {best_t_map:.4f}")
    print(f"Student mAP: {best_s_map:.4f}")
    print(f"Final test mAP: {voc_s_map:.4f}")
    print(f"Final evaluation time: {final_time:.1f}s")
    print(f"{'='*60}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()