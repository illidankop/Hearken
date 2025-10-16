############################ 
# Imports 
############################
import os
from random import shuffle
import sys


sys.path.insert(1, os.path.join(sys.path[0], '../utils'))  # Remove?
import numpy as np
import time
import logging
import matplotlib.pyplot as plt

import yaml
from box import Box
import wandb

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import warnings
import torch.optim as optim
from sklearn.metrics import f1_score  
torch.manual_seed(17)

# Self 
from model_config import Config
from losses import get_loss_func
from pytorch_utils import move_data_to_device, do_mixup
from utilities import (create_folder, get_filename, create_logging, StatisticsContainer, Mixup)
from data import BackgroundDataset,datasetFactory,collate_fn
from data import MixBackground, WhiteNoise, PitchShift,PolarityInversion, MovingAverage,Resample,TimeStretch

from models import Transfer_Cnn14, Transfer_MobileNetV2, Regression_MobileNetV2, MobileNetV2, Cnn14, Wavegram_Logmel_Cnn14, Transfer_Cnn14_mel128
from evaluate import Evaluator, log_samples_to_wandb
from statistics import mean


############################ 
# Functions
############################

def validation(config,validate_loader,evaluator,prev_loss,iteration,val_fold,epoch):
    """Validation step - log Accuracy, Binary Accuracy, F1 score and Confusion matrix of the model state over 
                         over validation set
    Args:
        config : model config
        validate_loader : validation set data loader 
        evaluator (_type_): _description_
        statistics_container (_type_): _description_
        prev_loss (_type_): _description_
        iteration (_type_): _description_
        val_fold (_type_): validation fold
        epoch (int): epoch number
    """

    logging.info('------------------------------------')
    logging.info('Iteration: {}'.format(iteration))

    statistics= evaluator.evaluate(validate_loader, config)
    logging.info('Validate accuracy: {:.3f}'.format(statistics['accuracy']))
    logging.info('Current Loss: {:.3f}'.format(prev_loss))

    ### Log confusion matrix to wandb
    cm = []
    for i in range(config.classes_num):
        for j in range(config.classes_num):
            cm.append([config.labels[i], config.labels[j], statistics['confusion'][i, j]])
    fields = {
        "Actual": "Actual",
        "Predicted": "Predicted",
        "nPredictions": "nPredictions",
    }
    cm_plot = wandb.plot_table(
        "wandb/confusion_matrix/v1",
        wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=cm),
        fields,
        {"title": f"Confusion matrix{val_fold}"},
    )
    wandb.log({f"ConfusionMatrix{val_fold}": cm_plot,"epoch":epoch,"fold":val_fold})
    wandb.log({f"Accuracy_{val_fold}":statistics['accuracy'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"Atm vs nonAtn Accuracy_{val_fold}":statistics['atmNonAccuracy'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"Macro Avg Accuracy_{val_fold}":statistics['macro_avg_accuracy'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"Balanced Accuracy_{val_fold}":statistics['balanced_accuracy'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"ATM Accuracy_{val_fold}":statistics['class_0_accuracy'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"BKG Accuracy_{val_fold}":statistics['class_1_accuracy'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"nonATM Accuracy_{val_fold}":statistics['class_2_accuracy'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"Macro Recall_{val_fold}":statistics['macro_recall'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"Weighted Recall_{val_fold}":statistics['weighted_recall'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"ATM Recall_{val_fold}":statistics['class_0_recall'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"BKG Recall_{val_fold}":statistics['class_1_recall'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"nonATM Recall_{val_fold}":statistics['class_2_recall'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"Macro Precision_{val_fold}":statistics['macro_precision'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"Weighted Precision_{val_fold}":statistics['weighted_precision'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"ATM Precision_{val_fold}":statistics['class_0_precision'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"BKG Precision_{val_fold}":statistics['class_1_precision'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"nonATM Precision_{val_fold}":statistics['class_2_precision'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"Macro_F1Score_{val_fold}":statistics['macro_F1Score'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"Weighted_F1Score_{val_fold}":statistics['weighted_F1Score'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"ATM F1Score_{val_fold}":statistics['class_0_f1'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"BKG F1Score_{val_fold}":statistics['class_1_f1'],"epoch":epoch,"fold":val_fold})
    wandb.log({f"nonATM F1Score_{val_fold}":statistics['class_2_f1'],"epoch":epoch,"fold":val_fold})
    


    # wandb.log({f"Val Loss_{val_fold}": statistics['val_loss'], "epoch": epoch})
    # samples_dict = statistics['samples']
    
    #logging samples to wandb
    # [log_samples_to_wandb(samples_dict[cls]["cls_desc"], samples_dict[cls]["audio_file_name"]
    #                       ,samples_dict[cls]["waveform"], samples_dict[cls]["sr"]) 
    #                         for cls in samples_dict.keys()
    #                         ] 

    wandb.log({f"roc{val_fold}" : wandb.plot.roc_curve( np.argmax(statistics['gt'],axis=-1), statistics['scores'],labels=['atm','back','non'])
                ,"epoch":epoch,"fold":val_fold})

    # return statistics['atmNonAccuracy']
    return statistics

'''
def validation(config, validate_loader, evaluator, prev_loss, iteration, val_fold, epoch):
    logging.info('------------------------------------')
    logging.info(f'Iteration: {iteration}')

    statistics = evaluator.evaluate(validate_loader, config)
    logging.info(f'Validate accuracy: {statistics["accuracy"]:.3f}')
    logging.info(f'Current Loss: {prev_loss:.3f}')

    # Assuming you adjust your evaluator to return 'val_loss'
    val_loss = statistics.get('val_loss', prev_loss)  # Use prev_loss as a fallback

    # Simplified logging to wandb
    wandb.log({
        f"Val Accuracy_{val_fold}": statistics['accuracy'],
        f"Val Loss_{val_fold}": val_loss,
        f"F1Score_{val_fold}": statistics['F1Score'],
        f"BinaryAccuracy{val_fold}": statistics['atmNonAccuracy'],
        f"Precision_{val_fold}": statistics['precision'],
        f"Recall_{val_fold}": statistics['recall'],
        "epoch": epoch,
        "fold": val_fold
    })

    # Additional logs (e.g., ROC curve) as needed
    wandb.log({
        f"roc{val_fold}": wandb.plot.roc_curve(np.argmax(statistics['gt'], axis=-1), statistics['scores'], labels=['atm', 'back', 'non']),
        "epoch": epoch,
        "fold": val_fold
    })

    return val_loss  # or statistics['atmNonAccuracy'] depending on what you want to track for early stopping or other logic
'''

def trainLoop(model,config,train_loader,device,loss_func,optimizer,validate_loader,evaluator,val_fold):
    """Training step

    Args:
        model
        config: model config
        train_loader: train dataloader 
        device (str) : cuda/cpu 
        loss_func 
        optimizer 
        validate_loader : validation data loader
        evaluator (_type_): _description_
        statistics_container (_type_): _description_
        val_fold (int): validation fold
    """
    prev_loss = 99
    accuracy = 0
    best_accuracy = 0
    best_avarege_score =0

    best_model_path = f"models/best_model_{val_fold}_03march_2.pth"
    model_path = f"models/mobileNet_{config.expType}_{val_fold}_03march_2.pth"

    # Early stopping initialization
    #===============================
    # best_train_loss = float('inf')
    # best_val_loss = float('inf')
    early_stop_flag = 0
    patience_counter = 0
    training_patience_counter = 0
    validation_patience_counter = 0 
    gap_patience_counter = 0
    patience_limit = 5  # Number of epochs to wait for possible improvement
    prev_train_epoch_loss = float('inf')
    prev_val_epoch_loss = float('inf')
    prev_mean_training_loss = float('inf')
    prev_mean_validation_loss = float('inf')
    prev_mean_gap_loss = float('inf')
    prev_mean_performance = 0.0

    best_train_epoch_loss = float('inf')
    best_val_epoch_loss = float('inf')
    best_mean_training_loss = float('inf')
    best_mean_validation_loss = float('inf')
    best_mean_gap_loss = float('inf')
    best_mean_performance = 0.0

    training_loss_check = []
    validation_loss_check = []
    gap_loss_check = []
    
    training_loss_values = []
    validation_loss_values = []
    gap_loss_values = []
    performance_values = []

    top_limits = 10
    allowed_number = 8  # Allowed number of '1's in the last 'top_limits' epochs
   
    for epoch in range(config.epochs):
        model.train()
        sum_train_batch_loss = 0
        loss = 0

        for iteration,batch_data_dict in enumerate(train_loader,0):

            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

            batch_output_dict = model(batch_data_dict['waveform'], None)

            batch_target_dict = {'target': batch_data_dict['target']}

            # loss
            loss = loss_func(batch_output_dict, batch_target_dict)
            train_prev_loss = loss
            
            wandb.log({f"Train Batch Loss": loss})

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_train_batch_loss += loss.item()

        train_epoch_loss = sum_train_batch_loss/len(train_loader) 
        wandb.log({f"Training Loss": train_epoch_loss, "epoch": epoch}) 

        # validation_statistics
        #----------------------
        model.eval()
        # accuracy = validation(config,validate_loader,evaluator,prev_loss,iteration,val_fold,epoch)
        statistics = validation(config, validate_loader, evaluator, prev_loss, iteration, val_fold, epoch)
        accuracy = statistics['accuracy']
        atm_vs_nonAtn_accuracy = statistics['atmNonAccuracy']
        macro_avg_accuracy = statistics['macro_avg_accuracy']
        balanced_accuracy = statistics['balanced_accuracy']
        macro_recall = statistics['macro_recall']
        weighted_recall = statistics['weighted_recall']
        macro_precision = statistics['macro_precision']
        weighted_precision = statistics['weighted_precision']
        macro_F1Score = statistics['macro_F1Score']
        weighted_F1Score = statistics['weighted_F1Score']
        avarege_score = (atm_vs_nonAtn_accuracy + macro_avg_accuracy + weighted_recall + macro_precision + weighted_precision + macro_F1Score + weighted_F1Score)/7
        # validation_loss
        #----------------
        model.eval()
        sum_val_batch_loss = 0
        val_batch_loss = 0

        for iteration,val_batch_data_dict in enumerate(validate_loader,0):

            for key in val_batch_data_dict.keys():
                val_batch_data_dict[key] = move_data_to_device(val_batch_data_dict[key], device)

            val_batch_output_dict = model(val_batch_data_dict['waveform'], None)

            val_batch_target_dict = {'target': val_batch_data_dict['target']}

            # loss
            val_batch_loss = loss_func(val_batch_output_dict, val_batch_target_dict)
            val_prev_loss = val_batch_loss
            
            wandb.log({f"Val Batch Loss": val_batch_loss})
            sum_val_batch_loss += val_batch_loss.item() 
        
        val_epoch_loss = sum_val_batch_loss / len(validate_loader)
        wandb.log({f"Validation Loss": val_epoch_loss, "epoch": epoch})

        print(f"Epoch {epoch}: Training Loss = {train_epoch_loss}, Validation Loss = {val_epoch_loss}")
        print("metrics results:")
        print("================")
        print(f"Epoch {epoch} Validation Metrics:")
        print(f"Accuracy: {accuracy}")
        print(f"ATM vs NonATM Accuracy: {atm_vs_nonAtn_accuracy}")
        print(f"Macro Avg Accuracy: {macro_avg_accuracy}")
        print(f"Balanced Accuracy: {balanced_accuracy}")
        print(f"Macro recall: {macro_recall}")
        print(f"Weighted recall: {weighted_recall}")
        print(f"Macro precision: {macro_precision}")
        print(f"Weighted precision: {weighted_precision}")
        print(f"Macro F1Score: {macro_F1Score}")
        print(f"Weighted F1Score: {weighted_F1Score}")
        
        # save best model using criteria:
        #---------------------------------
        # accuracy = statistics['accuracy']
        # if accuracy > best_accuracy:
        #     best_accuracy = accuracy
        if avarege_score > best_avarege_score:
            best_avarege_score = avarege_score
            print('best_avarege_score =', best_avarege_score )
            # Save model
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), best_model_path)
            else:
                torch.save(model.state_dict(), best_model_path)
            # Optionally log the model as an artifact
            wandb.save(f"models/best_model_mobileNet_{config.expType}_{val_fold}.pth")    
            wandb.log_artifact(best_model_path, type='model', name='best_model')
            best_model_accuracy = accuracy
            best_model_atm_vs_nonAtn_accuracy = atm_vs_nonAtn_accuracy
            best_model_macro_avg_accuracy = macro_avg_accuracy
            best_model_balanced_accuracy = balanced_accuracy
            best_model_macro_recall = macro_recall
            best_model_weighted_recall = weighted_recall
            best_model_macro_precision = macro_precision
            best_model_weighted_precision = weighted_precision
            best_model_macro_F1Score = macro_F1Score
            best_model_weighted_F1Score = weighted_F1Score
            best_model_epoch = epoch
            print('best_model_epoch =', best_model_epoch)

        # number of epochs without improvement in performance:
        last_best_performance_distance = epoch - best_model_epoch
        print('last_best_performance_distance =', last_best_performance_distance)
            

        # Early stopping condition mechnism
        #---------------------------------
        # Early Stopping  1:
        training_loss_values.append(train_epoch_loss)
        validation_loss_values.append(val_epoch_loss)
        gap_loss_values.append(val_epoch_loss- train_epoch_loss)
        performance_values.append(avarege_score) 

        if len(training_loss_check) >= top_limits and epoch >= 2 * top_limits:
            # check for trends
            mean_gap_loss = mean(gap_loss_values[-top_limits:])
            mean_training_loss = mean(training_loss_values[-top_limits:])
            mean_validation_loss = mean(validation_loss_values[-top_limits:])
            mean_performance = mean(performance_values[-top_limits:])

            # set patience_counter for early stopping
            # ---------------------------------------------
            if mean_training_loss < best_mean_training_loss:  # check if training loss is goiing down
                best_mean_training_loss = mean_training_loss
                patience_counter = 0
                if mean_validation_loss < best_mean_validation_loss: # check if validation loss is goiing down
                    best_mean_validation_loss = mean_validation_loss
                    patience_counter = 0
                    if mean_gap_loss < best_mean_gap_loss: # check if gap is decressing
                        best_mean_gap_loss = mean_gap_loss
                        patience_counter = 0
                        if mean_performance > best_mean_performance: # check if performance is increasing
                            best_mean_performance = mean_performance
                            patience_counter = 0
                        else:
                            patience_counter += 1        
                    else:
                        patience_counter += 1
                else:
                    patience_counter += 1
            else:
                patience_counter += 1 
            
            # set additional counters for early stopping
            # ---------------------------------------------
            if mean_training_loss < prev_mean_training_loss:  # check if training loss is decreasing
                prev_mean_training_loss = mean_training_loss
                training_patience_counter = 0
            else:
                training_patience_counter += 1
            
            if mean_validation_loss < prev_mean_validation_loss: # check if validation loss is decreasing
                prev_mean_validation_loss = mean_validation_loss
                validation_patience_counter = 0
            else:
                validation_patience_counter += 1

            if mean_gap_loss < prev_mean_gap_loss: # check if gap is decreasing
                prev_mean_gap_loss = mean_gap_loss
                gap_patience_counter = 0
            else:
                gap_patience_counter += 1  # gap is increasing

            if mean_performance > prev_mean_performance: # check if performance is increasing
                prev_mean_performance = mean_performance
                performance_counter = 0
            else:
                performance_counter += 1  # gap is decreasing
 
            print('patience_counter =', patience_counter)                     
            print('training_patience_counter =', training_patience_counter)
            print('validation_patience_counter =', validation_patience_counter)
            print('gap_patience_counter =', gap_patience_counter)
            print('performance_counter =', performance_counter)

            # early stopping set of conditions
            # ---------------------------------------------
            if patience_counter > patience_limit:
                early_stop_flag = 1
                print('early stop trigered due to: patience_counter')
            if  last_best_performance_distance > 20:
                early_stop_flag = 1
                print('early stop trigered due to: large_performance_distance')
            if training_patience_counter > patience_limit: 
                early_stop_flag = 1
                print('early stop trigered due to: training_patience_counter')
            if training_patience_counter > patience_limit: 
                early_stop_flag = 1
                print('early stop trigered due to: training_patience_counter')
            if (validation_patience_counter > patience_limit):
                early_stop_flag = 1
                print('early stop trigered due to: validation_patience_counter') 
            if (gap_patience_counter > patience_limit) and (performance_counter > patience_limit):    
                early_stop_flag = 1
                print('early stop trigered due to: performance decreasing') 

            if early_stop_flag == 1:
                print(f"Early stopping 1 triggered at epoch {epoch}")
                # Save the model checkpoint for the best training loss observed so far
                save_path = model_path #f"models/best_model_{val_fold}_30feb.pth"
                if hasattr(model, 'module'):
                    torch.save(model.module.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)
                # Log checkpoint
                wandb.save(f"models/mobileNet_{config.expType}_{val_fold}.pth")
                print("Training completed with early stopping 1.")
                break  
                    
                

        # Early Stopping mechanism 2:
        # ---------------------------    
        training_loss_check.append(int(train_epoch_loss >= prev_train_epoch_loss)) # add 1 if condition is true else 0
        validation_loss_check.append(int(val_epoch_loss >= prev_val_epoch_loss)) # add 1 if condition is true else 0
        gap_increase = (val_epoch_loss - train_epoch_loss) > (prev_val_epoch_loss - prev_train_epoch_loss)
        gap_loss_check.append(int(gap_increase))

        # Ensure lists don't grow beyond necessary by trimming the oldest entry if over limit
        for check_list in [training_loss_check, validation_loss_check, gap_loss_check]:
            if len(check_list) > top_limits:
                check_list.pop(0)

        print('training_loss_check =', training_loss_check)
        print('training_loss_check_len =', len(training_loss_check))
        print('training_loss_check_sum =',sum(training_loss_check[-top_limits:]))

        print('validation_loss_check =', validation_loss_check)
        print('validation_loss_check_len =', len(validation_loss_check))
        print('validation_loss_check_sum =',sum(validation_loss_check[-top_limits:]))

        print('gap_loss_check =', gap_loss_check)
        print('gap_loss_check_len =', len(gap_loss_check))
        print('gap_loss_check_sum =',sum(gap_loss_check[-top_limits:]))

        if len(training_loss_check) >= top_limits and epoch > 2 * top_limits:  # Ensures we have enough data points to make a decision
            if sum(training_loss_check[-top_limits:]) >= allowed_number or \
            sum(validation_loss_check[-top_limits:]) >= allowed_number or \
            sum(gap_loss_check[-top_limits:]) >= allowed_number:
                print(f"Early stopping 2 triggered at epoch {epoch}")
                # Save the model checkpoint for the best training loss observed so far
                save_path = model_path #f"models/best_model_{val_fold}_30feb.pth"
                if hasattr(model, 'module'):
                    torch.save(model.module.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)
                # Log checkpoint
                wandb.save(f"models/mobileNet_{config.expType}_{val_fold}.pth")
                print("Training completed with early stopping 2.")
                break 

        # Update previous losses
        prev_train_epoch_loss = train_epoch_loss
        prev_val_epoch_loss = val_epoch_loss

    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
    # Log checkpoint
    wandb.save(f"models/mobileNet_{config.expType}_{val_fold}.pth")
    print("Model saved at:", model_path)
    
    print("=---BEST MODEL-------=")
    print("Best model saved at:", best_model_path)
    print("best_model_epoch =", best_model_epoch)
    print("best_model_accuracy =", best_model_accuracy)
    print("best_model_atm_vs_nonAtn_accuracy =", best_model_atm_vs_nonAtn_accuracy)
    print("best_model_macro_avg_accuracy =",  best_model_macro_avg_accuracy)
    print("best_model_balanced_accuracy =",  best_model_balanced_accuracy)
    print("best_model_macro_recall =",  best_model_macro_recall)
    print("best_model_weighted_recall =",  best_model_weighted_recall)
    print("best_model_macro_precision =", best_model_macro_precision)
    print("best_model_weighted_precision =", best_model_weighted_precision)
    print("best_model_macro_F1Score =", best_model_macro_F1Score)
    print("best_model_weighted_F1Score=",best_model_weighted_F1Score)
    print('')
    
    print("trainloop as been done :)")


def train(args,config,folds,val_folds,device='cuda'):
    """
        Train loop + validation/test phase
    Args:
        args (_type_): _description_
        config (_type_): _description_
        folds (_type_): _description_
        val_fold (_type_): _description_
        statistics_path (_type_): _description_
        device (str, optional): _description_. Defaults to 'cuda'.
    """
    pretrain = True if args.pretrained_checkpoint_path else False

    # Dataset paths 
    hdf5_path = os.path.join(args.dataset_dir, args.dataset_name + '.h5')
    
    if(args.testset_name != None):
        hdf5_test_path = os.path.join(args.dataset_dir, args.testset_name + '.h5')
    else: 
        hdf5_test_path = None
    
    # Model
    Model = eval(args.model_type)
    model = Model(config.sample_rate, 
                    config.window_size, 
                    config.hop_size, 
                    config.mel_bins, 
                    config.fmin, 
                    config.fmax,
                    config.classes_num, 
                    args.freeze_base
                    )
    # Loading pretrained model
    if pretrain:
        logging.info('Load pretrained model from {}'.format(args.pretrained_checkpoint_path))
        model.load_from_pretrain(args.pretrained_checkpoint_path)

    if args.load_checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint)

    # Extra parallelism
    model = torch.nn.DataParallel(model)
    model.to(device)
    wandb.watch(model)

    # Augmentations init
    background_dataset = BackgroundDataset(config.clip_samples,hdf5_path,use_folds=folds,transform=[])
    augmentations_dict = {
                    'mixBackground':  MixBackground(background_dataset,
                                                    mb_alpha=args.mixBackground_mb_alpha,
                                                    prob=args.mixBackground_prob),
                    'whiteNoise': WhiteNoise(prob=args.whiteNoise_prob,
                                                noiseFactor=args.whiteNoise_noiseFactor),
                    'movingAverage': MovingAverage(cutOff=args.movingAverage_cutoff,
                                                sr=config.sample_rate,
                                                prob=args.movingAverage_prob),
                    'polarityInversion': PolarityInversion(prob=args.polarityInversion_prob),
                    'pitchShift': PitchShift(sr=config.sample_rate,
                                                pitchUpMax=args.pitchShift_pitchUpMax,
                                                pitchUpMin=args.pitchShift_pitchUpMin,
                                                pitchDnMax=args.pitchShift_pitchDnMax,
                                                pitchDnMin=args.pitchShift_pitchDnMin,
                                                prob=args.pitchShift_prob),
                    'TimeStretch': TimeStretch(stretchMin=args.TimeStretch_stretchMin,
                                     stretchMax=args.TimeStretch_stretchMax,
                                     prob=args.TimeStretch_prob),
                    'resample': Resample(sr=config.sample_rate,
                                        new_sr=args.resample_new_sr)
    }

    augmentations = torch.nn.Sequential()
    if(config.augmentations is not None):
        for a in config.augmentations:
            augmentations.add_module(a,augmentations_dict[a])

    # if 'mixup' in config.augmentation:
    #     augmentations.append(Mixup(mixup_alpha=1.))
    # ...


    # DataSet
    Dataset = datasetFactory(config.expType)

    train_dataset = Dataset(config.clip_samples,hdf5_path,use_folds=folds,transform=augmentations)
    
    # Use validation folds
    if(val_folds):
        val_dataset = Dataset(config.clip_samples,hdf5_path,use_folds=val_folds,
                                        transform=None)
    # Use test set as validation
    elif(hdf5_test_path != None):
        val_dataset = Dataset(config.clip_samples,hdf5_test_path,use_folds=None,
                                        transform=None)
    # Use train set for validation - for debug only
    else:
        val_dataset = Dataset(config.clip_samples,hdf5_path,use_folds=folds,
                                        transform=None)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            collate_fn=collate_fn,
                                            batch_size = config.batch_size,
                                            num_workers=args.num_workers, 
                                            pin_memory=True,shuffle=True)

    validate_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                collate_fn=collate_fn,
                                                batch_size = config.batch_size,
                                                num_workers=args.num_workers, 
                                                pin_memory=True,
                                                shuffle=True)


    # Optimizer
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999),
    #                        eps=1e-08, weight_decay=0., amsgrad=True)
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Loss
    loss_func = get_loss_func(config.loss_type)

    # Evaluator
    evaluator = Evaluator(model=model)

    # Training Loop
    trainLoop(model,config,train_loader,device,loss_func,optimizer,validate_loader,evaluator,val_folds)

  


############################ 
# Main
###########################
def main(args,expType):


    config = Config(args[expType])
    Dataset = datasetFactory(expType)


    # Set up for training
    device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
    print(f"Running on {device}")

    logs_dir = os.path.join(args.workspace, 'logs', args.filename,
                            args.dataset_name + args.model_type +
                                   'loss_type={}'.format(config.loss_type) +
                                   'bs={}'.format(config.batch_size) + 'freeze={}'.format(args.freeze_base))
    create_logging(logs_dir, 'w')
    logging.info(args)


    # Wandb & Configuration
    wandb.init(project="ATMDetection",config=vars(config),notes=f"Exp:{expType}")
    config.update(wandb.config)
    args.update(wandb.config.__dict__['_items'])
    

    # Cross validation on each fold
    #------------------------------
    # folds =  args.folds
    # for fold in folds:
    #     use_folds = [f for f in folds if f != fold]
    #     # if(fold != 2):
    #     #     continue
    #     wandb.log({'val_fold':fold})
    #     train(args,config,use_folds,val_folds = [fold],device=device)

    # Train on with test set as val
    #-------------------------------
    train(args,config,folds=args.folds,val_folds=None,device='cuda')

    print("Finished training")
    wandb.finish()   



if __name__ == '__main__':

    # Load config file
    args = yaml.load(open("args.yaml"), Loader=yaml.FullLoader)
    # Dot notation
    args = Box(args)

    
    args.filename = get_filename(__file__)

    # Set the type of classifier - according to config file
    expType = 'A_B_N'

    # Wandb offline
    # os.environ['WANDB_MODE'] = 'offline'

    main(args,expType)



 