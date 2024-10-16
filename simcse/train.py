# import debugpy;debugpy.connect(('10.119.57.39', 5627))
import argparse
import logging
import math
import os
import random

import datasets

from torch.utils.data.dataloader import DataLoader

import torch

from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator

import pdb
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

from data import ImgSentDataset, SentDataset
from cheatCSE import CheatCSE, BertForCL, RobertaForCL, CheatModel
from utils import evaluate

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cheat",
        action = "store_true",
        help = "use cheat features"
    )

    parser.add_argument(
        "--framework",
        type=str,
        default="simcse",
        help="The framework to use.",
        choices=["simcse", "cheatcse"]
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="A .txt file containing the training sentences."
    )
    ###应该不会用######
    parser.add_argument(
        "--feature_file",
        type=str,
        default=None,
        help="A .hdf5 file containing the image features (e.g. ResNet50 features)."
    )
    parser.add_argument(
        "--shuffle_imgs",
        action="store_true",
        help="Ablation study for random imgs",
    )
    parser.add_argument(
        "--random_imgs",
        action="store_true",
        help="Ablation study for random imgs",
    )
    ##################
    parser.add_argument(
        "--output_dir",
        type=str,
        default="result/",
        help="Where to store the final model."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=32,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to use.")
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="A seed for reproducible training. We used [0,1,2,3,4] in experiments.")
    parser.add_argument(
        "--temp",
        type=float,
        default=0.05,
        help="Temperature for softmax.")
    parser.add_argument(
        "--temp_cheat",
        type=float,
        default=0.05,
        help="Temperature for cheat-cross-modality contrastive learning"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=768,
        help="Text embedding dimention of pooled output (mlp)"
    )
    parser.add_argument(
        "--proj_dim",
        type=int,
        default=256,
        help="Projection dimension in grounding space"
    )
    parser.add_argument(
        "--lbd",
        type=float,
        default=0.01,
        help="weight for inter-modality loss"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=125,
        help="evaluation step interval")
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default='stsb_spearman',
        help="for saving best checkpoint")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type = int,
        default = 1,
        help = "Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument(
        "--shot",
        type=int,
        default=-1,
        help="few-shot setting")

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    return args

def main():
    args = parse_args()
    print(args)


    # Initialize the accelerator.

    accelerator = Accelerator()
    args.device = accelerator.device

    # Make one log on every process with the configuration for debugging.

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.

    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    set_seed(args.seed)


    # Load pretrained tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast = True)
    tokenizer.save_pretrained(args.output_dir)
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    config = AutoConfig.from_pretrained(args.model_name_or_path)

    if 'roberta' in args.model_name_or_path:
        lang_model = RobertaForCL.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            model_args=args
        )
    elif 'bert' in args.model_name_or_path:
        lang_model = BertForCL.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                model_args=args
        )
    else:
        raise NotImplementedError

    
    if args.framework.lower() == 'cheatcse':
        #后续添加相关参数，在cheatCSE.py文件里面
        cheat_model = CheatModel()
    
    else:
        cheat_model = None

    model = CheatCSE(lang_model,cheat_model, args)


    #Define collator function
    def data_collator(batch):
        keys = batch[0].keys()
        sentences = [b['sent'] for b in batch]

        new_batch = {}

        total = len(sentences)


        #tokenizer

        tokenized_sents = tokenizer(
            sentences,
            max_length=args.max_seq_length,
            truncation=True,
            padding="max_length" if args.pad_to_max_length else 'longest',
            return_tensors='pt'
        )

        for key in ['input_ids', 'attention_mask', 'token_type_ids', 'position_ids']:
            #这里需要后面补充
            if args.cheat:
                pass
            else:# (bs, len) -> (bs, 2, len)
                if key in tokenized_sents.keys():
                    new_batch[key] = tokenized_sents[key].unsqueeze(1).repeat(1, 2, 1)

            
        return new_batch

    # only sent
    train_dataset = SentDataset(text_file=args.text_file, shot = args.shot)


    train_dataloader = DataLoader(train_dataset,
                                shuffle=True,
                                batch_size=args.per_device_train_batch_size,
                                collate_fn=data_collator)

    

    # Log a few random samples from the training set:

    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not. Same as examples in huggingface and sentence-transformer.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    #num_update_steps_per_epoch = math.ceil(num_update_steps_per_epoch * args.percent)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    set_seed(args.seed)  # for sake of the status change of sampler


    #Start Training! num_processes -> 1

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Train train batch size (w.parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_metric = 0

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()

            if args.framework.lower() == 'cheatcse':
                #后面补充
                # intra_loss, inter_loss =  model.compute_loss(batch, cheat=True)
                # loss = intra_loss + args.lbd * inter_loss
                pass
            else:
                loss = model.compute_loss(batch, cheat=False)

            loss = loss / args.gradient_accumulation_steps

            accelerator.backward(loss)

            if (step+1) % args.gradient_accumulation_steps == 0 or len(train_dataloader) -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps +=1
            
            if  (step+1) % args.gradient_accumulation_steps == 0 and (completed_steps % args.eval_steps == 0  or  completed_steps >= args.max_train_steps):
                logger.info("***** Start evaluation *****")
                model.eval()
                # pdb.set_trace()
                metrics = evaluate(model.lang_model, tokenizer)
                logger.info(f" step {completed_steps}: eval_stsb_spearman = {metrics['eval_stsb_spearman']}")

                if metrics['eval_'+args.metric_for_best_model] > best_metric:
                    # evaluation
                    best_metric = metrics['eval_'+args.metric_for_best_model]

                    # save (1) pytorch_model.bin  (2) config.json
                    logger.info("Saving best model checkpoint to %s", args.output_dir)
                    accelerator.wait_for_everyone() # wait for all processes to reach that point in the script
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.lang_model.save_pretrained(args.output_dir, save_function=accelerator.save)

                    if args.framework.lower() == "cheatcse":
                        accelerator.save(
                            {
                                'cheat_model': unwrapped_model.cheat_model.state_dict(),
                                'grounding': unwrapped_model.grounding.state_dict()
                            },
                            os.path.join(args.output_dir, 'cheatcse.pt')
                        )
                
                if completed_steps >= args.max_train_steps:
                    path = os.path.join(args.output_dir, 'final_checkpoint')
                    logger.info("Saving final checkpoint to %s", path)
                    tokenizer.save_pretrained(path)
                    accelerator.wait_for_everyone()  # wait for all processes to reach that point in the script
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.lang_model.save_pretrained(path, save_function=accelerator.save)

                    if args.framework.lower() == 'cheatcse':
                        accelerator.save(
                            {
                                'cheat_model': unwrapped_model.cheat_model.state_dict(),
                                'grounding': unwrapped_model.grounding.state_dict()
                            },
                            os.path.join(path, 'cheatcse.pt')
                        )
                    break

    logger.info("Training completed.")



if __name__ == "__main__":
    main()










